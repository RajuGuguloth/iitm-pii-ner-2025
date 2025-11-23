"""
Prediction script for NER model.
Loads trained model and generates predictions on test data.
Converts token-level predictions to character-level spans.
"""

import json
import torch
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

from model import create_model
from dataset import get_tokenizer, NERDataset
from labels import ID2LABEL, LABEL2ID, get_entity_type


def load_model(checkpoint_path: str, device: str = None):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    # Device setup
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = create_model("distilbert-base-uncased")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from: {checkpoint_path}")
    print(f"Best dev F1: {checkpoint['best_dev_f1']:.4f}")
    print(f"Device: {device}")
    
    return model, device


def align_predictions_to_chars(
    text: str,
    token_predictions: List[int],
    offset_mapping: List[Tuple[int, int]]
) -> List[Dict]:
    """
    Convert token-level BIO predictions to character-level entity spans.
    
    Args:
        text: Original text
        token_predictions: Predicted label IDs for each token
        offset_mapping: Character offsets for each token
        
    Returns:
        List of entity dicts with 'start', 'end', 'label'
    """
    entities = []
    current_entity = None
    
    for token_idx, (pred_id, (start, end)) in enumerate(zip(token_predictions, offset_mapping)):
        # Skip special tokens and padding
        if start == 0 and end == 0:
            continue
        
        # Get BIO label
        bio_label = ID2LABEL[pred_id]
        
        # Skip 'O' (Outside) labels
        if bio_label == 'O':
            # Save current entity if exists
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None
            continue
        
        # Parse BIO tag
        tag, entity_type = bio_label.split('-', 1)
        
        if tag == 'B':
            # Beginning of new entity
            # Save previous entity if exists
            if current_entity is not None:
                entities.append(current_entity)
            
            # Start new entity
            current_entity = {
                'start': start,
                'end': end,
                'label': entity_type
            }
        
        elif tag == 'I':
            # Inside entity
            if current_entity is not None and current_entity['label'] == entity_type:
                # Extend current entity
                current_entity['end'] = end
            else:
                # I- tag without B- tag (treat as new entity)
                if current_entity is not None:
                    entities.append(current_entity)
                
                current_entity = {
                    'start': start,
                    'end': end,
                    'label': entity_type
                }
    
    # Don't forget the last entity
    if current_entity is not None:
        entities.append(current_entity)
    
    return entities


def predict_on_text(
    text: str,
    model,
    tokenizer,
    device,
    max_length: int = 128
) -> List[Dict]:
    """
    Predict entities for a single text.
    
    Args:
        text: Input text
        model: Trained NER model
        tokenizer: Tokenizer
        device: Device
        max_length: Maximum sequence length
        
    Returns:
        List of predicted entities
    """
    # Tokenize
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_offsets_mapping=True,
        return_tensors='pt'
    )
    
    # Extract offset mapping
    offset_mapping = encoding.pop('offset_mapping').squeeze(0).tolist()
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        predictions = model.predict(input_ids, attention_mask)
    
    # Convert to list
    predictions = predictions.squeeze(0).cpu().tolist()
    
    # Align predictions to character spans
    entities = align_predictions_to_chars(text, predictions, offset_mapping)
    
    return entities


def predict_on_file(
    input_path: str,
    output_path: str,
    model,
    tokenizer,
    device,
    max_length: int = 128
):
    """
    Predict entities for all examples in a JSONL file.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        model: Trained NER model
        tokenizer: Tokenizer
        device: Device
        max_length: Maximum sequence length
    """
    # Load input data
    with open(input_path, 'r', encoding='utf-8') as f:
        examples = [json.loads(line) for line in f]
    
    print(f"Predicting on {len(examples)} examples...")
    
    # Predict for each example
    predictions = []
    for example in tqdm(examples, desc="Predicting"):
        text = example['text']
        example_id = example['id']
        
        # Predict entities
        entities = predict_on_text(text, model, tokenizer, device, max_length)
        
        # Create prediction
        prediction = {
            'id': example_id,
            'text': text,
            'entities': entities
        }
        
        predictions.append(prediction)
    
    # Save predictions
    with open(output_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')
    
    print(f"Predictions saved to: {output_path}")
    
    return predictions


def main():
    """Main prediction function."""
    
    # Configuration
    CHECKPOINT_PATH = "out/best_model.pt"
    TEST_DATA_PATH = "data/test.jsonl"
    OUTPUT_PATH = "out/predictions.jsonl"
    MODEL_NAME = "distilbert-base-uncased"
    MAX_LENGTH = 128
    
    print("="*60)
    print("NER Model Prediction")
    print("="*60)
    
    # Check if checkpoint exists
    if not Path(CHECKPOINT_PATH).exists():
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        print("Please train the model first using train.py")
        return
    
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = get_tokenizer(MODEL_NAME)
    
    # Load model
    print("\n2. Loading model...")
    model, device = load_model(CHECKPOINT_PATH)
    
    # Run predictions
    print("\n3. Running predictions...")
    predictions = predict_on_file(
        TEST_DATA_PATH,
        OUTPUT_PATH,
        model,
        tokenizer,
        device,
        MAX_LENGTH
    )
    
    # Show sample predictions
    print("\n4. Sample predictions:")
    for i, pred in enumerate(predictions[:3]):
        print(f"\nExample {i+1}:")
        print(f"  Text: {pred['text'][:80]}...")
        print(f"  Entities: {len(pred['entities'])} found")
        for entity in pred['entities'][:3]:
            entity_text = pred['text'][entity['start']:entity['end']]
            print(f"    - {entity['label']}: \"{entity_text}\"")
    
    print("\n✓ Prediction complete!")


if __name__ == "__main__":
    main()