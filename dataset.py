"""
Dataset loading and preprocessing for NER task.
Handles JSONL reading, tokenization, and label alignment.
"""

import json
from typing import List, Dict, Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from labels import align_labels_with_tokens, get_label_ids, LABEL2ID, NUM_LABELS


class NERDataset(Dataset):
    """
    PyTorch Dataset for NER task.
    Loads JSONL data and prepares tokenized inputs with aligned labels.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 128
    ):
        """
        Args:
            data_path: Path to JSONL file
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """Load examples from JSONL file."""
        examples = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                examples.append(data)
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example with tokenized inputs and aligned labels.
        
        Returns:
            Dict with 'input_ids', 'attention_mask', 'labels', 'example_id'
        """
        example = self.examples[idx]
        
        text = example['text']
        entities = example.get('entities', [])
        example_id = example['id']
        
        # Tokenize with offset mapping to track character positions
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        
        # Extract offset mapping and remove it from encoding
        offset_mapping = encoding.pop('offset_mapping').squeeze(0)
        
        # Get token offsets (exclude special tokens and padding)
        token_offsets = []
        valid_token_indices = []
        
        for token_idx, (start, end) in enumerate(offset_mapping.tolist()):
            # Skip special tokens ([CLS], [SEP]) and padding (offset = (0, 0))
            if start == 0 and end == 0:
                continue
            token_offsets.append((start, end))
            valid_token_indices.append(token_idx)
        
        # Align labels with tokens
        bio_labels = align_labels_with_tokens(text, entities, token_offsets)
        label_ids = get_label_ids(bio_labels)
        
        # Create full label sequence (with -100 for special tokens and padding)
        labels = torch.full((self.max_length,), -100, dtype=torch.long)
        
        # Assign labels to valid tokens
        for i, token_idx in enumerate(valid_token_indices):
            if i < len(label_ids):
                labels[token_idx] = label_ids[i]
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels,
            'example_id': example_id,
            'text': text  # Keep original text for debugging
        }


def create_dataloader(
    data_path: str,
    tokenizer: AutoTokenizer,
    batch_size: int = 16,
    max_length: int = 128,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Create a DataLoader for NER dataset.
    
    Args:
        data_path: Path to JSONL file
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        shuffle: Whether to shuffle data
        num_workers: Number of workers for data loading
        
    Returns:
        DataLoader instance
    """
    dataset = NERDataset(data_path, tokenizer, max_length)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


def get_tokenizer(model_name: str = "bert-base-uncased") -> AutoTokenizer:
    """
    Load tokenizer.
    
    Args:
        model_name: HuggingFace model name
        
    Returns:
        Tokenizer instance
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


if __name__ == "__main__":
    # Test dataset loading
    print("Testing NER Dataset...")
    
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = get_tokenizer("bert-base-uncased")
    print(f"   Tokenizer loaded: {tokenizer.__class__.__name__}")
    
    # Load dataset
    print("\n2. Loading train dataset...")
    train_path = "data/train.jsonl"
    
    if Path(train_path).exists():
        dataset = NERDataset(train_path, tokenizer, max_length=128)
        print(f"   Dataset size: {len(dataset)}")
        
        # Test single example
        print("\n3. Testing single example...")
        example = dataset[0]
        
        print(f"   Example ID: {example['example_id']}")
        print(f"   Text: {example['text']}")
        print(f"   Input IDs shape: {example['input_ids'].shape}")
        print(f"   Attention mask shape: {example['attention_mask'].shape}")
        print(f"   Labels shape: {example['labels'].shape}")
        
        # Decode tokens and show labels
        print("\n4. Token-Label alignment:")
        tokens = tokenizer.convert_ids_to_tokens(example['input_ids'])
        labels = example['labels'].tolist()
        
        for i, (token, label_id) in enumerate(zip(tokens[:20], labels[:20])):
            if label_id != -100:
                from labels import ID2LABEL
                label = ID2LABEL[label_id]
                print(f"   {i:2d}. {token:15s} -> {label}")
        
        # Test DataLoader
        print("\n5. Testing DataLoader...")
        dataloader = create_dataloader(
            train_path,
            tokenizer,
            batch_size=4,
            shuffle=False
        )
        
        batch = next(iter(dataloader))
        print(f"   Batch input_ids shape: {batch['input_ids'].shape}")
        print(f"   Batch labels shape: {batch['labels'].shape}")
        print(f"   Batch size: {len(batch['example_id'])}")
        
        print("\n✓ Dataset loading test complete!")
    else:
        print(f"   Error: {train_path} not found!")
        print("   Please run generate_data.py first.")