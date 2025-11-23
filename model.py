"""
NER Model architecture using DistilBERT for token classification.
Optimized for low latency while maintaining good performance.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForTokenClassification, AutoConfig

from labels import NUM_LABELS, ID2LABEL, LABEL2ID


class NERModel(nn.Module):
    """
    Token classification model for NER.
    Uses DistilBERT as the base encoder for efficiency.
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = NUM_LABELS,
        dropout: float = 0.1
    ):
        """
        Args:
            model_name: HuggingFace model name
            num_labels: Number of NER labels (15 for our BIO scheme)
            dropout: Dropout rate for classification head
        """
        super(NERModel, self).__init__()
        
        # Load config
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=ID2LABEL,
            label2id=LABEL2ID
        )
        
        # Load pre-trained model
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True  # In case of label size mismatch
        )
        
        # Additional dropout for regularization
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None
    ):
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            labels: Ground truth labels (batch_size, seq_len), optional
            
        Returns:
            If labels provided: (loss, logits)
            If labels not provided: logits
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        if labels is not None:
            return outputs.loss, outputs.logits
        else:
            return outputs.logits
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ):
        """
        Predict labels for given inputs.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            
        Returns:
            Predicted label IDs (batch_size, seq_len)
        """
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=-1)
        
        return predictions


def create_model(
    model_name: str = "distilbert-base-uncased",
    dropout: float = 0.1
) -> NERModel:
    """
    Create NER model.
    
    Args:
        model_name: HuggingFace model name
        dropout: Dropout rate
        
    Returns:
        NERModel instance
    """
    model = NERModel(model_name=model_name, dropout=dropout)
    return model


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing NER Model...")
    
    # Create model
    print("\n1. Creating model...")
    model = create_model("distilbert-base-uncased")
    
    num_params = count_parameters(model)
    print(f"   Model created: {model.__class__.__name__}")
    print(f"   Trainable parameters: {num_params:,}")
    print(f"   Model size: ~{num_params * 4 / (1024**2):.1f} MB")
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    batch_size = 2
    seq_len = 32
    
    # Create dummy inputs
    input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, NUM_LABELS, (batch_size, seq_len))
    # Set some labels to -100 (ignore)
    labels[:, :2] = -100  # Ignore first 2 tokens (like [CLS])
    
    # Forward pass with labels (training mode)
    print("   Running forward pass with labels...")
    loss, logits = model(input_ids, attention_mask, labels)
    
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Expected shape: (batch_size={batch_size}, seq_len={seq_len}, num_labels={NUM_LABELS})")
    
    # Forward pass without labels (inference mode)
    print("\n3. Testing prediction...")
    predictions = model.predict(input_ids, attention_mask)
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Sample predictions (first 10 tokens): {predictions[0, :10].tolist()}")
    
    # Decode predictions
    print("\n4. Decoding predictions...")
    from labels import ID2LABEL
    sample_pred_labels = [ID2LABEL[pred.item()] for pred in predictions[0, :10]]
    print(f"   Decoded labels: {sample_pred_labels}")
    
    print("\n✓ Model test complete!")
    
    # Model selection info
    print("\n--- Model Selection Notes ---")
    print("DistilBERT chosen for:")
    print("  • 40% smaller than BERT-base")
    print("  • 60% faster inference")
    print("  • 97% of BERT's performance")
    print("  • Better for latency constraint (≤20ms)")