"""
Training script for NER model.
Includes training loop, validation, checkpointing, and early stopping.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from model import create_model
from dataset import create_dataloader, get_tokenizer
from labels import NUM_LABELS, ID2LABEL


class Trainer:
    """
    Trainer class for NER model.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        learning_rate: float = 2e-5,
        num_epochs: int = 5,
        device: str = None,
        output_dir: str = "out",
        patience: int = 3
    ):
        """
        Args:
            model: NER model
            train_loader: Training data loader
            dev_loader: Development data loader
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            device: Device to train on (cpu/cuda/mps)
            output_dir: Directory to save checkpoints
            patience: Early stopping patience
        """
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.num_epochs = num_epochs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.patience = patience
        
        # Device setup
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        num_training_steps = len(train_loader) * num_epochs
        num_warmup_steps = num_training_steps // 10  # 10% warmup
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Training state
        self.best_dev_f1 = 0.0
        self.epochs_without_improvement = 0
        self.global_step = 0
        self.train_losses = []
        self.dev_metrics = []
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            loss, logits = self.model(input_ids, attention_mask, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate on development set.
        Uses simple token-level accuracy for quick evaluation during training.
        
        Returns:
            Dictionary with metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.dev_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                loss, logits = self.model(input_ids, attention_mask, labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                
                # Collect predictions and labels (excluding -100)
                mask = labels != -100
                all_predictions.extend(predictions[mask].cpu().tolist())
                all_labels.extend(labels[mask].cpu().tolist())
        
        # Calculate metrics
        avg_loss = total_loss / num_batches
        
        # Token-level accuracy
        correct = sum([p == l for p, l in zip(all_predictions, all_labels)])
        total = len(all_predictions)
        accuracy = correct / total if total > 0 else 0.0
        
        # Simple F1 approximation (treating each label independently)
        # For proper span-level F1, use eval_span_f1.py after training
        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1
        }
        
        return metrics
    
    def train(self):
        """
        Full training loop with validation and checkpointing.
        """
        print(f"\nStarting training for {self.num_epochs} epochs...")
        print(f"Training examples: {len(self.train_loader.dataset)}")
        print(f"Dev examples: {len(self.dev_loader.dataset)}")
        print(f"Batch size: {self.train_loader.batch_size}")
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print(f"{'='*60}")
            
            # Training
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            print(f"Average training loss: {train_loss:.4f}")
            
            # Evaluation
            dev_metrics = self.evaluate()
            self.dev_metrics.append(dev_metrics)
            
            print(f"Dev loss: {dev_metrics['loss']:.4f}")
            print(f"Dev accuracy: {dev_metrics['accuracy']:.4f}")
            print(f"Dev F1 (weighted): {dev_metrics['f1']:.4f}")
            
            # Save checkpoint if best model
            if dev_metrics['f1'] > self.best_dev_f1:
                self.best_dev_f1 = dev_metrics['f1']
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch, is_best=True)
                print(f"✓ New best model! F1: {self.best_dev_f1:.4f}")
            else:
                self.epochs_without_improvement += 1
                print(f"No improvement for {self.epochs_without_improvement} epoch(s)")
            
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Training complete
        elapsed_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Time elapsed: {elapsed_time / 60:.2f} minutes")
        print(f"Best dev F1: {self.best_dev_f1:.4f}")
        print(f"Model saved to: {self.output_dir / 'best_model.pt'}")
        
        # Save training history
        self.save_training_history()
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_dev_f1': self.best_dev_f1,
            'train_losses': self.train_losses,
            'dev_metrics': self.dev_metrics
        }
        
        if is_best:
            checkpoint_path = self.output_dir / 'best_model.pt'
            torch.save(checkpoint, checkpoint_path)
    
    def save_training_history(self):
        """Save training history to JSON."""
        history = {
            'train_losses': self.train_losses,
            'dev_metrics': self.dev_metrics,
            'best_dev_f1': self.best_dev_f1
        }
        
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Training history saved to: {history_path}")


def main():
    """Main training function."""
    
    # Hyperparameters - OPTIMIZED FOR SPEED
    MODEL_NAME = "distilbert-base-uncased"
    BATCH_SIZE = 32  # Larger batches = faster
    MAX_LENGTH = 128
    LEARNING_RATE = 3e-5  # Higher LR = faster convergence
    NUM_EPOCHS = 2  # Reduced from 5 to 2
    PATIENCE = 2  # Reduced patience
    OUTPUT_DIR = "out"
    
    print("="*60)
    print("NER Model Training")
    print("="*60)
    
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = get_tokenizer(MODEL_NAME)
    
    # Create data loaders
    print("\n2. Creating data loaders...")
    train_loader = create_dataloader(
        "data/train.jsonl",
        tokenizer,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        shuffle=True
    )
    
    dev_loader = create_dataloader(
        "data/dev.jsonl",
        tokenizer,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        shuffle=False
    )
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Dev batches: {len(dev_loader)}")
    
    # Create model
    print("\n3. Creating model...")
    model = create_model(MODEL_NAME)
    
    # Create trainer
    print("\n4. Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        output_dir=OUTPUT_DIR,
        patience=PATIENCE
    )
    
    # Train
    print("\n5. Starting training...")
    trainer.train()
    
    print("\n✓ Training pipeline complete!")


if __name__ == "__main__":
    main()