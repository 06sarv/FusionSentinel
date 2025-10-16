"""
Training loop for FusionSentinel
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from typing import Dict, Tuple
import os
import time

from .callbacks import EarlyStopping, ModelCheckpoint


class Trainer:
    """Trainer for FusionSentinel model"""
    
    def __init__(self, model: nn.Module, config: Dict, device: str = None):
        """
        Args:
            model: FusionSentinel model
            config: Configuration dictionary
            device: Device to train on ('cuda' or 'cpu')
        """
        self.model = model
        self.config = config
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Training config
        train_cfg = config['training']
        self.epochs = train_cfg['epochs']
        self.learning_rate = train_cfg['learning_rate']
        self.weight_decay = train_cfg['weight_decay']
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        if train_cfg['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=train_cfg['t_0'],
                T_mult=train_cfg['t_mult']
            )
        else:
            self.scheduler = None
        
        # Loss function (will be set with class weights if provided)
        self.criterion = None
        
        # Callbacks
        self.early_stopping = EarlyStopping(
            patience=train_cfg['patience'],
            min_delta=train_cfg['min_delta'],
            mode='min'
        )
        
        self.checkpoint = ModelCheckpoint(
            checkpoint_dir=train_cfg['checkpoint_dir'],
            save_best_only=train_cfg['save_best_only'],
            mode='min',
            verbose=True
        )
        
        # Tensorboard
        self.writer = SummaryWriter(log_dir='runs/fusion_sentinel')
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def set_class_weights(self, class_weights: torch.Tensor):
        """Set class weights for imbalanced datasets"""
        class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"✓ Class weights set: {class_weights.cpu().numpy()}")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            avg_loss, avg_accuracy
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training', leave=False)
        
        for batch_idx, (network, syscall, syscall_mask, telemetry, labels) in enumerate(pbar):
            # Move to device
            network = network.to(self.device)
            syscall = syscall.to(self.device)
            syscall_mask = syscall_mask.to(self.device)
            telemetry = telemetry.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, _ = self.model(network, syscall, telemetry, syscall_mask)
            
            # Compute loss
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = 100.0 * correct / total
        
        return avg_loss, avg_acc
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model.
        
        Returns:
            avg_loss, avg_accuracy
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation', leave=False)
            
            for network, syscall, syscall_mask, telemetry, labels in pbar:
                # Move to device
                network = network.to(self.device)
                syscall = syscall.to(self.device)
                syscall_mask = syscall_mask.to(self.device)
                telemetry = telemetry.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits, _ = self.model(network, syscall, telemetry, syscall_mask)
                
                # Compute loss
                loss = self.criterion(logits, labels)
                
                # Metrics
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.0 * correct / total:.2f}%'
                })
        
        avg_loss = total_loss / len(val_loader)
        avg_acc = 100.0 * correct / total
        
        return avg_loss, avg_acc
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        # Set criterion if not already set
        if self.criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        
        print("=" * 80)
        print("Starting Training")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Epochs: {self.epochs}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Batch Size: {self.config['training']['batch_size']}")
        print(f"Model Parameters: {self.model.count_parameters():,}")
        print("=" * 80)
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Tensorboard logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch}/{self.epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            
            # Save checkpoint
            metrics = {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            }
            self.checkpoint(self.model, self.optimizer, self.scheduler, epoch, val_loss, metrics)
            
            # Check for best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"  ★ New best validation loss: {val_loss:.4f}")
            
            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\n⚠ Early stopping triggered after {epoch} epochs")
                break
        
        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print(f"Training Complete! Total time: {total_time/60:.1f} minutes")
        print(f"Best Validation Loss: {best_val_loss:.4f}")
        print("=" * 80)
        
        self.writer.close()
    
    def load_checkpoint(self, checkpoint_path: str = None):
        """Load a saved checkpoint"""
        epoch, score, metrics = self.checkpoint.load_checkpoint(
            self.model, self.optimizer, self.scheduler, checkpoint_path
        )
        print(f"Loaded checkpoint from epoch {epoch} with score {score:.4f}")
        return epoch, score, metrics
