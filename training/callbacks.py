"""
Training callbacks for FusionSentinel
"""

import torch
import os
import numpy as np


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.001, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value (loss or accuracy)
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        # Check for improvement
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False
    
    def reset(self):
        """Reset the early stopping state"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class ModelCheckpoint:
    """Save model checkpoints during training"""
    
    def __init__(self, checkpoint_dir: str, save_best_only: bool = True, 
                 mode: str = 'min', verbose: bool = True):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_best_only: Only save when metric improves
            mode: 'min' for loss, 'max' for accuracy
            verbose: Print messages
        """
        self.checkpoint_dir = checkpoint_dir
        self.save_best_only = save_best_only
        self.mode = mode
        self.verbose = verbose
        self.best_score = None
        
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def __call__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 scheduler, epoch: int, score: float, metrics: dict = None):
        """
        Save checkpoint if conditions are met.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler
            epoch: Current epoch
            score: Current metric value
            metrics: Additional metrics to save
        """
        should_save = False
        
        if not self.save_best_only:
            should_save = True
        else:
            if self.best_score is None:
                should_save = True
            elif self.mode == 'min' and score < self.best_score:
                should_save = True
            elif self.mode == 'max' and score > self.best_score:
                should_save = True
        
        if should_save:
            self.best_score = score
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'score': score,
                'metrics': metrics
            }
            
            # Save best model
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            
            if self.verbose:
                print(f"✓ Checkpoint saved: {best_path} (score: {score:.4f})")
        
        # Always save last model
        last_path = os.path.join(self.checkpoint_dir, 'last_model.pth')
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'score': score,
            'metrics': metrics
        }
        torch.save(checkpoint, last_path)
    
    def load_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None,
                       scheduler=None, checkpoint_path: str = None):
        """
        Load a checkpoint.
        
        Args:
            model: Model to load weights into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            checkpoint_path: Path to checkpoint (default: best_model.pth)
            
        Returns:
            epoch, score, metrics
        """
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.verbose:
            print(f"✓ Checkpoint loaded: {checkpoint_path}")
        
        return checkpoint['epoch'], checkpoint['score'], checkpoint.get('metrics', {})
