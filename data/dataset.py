"""
PyTorch Dataset for multi-modal cyber threat data
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple


class MultiModalDataset(Dataset):
    """
    Dataset for multi-modal cyber threat detection.
    Handles network traffic, system calls, and telemetry data.
    """
    
    def __init__(self, network_data: np.ndarray, syscall_data: np.ndarray, 
                 syscall_masks: np.ndarray, telemetry_data: np.ndarray, 
                 labels: np.ndarray = None):
        """
        Args:
            network_data: (N, network_features) - network traffic features
            syscall_data: (N, seq_len) - system call token IDs
            syscall_masks: (N, seq_len) - attention masks for syscalls
            telemetry_data: (N, telemetry_features) - host telemetry
            labels: (N,) - class labels (optional for inference)
        """
        self.network_data = torch.FloatTensor(network_data)
        self.syscall_data = torch.LongTensor(syscall_data)
        self.syscall_masks = torch.LongTensor(syscall_masks)
        self.telemetry_data = torch.FloatTensor(telemetry_data)
        
        if labels is not None:
            self.labels = torch.LongTensor(labels)
        else:
            self.labels = None
        
        # Ensure network data has sequence dimension for CNN
        if self.network_data.dim() == 2:
            self.network_data = self.network_data.unsqueeze(1)  # (N, 1, features)
    
    def __len__(self) -> int:
        return len(self.network_data)
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Returns:
            network: (1, network_features) or (seq_len, network_features)
            syscall: (seq_len,)
            syscall_mask: (seq_len,)
            telemetry: (telemetry_features,)
            label: scalar (if available)
        """
        network = self.network_data[idx]
        syscall = self.syscall_data[idx]
        syscall_mask = self.syscall_masks[idx]
        telemetry = self.telemetry_data[idx]
        
        if self.labels is not None:
            label = self.labels[idx]
            return network, syscall, syscall_mask, telemetry, label
        
        return network, syscall, syscall_mask, telemetry
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced datasets"""
        if self.labels is None:
            return None
        
        labels_np = self.labels.numpy()
        unique, counts = np.unique(labels_np, return_counts=True)
        
        # Inverse frequency weighting
        weights = 1.0 / counts
        weights = weights / weights.sum() * len(unique)
        
        # Create weight tensor for all classes
        class_weights = torch.zeros(len(unique))
        for idx, weight in zip(unique, weights):
            class_weights[idx] = weight
        
        return class_weights


def create_dataloaders(train_dataset: MultiModalDataset, 
                       val_dataset: MultiModalDataset,
                       test_dataset: MultiModalDataset,
                       batch_size: int = 64,
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation, and test sets.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        
    Returns:
        train_loader, val_loader, test_loader
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
