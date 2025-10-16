"""
Model evaluation utilities
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from tqdm import tqdm
from typing import Dict, Tuple, List
import pandas as pd


class Evaluator:
    """Evaluator for FusionSentinel model"""
    
    def __init__(self, model: nn.Module, device: str = None):
        """
        Args:
            model: Trained FusionSentinel model
            device: Device to evaluate on
        """
        self.model = model
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions for a dataset.
        
        Args:
            dataloader: DataLoader for the dataset
            
        Returns:
            predictions: (N,) predicted class labels
            probabilities: (N, num_classes) class probabilities
            true_labels: (N,) true labels
        """
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Predicting'):
                network, syscall, syscall_mask, telemetry, labels = batch
                
                # Move to device
                network = network.to(self.device)
                syscall = syscall.to(self.device)
                syscall_mask = syscall_mask.to(self.device)
                telemetry = telemetry.to(self.device)
                
                # Forward pass
                logits, _ = self.model(network, syscall, telemetry, syscall_mask)
                
                # Get predictions and probabilities
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                # Store results
                all_predictions.append(predictions.cpu().numpy())
                all_probabilities.append(probabilities.cpu().numpy())
                all_labels.append(labels.numpy())
        
        # Concatenate all batches
        predictions = np.concatenate(all_predictions)
        probabilities = np.concatenate(all_probabilities)
        true_labels = np.concatenate(all_labels)
        
        return predictions, probabilities, true_labels
    
    def evaluate(self, dataloader: DataLoader, class_names: List[str] = None) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            dataloader: DataLoader for the dataset
            class_names: List of class names for reporting
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("=" * 80)
        print("Evaluating Model")
        print("=" * 80)
        
        # Get predictions
        predictions, probabilities, true_labels = self.predict(dataloader)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(true_labels, predictions, average=None, zero_division=0)
        recall_per_class = recall_score(true_labels, predictions, average=None, zero_division=0)
        f1_per_class = f1_score(true_labels, predictions, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Classification report
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(len(np.unique(true_labels)))]
        
        report = classification_report(true_labels, predictions, target_names=class_names)
        
        # Print results
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        print(f"\nPer-Class Metrics:")
        for i, class_name in enumerate(class_names):
            if i < len(precision_per_class):
                print(f"  {class_name:15s} - P: {precision_per_class[i]:.3f}, "
                      f"R: {recall_per_class[i]:.3f}, F1: {f1_per_class[i]:.3f}")
        
        print(f"\nClassification Report:")
        print(report)
        print("=" * 80)
        
        # Return metrics dictionary
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'confusion_matrix': cm,
            'predictions': predictions,
            'probabilities': probabilities,
            'true_labels': true_labels,
            'classification_report': report
        }
        
        return metrics
    
    def get_attention_weights(self, dataloader: DataLoader, num_samples: int = 10) -> Dict:
        """
        Extract attention weights for visualization.
        
        Args:
            dataloader: DataLoader for the dataset
            num_samples: Number of samples to extract attention for
            
        Returns:
            Dictionary of attention weights and corresponding data
        """
        attention_data = {
            'network_syscall': [],
            'syscall_telemetry': [],
            'predictions': [],
            'true_labels': []
        }
        
        samples_collected = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if samples_collected >= num_samples:
                    break
                
                network, syscall, syscall_mask, telemetry, labels = batch
                
                # Move to device
                network = network.to(self.device)
                syscall = syscall.to(self.device)
                syscall_mask = syscall_mask.to(self.device)
                telemetry = telemetry.to(self.device)
                
                # Forward pass
                logits, attention_weights = self.model(network, syscall, telemetry, syscall_mask)
                predictions = torch.argmax(logits, dim=1)
                
                # Store attention weights
                batch_size = min(num_samples - samples_collected, network.size(0))
                
                attention_data['network_syscall'].append(
                    attention_weights['network_syscall'][:batch_size].cpu().numpy()
                )
                attention_data['syscall_telemetry'].append(
                    attention_weights['syscall_telemetry'][:batch_size].cpu().numpy()
                )
                attention_data['predictions'].append(predictions[:batch_size].cpu().numpy())
                attention_data['true_labels'].append(labels[:batch_size].numpy())
                
                samples_collected += batch_size
        
        # Concatenate all samples
        for key in attention_data:
            attention_data[key] = np.concatenate(attention_data[key])
        
        return attention_data
