"""
Visualization utilities for FusionSentinel
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List
import os


# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class AttentionVisualizer:
    """Visualize attention weights from the model"""
    
    def __init__(self, save_dir: str = 'visualizations'):
        """
        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_attention_heatmap(self, attention_weights: np.ndarray, 
                               title: str = "Attention Weights",
                               save_name: str = None):
        """
        Plot attention heatmap.
        
        Args:
            attention_weights: (num_heads, seq_len_q, seq_len_k) or (seq_len_q, seq_len_k)
            title: Plot title
            save_name: Filename to save (optional)
        """
        # Average over heads if multi-head
        if attention_weights.ndim == 3:
            attention_weights = attention_weights.mean(axis=0)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention_weights, cmap='viridis', cbar=True, 
                   xticklabels=False, yticklabels=False)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Key Sequence Position')
        plt.ylabel('Query Sequence Position')
        plt.tight_layout()
        
        if save_name:
            save_path = os.path.join(self.save_dir, save_name)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        plt.close()
    
    def plot_attention_comparison(self, attention_data: Dict, 
                                  sample_idx: int = 0,
                                  class_names: List[str] = None):
        """
        Plot attention weights for different modality interactions.
        
        Args:
            attention_data: Dictionary with attention weights
            sample_idx: Index of sample to visualize
            class_names: List of class names
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Network-Syscall attention
        net_sys_attn = attention_data['network_syscall'][sample_idx]
        if net_sys_attn.ndim == 3:
            net_sys_attn = net_sys_attn.mean(axis=0)
        
        sns.heatmap(net_sys_attn, ax=axes[0], cmap='viridis', cbar=True,
                   xticklabels=False, yticklabels=False)
        axes[0].set_title('Network ← Syscall Attention', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Network Features')
        axes[0].set_ylabel('Syscall Sequence')
        
        # Syscall-Telemetry attention
        sys_tel_attn = attention_data['syscall_telemetry'][sample_idx]
        if sys_tel_attn.ndim == 3:
            sys_tel_attn = sys_tel_attn.mean(axis=0)
        
        sns.heatmap(sys_tel_attn, ax=axes[1], cmap='plasma', cbar=True,
                   xticklabels=False, yticklabels=False)
        axes[1].set_title('Syscall ← Telemetry Attention', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Telemetry Features')
        axes[1].set_ylabel('Syscall Sequence')
        
        # Add prediction info
        pred = attention_data['predictions'][sample_idx]
        true = attention_data['true_labels'][sample_idx]
        
        if class_names:
            pred_name = class_names[pred] if pred < len(class_names) else f"Class_{pred}"
            true_name = class_names[true] if true < len(class_names) else f"Class_{true}"
        else:
            pred_name = f"Class_{pred}"
            true_name = f"Class_{true}"
        
        fig.suptitle(f'Sample {sample_idx} | Predicted: {pred_name} | True: {true_name}',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'attention_sample_{sample_idx}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        
        plt.close()
    
    def plot_attention_distribution(self, attention_data: Dict, num_samples: int = 10):
        """
        Plot distribution of attention weights across samples.
        
        Args:
            attention_data: Dictionary with attention weights
            num_samples: Number of samples to analyze
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Network-Syscall attention distribution
        net_sys_attns = []
        for i in range(min(num_samples, len(attention_data['network_syscall']))):
            attn = attention_data['network_syscall'][i]
            if attn.ndim == 3:
                attn = attn.mean(axis=0)
            net_sys_attns.append(attn.flatten())
        
        net_sys_attns = np.concatenate(net_sys_attns)
        axes[0].hist(net_sys_attns, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].set_title('Network-Syscall Attention Distribution', fontweight='bold')
        axes[0].set_xlabel('Attention Weight')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
        
        # Syscall-Telemetry attention distribution
        sys_tel_attns = []
        for i in range(min(num_samples, len(attention_data['syscall_telemetry']))):
            attn = attention_data['syscall_telemetry'][i]
            if attn.ndim == 3:
                attn = attn.mean(axis=0)
            sys_tel_attns.append(attn.flatten())
        
        sys_tel_attns = np.concatenate(sys_tel_attns)
        axes[1].hist(sys_tel_attns, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
        axes[1].set_title('Syscall-Telemetry Attention Distribution', fontweight='bold')
        axes[1].set_xlabel('Attention Weight')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, 'attention_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        
        plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                         save_path: str = None, normalize: bool = True):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save figure
        normalize: Whether to normalize the matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names,
               cbar_kws={'label': 'Proportion' if normalize else 'Count'})
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.close()


def plot_training_history(history: Dict, save_path: str = None):
    """
    Plot training history (loss and accuracy curves).
    
    Args:
        history: Dictionary with training history
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.close()


def plot_per_class_metrics(metrics: Dict, class_names: List[str], save_path: str = None):
    """
    Plot per-class precision, recall, and F1-score.
    
    Args:
        metrics: Dictionary with per-class metrics
        class_names: List of class names
        save_path: Path to save figure
    """
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.bar(x - width, metrics['precision_per_class'], width, label='Precision', color='skyblue')
    ax.bar(x, metrics['recall_per_class'], width, label='Recall', color='lightcoral')
    ax.bar(x + width, metrics['f1_per_class'], width, label='F1-Score', color='lightgreen')
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.close()
