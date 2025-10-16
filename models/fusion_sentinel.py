"""
FusionSentinel: Multi-Modal Cyber Threat Detection Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import (
    NetworkCNN,
    SyscallTransformer,
    TelemetryMLP,
    CrossModalAttention,
    FusionBiLSTM
)


class FusionSentinel(nn.Module):
    """
    Multi-modal deep learning architecture for cyber threat detection.
    
    Combines:
    - 1D CNN for network traffic features
    - Transformer for system call sequences
    - MLP for host telemetry
    - Cross-modal attention for feature fusion
    - BiLSTM for temporal reasoning
    """
    
    def __init__(self, config):
        super(FusionSentinel, self).__init__()
        
        self.config = config
        
        # Extract configuration
        model_cfg = config['model']
        
        # Network traffic CNN
        self.network_cnn = NetworkCNN(
            input_dim=model_cfg['network_input_dim'],
            channels=model_cfg['cnn_channels'],
            kernel_size=model_cfg['cnn_kernel_size']
        )
        
        # System call Transformer
        self.syscall_transformer = SyscallTransformer(
            vocab_size=model_cfg['syscall_vocab_size'],
            embed_dim=model_cfg['syscall_embed_dim'],
            num_heads=model_cfg['transformer_heads'],
            num_layers=model_cfg['transformer_layers'],
            dropout=model_cfg['transformer_dropout']
        )
        
        # Host telemetry MLP
        self.telemetry_mlp = TelemetryMLP(
            input_dim=model_cfg['telemetry_input_dim'],
            hidden_dims=model_cfg['mlp_hidden_dims'],
            dropout=0.2
        )
        
        # Projection layers to align dimensions
        attention_dim = model_cfg['attention_dim']
        self.network_proj = nn.Linear(self.network_cnn.output_dim, attention_dim)
        self.syscall_proj = nn.Linear(self.syscall_transformer.output_dim, attention_dim)
        self.telemetry_proj = nn.Linear(self.telemetry_mlp.output_dim, attention_dim)
        
        # Cross-modal attention layers
        self.cross_attention_net_sys = CrossModalAttention(
            dim=attention_dim,
            num_heads=model_cfg['attention_heads'],
            dropout=0.1
        )
        
        self.cross_attention_sys_tel = CrossModalAttention(
            dim=attention_dim,
            num_heads=model_cfg['attention_heads'],
            dropout=0.1
        )
        
        # Fusion BiLSTM
        self.fusion_lstm = FusionBiLSTM(
            input_dim=attention_dim,
            hidden_dim=model_cfg['lstm_hidden_dim'],
            num_layers=model_cfg['lstm_layers'],
            dropout=model_cfg['lstm_dropout']
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_lstm.output_dim, model_cfg['classifier_hidden']),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(model_cfg['classifier_hidden'], model_cfg['num_classes'])
        )
        
        # Store attention weights for visualization
        self.attention_weights = {}
        
    def forward(self, network_data, syscall_data, telemetry_data, syscall_mask=None):
        """
        Forward pass through the multi-modal architecture.
        
        Args:
            network_data: (batch, seq_len, network_features) - network flow data
            syscall_data: (batch, syscall_seq_len) - system call token IDs
            telemetry_data: (batch, telemetry_features) - host telemetry
            syscall_mask: (batch, syscall_seq_len) - padding mask for syscalls
            
        Returns:
            logits: (batch, num_classes) - classification logits
            attention_weights: dict of attention weight tensors
        """
        batch_size = network_data.size(0)
        
        # Extract features from each modality
        # Network traffic: CNN
        net_features = self.network_cnn(network_data)  # (batch, seq_len', cnn_dim)
        net_features = self.network_proj(net_features)  # (batch, seq_len', attention_dim)
        
        # System calls: Transformer
        sys_features = self.syscall_transformer(syscall_data, syscall_mask)  # (batch, syscall_seq_len, transformer_dim)
        sys_features = self.syscall_proj(sys_features)  # (batch, syscall_seq_len, attention_dim)
        
        # Telemetry: MLP
        tel_features = self.telemetry_mlp(telemetry_data)  # (batch, mlp_dim)
        tel_features = self.telemetry_proj(tel_features)  # (batch, attention_dim)
        tel_features = tel_features.unsqueeze(1)  # (batch, 1, attention_dim)
        
        # Cross-modal attention fusion
        # Attend syscalls to network features
        fused_net_sys, attn_net_sys = self.cross_attention_net_sys(
            query=sys_features,
            key=net_features,
            value=net_features
        )  # (batch, syscall_seq_len, attention_dim)
        
        # Attend fused features to telemetry
        fused_all, attn_sys_tel = self.cross_attention_sys_tel(
            query=fused_net_sys,
            key=tel_features,
            value=tel_features
        )  # (batch, syscall_seq_len, attention_dim)
        
        # Store attention weights for visualization
        self.attention_weights = {
            'network_syscall': attn_net_sys.detach(),
            'syscall_telemetry': attn_sys_tel.detach()
        }
        
        # Temporal reasoning with BiLSTM
        lstm_out, _ = self.fusion_lstm(fused_all)  # (batch, seq_len, lstm_dim*2)
        
        # Global pooling (mean over sequence)
        pooled = lstm_out.mean(dim=1)  # (batch, lstm_dim*2)
        
        # Classification
        logits = self.classifier(pooled)  # (batch, num_classes)
        
        return logits, self.attention_weights
    
    def get_attention_weights(self):
        """Return the most recent attention weights for visualization"""
        return self.attention_weights
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def summary(self):
        """Print model summary"""
        print("=" * 80)
        print("FusionSentinel Model Summary")
        print("=" * 80)
        print(f"Total Parameters: {self.count_parameters():,}")
        print("\nArchitecture:")
        print(f"  Network CNN: {self.network_cnn.output_dim} features")
        print(f"  Syscall Transformer: {self.syscall_transformer.output_dim} features")
        print(f"  Telemetry MLP: {self.telemetry_mlp.output_dim} features")
        print(f"  Attention Dimension: {self.config['model']['attention_dim']}")
        print(f"  LSTM Hidden: {self.config['model']['lstm_hidden_dim']} x 2 (bidirectional)")
        print(f"  Output Classes: {self.config['model']['num_classes']}")
        print("=" * 80)
