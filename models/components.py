"""
Neural Network Components for FusionSentinel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NetworkCNN(nn.Module):
    """1D CNN for extracting local patterns from network traffic features"""
    
    def __init__(self, input_dim, channels=[64, 128, 256], kernel_size=3):
        super(NetworkCNN, self).__init__()
        
        self.input_dim = input_dim
        layers = []
        
        in_channels = 1
        for out_channels in channels:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(0.2)
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*layers)
        self.output_dim = channels[-1]
        
    def forward(self, x):
        """
        Args:
            x: (batch, input_dim) - network flow features
        Returns:
            (batch, seq_len', output_dim) - extracted features
        """
        # Reshape for 1D CNN: (batch, 1, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)
        
        x = self.cnn(x)  # (batch, channels, seq_len')
        x = x.transpose(1, 2)  # (batch, seq_len', channels)
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SyscallTransformer(nn.Module):
    """Transformer encoder for modeling system call sequences"""
    
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, num_layers=4, dropout=0.1):
        super(SyscallTransformer, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_dim = embed_dim
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len) - system call token IDs
            mask: (batch, seq_len) - padding mask
        Returns:
            (batch, seq_len, embed_dim) - encoded features
        """
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = self.pos_encoder(x)
        
        # Create attention mask for padding
        if mask is not None:
            mask = mask == 0  # True for padding positions
        
        x = self.transformer(x, src_key_padding_mask=mask)
        
        return x


class TelemetryMLP(nn.Module):
    """MLP for embedding host telemetry features"""
    
    def __init__(self, input_dim, hidden_dims=[128, 256], dropout=0.2):
        super(TelemetryMLP, self).__init__()
        
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
        
    def forward(self, x):
        """
        Args:
            x: (batch, input_dim) - telemetry features
        Returns:
            (batch, output_dim) - embedded features
        """
        return self.mlp(x)


class CrossModalAttention(nn.Module):
    """Cross-modal attention for fusing features from different modalities"""
    
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(CrossModalAttention, self).__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch, seq_len_q, dim)
            key: (batch, seq_len_k, dim)
            value: (batch, seq_len_v, dim)
            mask: attention mask
        Returns:
            output: (batch, seq_len_q, dim)
            attention_weights: (batch, num_heads, seq_len_q, seq_len_k)
        """
        # Multi-head attention
        attn_output, attn_weights = self.multihead_attn(
            query, key, value, attn_mask=mask, need_weights=True
        )
        
        # Residual connection and normalization
        query = self.norm1(query + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(query)
        output = self.norm2(query + ffn_output)
        
        return output, attn_weights


class FusionBiLSTM(nn.Module):
    """Bidirectional LSTM for temporal reasoning over fused features"""
    
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.2):
        super(FusionBiLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        self.output_dim = hidden_dim * 2  # Bidirectional
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            output: (batch, seq_len, hidden_dim*2)
            (h_n, c_n): final hidden and cell states
        """
        output, (h_n, c_n) = self.lstm(x)
        return output, (h_n, c_n)
