"""
Data preprocessing utilities for multi-modal cyber threat detection
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class NetworkPreprocessor:
    """Preprocessor for network traffic data (CICIDS2017)"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        
    def fit(self, data: pd.DataFrame) -> 'NetworkPreprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            data: DataFrame with network features
        """
        # Remove infinite values and NaNs
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.fillna(0)
        
        # Store feature names
        self.feature_names = data.columns.tolist()
        
        # Fit scaler
        self.scaler.fit(data)
        self.is_fitted = True
        
        return self
    
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Transform network data.
        
        Args:
            data: DataFrame with network features
            
        Returns:
            Normalized numpy array
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Handle missing values
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.fillna(0)
        
        # Scale features
        scaled_data = self.scaler.transform(data)
        
        return scaled_data.astype(np.float32)
    
    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(data).transform(data)
    
    def save(self, path: str):
        """Save preprocessor state"""
        joblib.dump({
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }, path)
    
    def load(self, path: str):
        """Load preprocessor state"""
        state = joblib.load(path)
        self.scaler = state['scaler']
        self.feature_names = state['feature_names']
        self.is_fitted = state['is_fitted']


class SyscallPreprocessor:
    """Preprocessor for system call sequences (ADFA-LD)"""
    
    def __init__(self, max_vocab_size: int = 500, max_seq_len: int = 200):
        self.max_vocab_size = max_vocab_size
        self.max_seq_len = max_seq_len
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        self.reverse_vocab = {0: '<PAD>', 1: '<UNK>'}
        self.is_fitted = False
        
    def fit(self, sequences: List[List[str]]) -> 'SyscallPreprocessor':
        """
        Build vocabulary from system call sequences.
        
        Args:
            sequences: List of system call sequences
        """
        # Count syscall frequencies
        syscall_counts = {}
        for seq in sequences:
            for syscall in seq:
                syscall_counts[syscall] = syscall_counts.get(syscall, 0) + 1
        
        # Sort by frequency and take top max_vocab_size
        sorted_syscalls = sorted(syscall_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Build vocabulary (reserve 0 for padding, 1 for unknown)
        for idx, (syscall, _) in enumerate(sorted_syscalls[:self.max_vocab_size - 2], start=2):
            self.vocab[syscall] = idx
            self.reverse_vocab[idx] = syscall
        
        self.is_fitted = True
        return self
    
    def transform(self, sequences: List[List[str]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform system call sequences to token IDs.
        
        Args:
            sequences: List of system call sequences
            
        Returns:
            token_ids: (num_samples, max_seq_len) padded token IDs
            masks: (num_samples, max_seq_len) attention masks (1 for real tokens, 0 for padding)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        token_ids = []
        masks = []
        
        for seq in sequences:
            # Convert to token IDs
            tokens = [self.vocab.get(syscall, 1) for syscall in seq[:self.max_seq_len]]
            
            # Create mask (1 for real tokens)
            mask = [1] * len(tokens)
            
            # Pad sequence
            padding_len = self.max_seq_len - len(tokens)
            tokens.extend([0] * padding_len)
            mask.extend([0] * padding_len)
            
            token_ids.append(tokens)
            masks.append(mask)
        
        return np.array(token_ids, dtype=np.int64), np.array(masks, dtype=np.int64)
    
    def fit_transform(self, sequences: List[List[str]]) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and transform in one step"""
        return self.fit(sequences).transform(sequences)
    
    def save(self, path: str):
        """Save preprocessor state"""
        joblib.dump({
            'vocab': self.vocab,
            'reverse_vocab': self.reverse_vocab,
            'max_vocab_size': self.max_vocab_size,
            'max_seq_len': self.max_seq_len,
            'is_fitted': self.is_fitted
        }, path)
    
    def load(self, path: str):
        """Load preprocessor state"""
        state = joblib.load(path)
        self.vocab = state['vocab']
        self.reverse_vocab = state['reverse_vocab']
        self.max_vocab_size = state['max_vocab_size']
        self.max_seq_len = state['max_seq_len']
        self.is_fitted = state['is_fitted']


class TelemetryPreprocessor:
    """Preprocessor for host telemetry data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        
    def fit(self, data: pd.DataFrame) -> 'TelemetryPreprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            data: DataFrame with telemetry features
        """
        # Handle missing values
        data = data.fillna(0)
        
        # Store feature names
        self.feature_names = data.columns.tolist()
        
        # Fit scaler
        self.scaler.fit(data)
        self.is_fitted = True
        
        return self
    
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Transform telemetry data.
        
        Args:
            data: DataFrame with telemetry features
            
        Returns:
            Normalized numpy array
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Handle missing values
        data = data.fillna(0)
        
        # Scale features
        scaled_data = self.scaler.transform(data)
        
        return scaled_data.astype(np.float32)
    
    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(data).transform(data)
    
    def save(self, path: str):
        """Save preprocessor state"""
        joblib.dump({
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }, path)
    
    def load(self, path: str):
        """Load preprocessor state"""
        state = joblib.load(path)
        self.scaler = state['scaler']
        self.feature_names = state['feature_names']
        self.is_fitted = state['is_fitted']


class DataPreprocessor:
    """Main preprocessor that coordinates all modality preprocessors"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.network_preprocessor = NetworkPreprocessor()
        self.syscall_preprocessor = SyscallPreprocessor(
            max_vocab_size=config['model']['syscall_vocab_size'],
            max_seq_len=200
        )
        self.telemetry_preprocessor = TelemetryPreprocessor()
        self.label_encoder = LabelEncoder()
        
    def fit(self, network_data: pd.DataFrame, syscall_data: List[List[str]], 
            telemetry_data: pd.DataFrame, labels: np.ndarray):
        """Fit all preprocessors"""
        print("Fitting network preprocessor...")
        self.network_preprocessor.fit(network_data)
        
        print("Fitting syscall preprocessor...")
        self.syscall_preprocessor.fit(syscall_data)
        
        print("Fitting telemetry preprocessor...")
        self.telemetry_preprocessor.fit(telemetry_data)
        
        print("Fitting label encoder...")
        self.label_encoder.fit(labels)
        
        print("Preprocessing complete!")
        
    def transform(self, network_data: pd.DataFrame, syscall_data: List[List[str]], 
                  telemetry_data: pd.DataFrame, labels: np.ndarray = None):
        """Transform all data"""
        network_transformed = self.network_preprocessor.transform(network_data)
        syscall_transformed, syscall_masks = self.syscall_preprocessor.transform(syscall_data)
        telemetry_transformed = self.telemetry_preprocessor.transform(telemetry_data)
        
        if labels is not None:
            labels_transformed = self.label_encoder.transform(labels)
            return network_transformed, syscall_transformed, syscall_masks, telemetry_transformed, labels_transformed
        
        return network_transformed, syscall_transformed, syscall_masks, telemetry_transformed
    
    def save(self, save_dir: str):
        """Save all preprocessors"""
        os.makedirs(save_dir, exist_ok=True)
        
        self.network_preprocessor.save(os.path.join(save_dir, 'network_preprocessor.pkl'))
        self.syscall_preprocessor.save(os.path.join(save_dir, 'syscall_preprocessor.pkl'))
        self.telemetry_preprocessor.save(os.path.join(save_dir, 'telemetry_preprocessor.pkl'))
        joblib.dump(self.label_encoder, os.path.join(save_dir, 'label_encoder.pkl'))
        
        print(f"Preprocessors saved to {save_dir}")
    
    def load(self, save_dir: str):
        """Load all preprocessors"""
        self.network_preprocessor.load(os.path.join(save_dir, 'network_preprocessor.pkl'))
        self.syscall_preprocessor.load(os.path.join(save_dir, 'syscall_preprocessor.pkl'))
        self.telemetry_preprocessor.load(os.path.join(save_dir, 'telemetry_preprocessor.pkl'))
        self.label_encoder = joblib.load(os.path.join(save_dir, 'label_encoder.pkl'))
        
        print(f"Preprocessors loaded from {save_dir}")
