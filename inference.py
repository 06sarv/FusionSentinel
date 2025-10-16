"""
Inference script for FusionSentinel
"""

import torch
import numpy as np
import argparse

from models import FusionSentinel
from data.preprocessing import DataPreprocessor
from utils import load_config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run inference with FusionSentinel')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cuda/cpu)')
    return parser.parse_args()


class FusionSentinelInference:
    """Inference wrapper for FusionSentinel"""
    
    def __init__(self, config_path: str, checkpoint_path: str, device: str = 'cpu'):
        """
        Initialize inference model.
        
        Args:
            config_path: Path to configuration file
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
        """
        self.config = load_config(config_path)
        self.device = torch.device(device)
        
        # Load model
        self.model = FusionSentinel(self.config)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load preprocessors
        self.preprocessor = DataPreprocessor(self.config)
        self.preprocessor.load('data/processed')
        
        self.class_names = self.config['attack_types']
        
        print("✓ Model loaded successfully")
        print(f"  Device: {self.device}")
        print(f"  Classes: {len(self.class_names)}")
    
    def predict(self, network_features, syscall_sequence, telemetry_features):
        """
        Run inference on a single sample.
        
        Args:
            network_features: Network traffic features (dict or array)
            syscall_sequence: List of system call strings
            telemetry_features: Host telemetry features (dict or array)
            
        Returns:
            prediction: Predicted class name
            probability: Prediction probability
            all_probabilities: Probabilities for all classes
        """
        # Preprocess inputs
        # (In production, you'd convert raw inputs to the expected format)
        
        # For demonstration, assume inputs are already in correct format
        network = torch.FloatTensor(network_features).unsqueeze(0).unsqueeze(0)
        
        # Tokenize syscalls
        syscall_tokens = [self.preprocessor.syscall_preprocessor.vocab.get(sc, 1) 
                         for sc in syscall_sequence[:200]]
        syscall_mask = [1] * len(syscall_tokens)
        
        # Pad
        padding_len = 200 - len(syscall_tokens)
        syscall_tokens.extend([0] * padding_len)
        syscall_mask.extend([0] * padding_len)
        
        syscall = torch.LongTensor(syscall_tokens).unsqueeze(0)
        syscall_mask_tensor = torch.LongTensor(syscall_mask).unsqueeze(0)
        
        telemetry = torch.FloatTensor(telemetry_features).unsqueeze(0)
        
        # Move to device
        network = network.to(self.device)
        syscall = syscall.to(self.device)
        syscall_mask_tensor = syscall_mask_tensor.to(self.device)
        telemetry = telemetry.to(self.device)
        
        # Run inference
        with torch.no_grad():
            logits, attention_weights = self.model(network, syscall, telemetry, syscall_mask_tensor)
            probabilities = torch.softmax(logits, dim=1)
            prediction_idx = torch.argmax(probabilities, dim=1).item()
            prediction_prob = probabilities[0, prediction_idx].item()
        
        prediction_name = self.class_names[prediction_idx]
        all_probs = probabilities[0].cpu().numpy()
        
        return prediction_name, prediction_prob, all_probs, attention_weights
    
    def predict_batch(self, network_batch, syscall_batch, telemetry_batch):
        """
        Run inference on a batch of samples.
        
        Args:
            network_batch: Batch of network features
            syscall_batch: Batch of syscall sequences
            telemetry_batch: Batch of telemetry features
            
        Returns:
            predictions: List of predicted class names
            probabilities: List of prediction probabilities
        """
        # Similar to predict() but for batches
        # Implementation left as exercise
        pass


def main():
    """Demo inference"""
    args = parse_args()
    
    print("=" * 80)
    print("FusionSentinel Inference Demo")
    print("=" * 80)
    
    # Initialize inference model
    inference = FusionSentinelInference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # Example inference (with dummy data)
    print("\nRunning example inference...")
    
    # Dummy inputs
    network_features = np.random.randn(78).astype(np.float32)
    syscall_sequence = ['open', 'read', 'write', 'close', 'socket', 'connect'] * 10
    telemetry_features = np.random.rand(20).astype(np.float32)
    
    # Predict
    prediction, probability, all_probs, attention = inference.predict(
        network_features, syscall_sequence, telemetry_features
    )
    
    print("\n" + "=" * 80)
    print("Prediction Results")
    print("=" * 80)
    print(f"Predicted Class: {prediction}")
    print(f"Confidence: {probability:.4f} ({probability*100:.2f}%)")
    print("\nAll Class Probabilities:")
    for i, (class_name, prob) in enumerate(zip(inference.class_names, all_probs)):
        print(f"  {class_name:15s}: {prob:.4f} ({prob*100:.2f}%)")
    print("=" * 80)


if __name__ == '__main__':
    main()
