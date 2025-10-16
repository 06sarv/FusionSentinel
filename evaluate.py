"""
Evaluation script for trained FusionSentinel model
"""

import torch
import argparse
import os

from models import FusionSentinel
from data import MultiModalDataset
from torch.utils.data import DataLoader
from data.preprocessing import DataPreprocessor
from evaluation import Evaluator, plot_confusion_matrix
from evaluation.visualizer import AttentionVisualizer, plot_per_class_metrics
from utils import load_synthetic_data, load_config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate FusionSentinel Model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data/raw',
                       help='Directory containing test data')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate attention visualizations')
    parser.add_argument('--num-vis-samples', type=int, default=10,
                       help='Number of samples for attention visualization')
    return parser.parse_args()


def main():
    """Main evaluation pipeline"""
    args = parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Override device if specified
    if args.device:
        device = args.device
    else:
        device = config['training']['device']
    
    # Load data
    print("\n" + "=" * 80)
    print("Loading Test Data")
    print("=" * 80)
    
    network_df, syscall_sequences, telemetry_df, labels = load_synthetic_data(args.data_dir)
    
    # Load preprocessors
    print("\nLoading preprocessors...")
    preprocessor = DataPreprocessor(config)
    preprocessor.load('data/processed')
    
    # Transform data
    print("Preprocessing data...")
    test_net, test_sys, test_sys_mask, test_tel, test_labels = preprocessor.transform(
        network_df, syscall_sequences, telemetry_df, labels
    )
    
    # Create dataset and dataloader
    test_dataset = MultiModalDataset(test_net, test_sys, test_sys_mask, test_tel, test_labels)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['evaluation']['test_batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    print(f"✓ Test dataset: {len(test_dataset)} samples")
    
    # Create model
    print("\n" + "=" * 80)
    print("Loading Model")
    print("=" * 80)
    
    model = FusionSentinel(config)
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Create evaluator
    evaluator = Evaluator(model, device=device)
    
    # Evaluate
    print("\n" + "=" * 80)
    print("Evaluating Model")
    print("=" * 80)
    
    class_names = config['attack_types']
    metrics = evaluator.evaluate(test_loader, class_names=class_names)
    
    # Save results
    os.makedirs('evaluation_results', exist_ok=True)
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        class_names,
        save_path='evaluation_results/confusion_matrix.png',
        normalize=True
    )
    
    # Plot per-class metrics
    print("Generating per-class metrics plot...")
    plot_per_class_metrics(
        metrics,
        class_names,
        save_path='evaluation_results/per_class_metrics.png'
    )
    
    # Visualize attention if requested
    if args.visualize:
        print("\n" + "=" * 80)
        print("Generating Attention Visualizations")
        print("=" * 80)
        
        attention_data = evaluator.get_attention_weights(test_loader, num_samples=args.num_vis_samples)
        
        visualizer = AttentionVisualizer(save_dir='evaluation_results/attention')
        
        # Plot attention for samples
        for i in range(min(args.num_vis_samples, len(attention_data['predictions']))):
            visualizer.plot_attention_comparison(attention_data, sample_idx=i, class_names=class_names)
        
        # Plot attention distribution
        visualizer.plot_attention_distribution(attention_data, num_samples=args.num_vis_samples)
    
    print("\n" + "=" * 80)
    print("✓ EVALUATION COMPLETE!")
    print("=" * 80)
    print(f"\nResults Summary:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"\nResults saved to: evaluation_results/")
    print("=" * 80)


if __name__ == '__main__':
    main()
