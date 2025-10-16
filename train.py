"""
Main training script for FusionSentinel
"""

import torch
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import os

from models import FusionSentinel
from data import MultiModalDataset, create_dataloaders
from data.preprocessing import DataPreprocessor
from training import Trainer
from evaluation import Evaluator, plot_confusion_matrix, plot_training_history
from evaluation.visualizer import AttentionVisualizer, plot_per_class_metrics
from utils import generate_synthetic_data, load_synthetic_data, load_config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train FusionSentinel Model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--generate-data', action='store_true',
                       help='Generate synthetic data')
    parser.add_argument('--num-samples', type=int, default=10000,
                       help='Number of samples to generate')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    return parser.parse_args()


def main():
    """Main training pipeline"""
    args = parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Override device if specified
    if args.device:
        config['training']['device'] = args.device
    
    # Generate or load data
    data_dir = 'data/raw'
    
    if args.generate_data or not os.path.exists(os.path.join(data_dir, 'network_data.csv')):
        print("\n" + "=" * 80)
        print("STEP 1: Generating Synthetic Data")
        print("=" * 80)
        network_df, syscall_sequences, telemetry_df, labels = generate_synthetic_data(
            num_samples=args.num_samples,
            network_features=config['model']['network_input_dim'],
            telemetry_features=config['model']['telemetry_input_dim'],
            num_classes=config['model']['num_classes'],
            save_dir=data_dir
        )
    else:
        print("\n" + "=" * 80)
        print("STEP 1: Loading Existing Data")
        print("=" * 80)
        network_df, syscall_sequences, telemetry_df, labels = load_synthetic_data(data_dir)
    
    # Split data
    print("\n" + "=" * 80)
    print("STEP 2: Splitting Data")
    print("=" * 80)
    
    indices = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=labels)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, 
                                         stratify=labels[temp_idx])
    
    print(f"Train samples: {len(train_idx)}")
    print(f"Validation samples: {len(val_idx)}")
    print(f"Test samples: {len(test_idx)}")
    
    # Preprocess data
    print("\n" + "=" * 80)
    print("STEP 3: Preprocessing Data")
    print("=" * 80)
    
    preprocessor = DataPreprocessor(config)
    
    # Fit on training data
    preprocessor.fit(
        network_df.iloc[train_idx],
        [syscall_sequences[i] for i in train_idx],
        telemetry_df.iloc[train_idx],
        labels[train_idx]
    )
    
    # Transform all splits
    print("Transforming training data...")
    train_net, train_sys, train_sys_mask, train_tel, train_labels = preprocessor.transform(
        network_df.iloc[train_idx],
        [syscall_sequences[i] for i in train_idx],
        telemetry_df.iloc[train_idx],
        labels[train_idx]
    )
    
    print("Transforming validation data...")
    val_net, val_sys, val_sys_mask, val_tel, val_labels = preprocessor.transform(
        network_df.iloc[val_idx],
        [syscall_sequences[i] for i in val_idx],
        telemetry_df.iloc[val_idx],
        labels[val_idx]
    )
    
    print("Transforming test data...")
    test_net, test_sys, test_sys_mask, test_tel, test_labels = preprocessor.transform(
        network_df.iloc[test_idx],
        [syscall_sequences[i] for i in test_idx],
        telemetry_df.iloc[test_idx],
        labels[test_idx]
    )
    
    # Save preprocessors
    preprocessor.save('data/processed')
    
    # Create datasets
    print("\n" + "=" * 80)
    print("STEP 4: Creating Datasets")
    print("=" * 80)
    
    train_dataset = MultiModalDataset(train_net, train_sys, train_sys_mask, train_tel, train_labels)
    val_dataset = MultiModalDataset(val_net, val_sys, val_sys_mask, val_tel, val_labels)
    test_dataset = MultiModalDataset(test_net, test_sys, test_sys_mask, test_tel, test_labels)
    
    print(f"✓ Train dataset: {len(train_dataset)} samples")
    print(f"✓ Validation dataset: {len(val_dataset)} samples")
    print(f"✓ Test dataset: {len(test_dataset)} samples")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers']
    )
    
    # Create model
    print("\n" + "=" * 80)
    print("STEP 5: Creating Model")
    print("=" * 80)
    
    model = FusionSentinel(config)
    model.summary()
    
    # Create trainer
    trainer = Trainer(model, config, device=config['training']['device'])
    
    # Set class weights for imbalanced data
    class_weights = train_dataset.get_class_weights()
    trainer.set_class_weights(class_weights)
    
    # Load checkpoint if specified
    if args.checkpoint:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
    
    # Train model
    print("\n" + "=" * 80)
    print("STEP 6: Training Model")
    print("=" * 80)
    
    trainer.fit(train_loader, val_loader)
    
    # Plot training history
    print("\n" + "=" * 80)
    print("STEP 7: Visualizing Training History")
    print("=" * 80)
    
    os.makedirs('visualizations', exist_ok=True)
    plot_training_history(trainer.history, save_path='visualizations/training_history.png')
    
    # Evaluate on test set
    print("\n" + "=" * 80)
    print("STEP 8: Evaluating on Test Set")
    print("=" * 80)
    
    # Load best model
    trainer.load_checkpoint()
    
    evaluator = Evaluator(model, device=config['training']['device'])
    
    class_names = config['attack_types']
    metrics = evaluator.evaluate(test_loader, class_names=class_names)
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        class_names,
        save_path='visualizations/confusion_matrix.png',
        normalize=True
    )
    
    # Plot per-class metrics
    print("Generating per-class metrics plot...")
    plot_per_class_metrics(
        metrics,
        class_names,
        save_path='visualizations/per_class_metrics.png'
    )
    
    # Visualize attention weights
    print("\n" + "=" * 80)
    print("STEP 9: Visualizing Attention Weights")
    print("=" * 80)
    
    attention_data = evaluator.get_attention_weights(test_loader, num_samples=10)
    
    visualizer = AttentionVisualizer(save_dir='visualizations')
    
    # Plot attention for first 5 samples
    for i in range(min(5, len(attention_data['predictions']))):
        visualizer.plot_attention_comparison(attention_data, sample_idx=i, class_names=class_names)
    
    # Plot attention distribution
    visualizer.plot_attention_distribution(attention_data, num_samples=10)
    
    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nResults Summary:")
    print(f"  Test Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Test Precision: {metrics['precision']:.4f}")
    print(f"  Test Recall:    {metrics['recall']:.4f}")
    print(f"  Test F1-Score:  {metrics['f1_score']:.4f}")
    print(f"\nModel saved to: {config['training']['checkpoint_dir']}")
    print(f"Visualizations saved to: visualizations/")
    print("=" * 80)


if __name__ == '__main__':
    main()
