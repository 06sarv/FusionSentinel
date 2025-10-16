"""
Synthetic data generator for demonstration and testing
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
import os


def generate_synthetic_data(num_samples: int = 10000, 
                           network_features: int = 78,
                           telemetry_features: int = 20,
                           num_classes: int = 10,
                           save_dir: str = 'data/raw') -> Tuple:
    """
    Generate synthetic multi-modal cyber threat data for demonstration.
    
    Args:
        num_samples: Number of samples to generate
        network_features: Number of network traffic features
        telemetry_features: Number of telemetry features
        num_classes: Number of attack classes
        save_dir: Directory to save data
        
    Returns:
        network_data, syscall_data, telemetry_data, labels
    """
    print("=" * 80)
    print("Generating Synthetic Multi-Modal Cyber Threat Data")
    print("=" * 80)
    
    np.random.seed(42)
    
    # Generate labels
    labels = np.random.randint(0, num_classes, size=num_samples)
    
    # Generate network traffic features (CICIDS2017-like)
    print(f"Generating {num_samples} network traffic samples with {network_features} features...")
    network_data = np.random.randn(num_samples, network_features).astype(np.float32)
    
    # Add class-specific patterns
    for class_id in range(num_classes):
        mask = labels == class_id
        # Add distinctive patterns for each class
        network_data[mask] += np.random.randn(network_features) * 2 + class_id * 0.5
    
    # Add some extreme values (simulating attacks)
    attack_mask = labels > 0
    network_data[attack_mask] *= np.random.uniform(1.5, 3.0, size=(attack_mask.sum(), network_features))
    
    # Create DataFrame with feature names
    network_columns = [f'net_feature_{i}' for i in range(network_features)]
    network_df = pd.DataFrame(network_data, columns=network_columns)
    
    # Generate system call sequences (ADFA-LD-like)
    print(f"Generating {num_samples} system call sequences...")
    syscall_vocab = [
        'open', 'close', 'read', 'write', 'fork', 'exec', 'socket', 'connect',
        'bind', 'listen', 'accept', 'send', 'recv', 'mmap', 'munmap', 'brk',
        'ioctl', 'fcntl', 'stat', 'fstat', 'lstat', 'poll', 'select', 'kill',
        'getpid', 'getuid', 'setuid', 'chmod', 'chown', 'mkdir', 'rmdir', 'unlink'
    ]
    
    syscall_sequences = []
    for i in range(num_samples):
        # Sequence length varies by class
        seq_len = np.random.randint(50, 200)
        
        # Normal behavior: more common syscalls
        if labels[i] == 0:
            weights = np.array([0.2, 0.2, 0.15, 0.15, 0.05, 0.05] + [0.02] * (len(syscall_vocab) - 6))
        else:
            # Attack behavior: more suspicious syscalls
            weights = np.array([0.05, 0.05, 0.1, 0.1, 0.15, 0.15] + [0.04] * (len(syscall_vocab) - 6))
        
        weights = weights / weights.sum()
        sequence = np.random.choice(syscall_vocab, size=seq_len, p=weights)
        syscall_sequences.append(sequence.tolist())
    
    # Generate host telemetry features
    print(f"Generating {num_samples} host telemetry samples with {telemetry_features} features...")
    telemetry_data = np.random.rand(num_samples, telemetry_features).astype(np.float32)
    
    # Add class-specific patterns
    for class_id in range(num_classes):
        mask = labels == class_id
        # CPU usage, memory usage, I/O patterns
        telemetry_data[mask, :5] += class_id * 0.1  # CPU metrics
        telemetry_data[mask, 5:10] += np.random.rand(5) * class_id * 0.15  # Memory metrics
        telemetry_data[mask, 10:] += np.random.randn(telemetry_features - 10) * 0.2  # Other metrics
    
    # Attacks have higher resource usage
    attack_mask = labels > 0
    telemetry_data[attack_mask] *= np.random.uniform(1.2, 2.5, size=(attack_mask.sum(), telemetry_features))
    telemetry_data = np.clip(telemetry_data, 0, 1)  # Keep in [0, 1] range
    
    # Create DataFrame with feature names
    telemetry_columns = [
        'cpu_usage', 'cpu_user', 'cpu_system', 'cpu_idle', 'cpu_iowait',
        'mem_used', 'mem_free', 'mem_cached', 'mem_buffers', 'mem_swap',
        'disk_read_bytes', 'disk_write_bytes', 'disk_read_ops', 'disk_write_ops',
        'net_bytes_sent', 'net_bytes_recv', 'net_packets_sent', 'net_packets_recv',
        'process_count', 'thread_count'
    ]
    telemetry_df = pd.DataFrame(telemetry_data, columns=telemetry_columns)
    
    # Create label names
    attack_types = [
        "Normal", "DoS", "DDoS", "PortScan", "BruteForce",
        "WebAttack", "Infiltration", "Botnet", "Heartbleed", "Backdoor"
    ]
    label_names = [attack_types[i] for i in labels]
    
    # Save data
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\nSaving data to {save_dir}...")
        network_df.to_csv(os.path.join(save_dir, 'network_data.csv'), index=False)
        telemetry_df.to_csv(os.path.join(save_dir, 'telemetry_data.csv'), index=False)
        
        # Save syscalls as text file
        with open(os.path.join(save_dir, 'syscall_data.txt'), 'w') as f:
            for seq in syscall_sequences:
                f.write(' '.join(seq) + '\n')
        
        # Save labels
        labels_df = pd.DataFrame({'label': labels, 'label_name': label_names})
        labels_df.to_csv(os.path.join(save_dir, 'labels.csv'), index=False)
        
        print("✓ Data saved successfully!")
    
    print("\nData Summary:")
    print(f"  Total Samples: {num_samples}")
    print(f"  Network Features: {network_features}")
    print(f"  Telemetry Features: {telemetry_features}")
    print(f"  Syscall Vocab Size: {len(syscall_vocab)}")
    print(f"  Classes: {num_classes}")
    print(f"\nClass Distribution:")
    for i, attack_type in enumerate(attack_types):
        count = (labels == i).sum()
        print(f"    {attack_type:15s}: {count:5d} ({count/num_samples*100:.1f}%)")
    print("=" * 80)
    
    return network_df, syscall_sequences, telemetry_df, labels


def load_synthetic_data(data_dir: str = 'data/raw') -> Tuple:
    """
    Load previously generated synthetic data.
    
    Args:
        data_dir: Directory containing the data
        
    Returns:
        network_data, syscall_data, telemetry_data, labels
    """
    print(f"Loading data from {data_dir}...")
    
    # Load network data
    network_df = pd.read_csv(os.path.join(data_dir, 'network_data.csv'))
    
    # Load telemetry data
    telemetry_df = pd.read_csv(os.path.join(data_dir, 'telemetry_data.csv'))
    
    # Load syscall sequences
    syscall_sequences = []
    with open(os.path.join(data_dir, 'syscall_data.txt'), 'r') as f:
        for line in f:
            syscall_sequences.append(line.strip().split())
    
    # Load labels
    labels_df = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
    labels = labels_df['label'].values
    
    print(f"✓ Loaded {len(labels)} samples")
    
    return network_df, syscall_sequences, telemetry_df, labels
