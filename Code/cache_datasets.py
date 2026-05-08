"""
Dataset Caching Utility for Walsh Decomposition Network

Pre-generates and caches large datasets to disk, avoiding expensive
cipher encryption operations on every training run.

Usage:
    # Generate and cache datasets once (takes minutes)
    python cache_datasets.py
    
    # Then import in your training scripts
    from cache_datasets import load_dataset
    X_speck, y_speck = load_dataset('speck_7rounds_200k')
"""

import os
import pickle
import numpy as np
import torch
from pathlib import Path

# Configuration

CACHE_DIR = Path(__file__).parent / '.dataset_cache'
CACHE_DIR.mkdir(exist_ok=True, parents=True)

# Dataset configurations: (name, cipher, rounds, n_samples, input_diff)
DATASETS = {
    'speck_7rounds_50k': ('speck', 7, 50000, 0x00400000),
    'speck_7rounds_100k': ('speck', 7, 100000, 0x00400000),
    'speck_7rounds_200k': ('speck', 7, 200000, 0x00400000),
    'speck_7rounds_500k': ('speck', 7, 500000, 0x00400000),
    
    'speck_6rounds_200k': ('speck', 6, 200000, 0x00400000),
    'speck_8rounds_200k': ('speck', 8, 200000, 0x00400000),
    
    'simon_8rounds_50k': ('simon', 8, 50000, 0x00400000),
    'simon_8rounds_100k': ('simon', 8, 100000, 0x00400000),
    'simon_8rounds_200k': ('simon', 8, 200000, 0x00400000),
    'simon_8rounds_500k': ('simon', 8, 500000, 0x00400000),
}

# Caching Functions

def get_cache_path(name):
    """Get the cache file path for a dataset"""
    return CACHE_DIR / f"{name}.pkl"

def dataset_exists(name):
    """Check if a dataset is already cached"""
    return get_cache_path(name).exists()

def save_dataset(X, y, name):
    """
    Save dataset to cache
    
    Args:
        X: numpy array or torch tensor of features
        y: numpy array or torch tensor of labels
        name: string identifier for the dataset
    """
    # Convert to numpy if needed
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    
    cache_path = get_cache_path(name)
    
    # Save with compression
    with open(cache_path, 'wb') as f:
        pickle.dump({'X': X, 'y': y}, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Report size
    size_mb = cache_path.stat().st_size / (1024 * 1024)
    print(f"✓ Cached '{name}' ({X.shape[0]:,} samples, {size_mb:.1f} MB)")

def load_dataset(name, return_torch=True):
    """
    Load cached dataset
    
    Args:
        name: string identifier (must be pre-cached)
        return_torch: if True, return torch tensors; else numpy arrays
    
    Returns:
        X, y: features and labels
        
    Raises:
        FileNotFoundError if dataset not cached
    """
    cache_path = get_cache_path(name)
    
    if not cache_path.exists():
        available = [k for k in DATASETS.keys() if dataset_exists(k)]
        raise FileNotFoundError(
            f"Dataset '{name}' not cached.\n"
            f"Available: {available}\n"
            f"Generate with: python cache_datasets.py"
        )
    
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    
    X, y = data['X'], data['y']
    
    if return_torch:
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()
    
    return X, y

def generate_dataset(cipher, rounds, n_samples, input_diff):
    """
    Generate a dataset from scratch
    
    Args:
        cipher: 'speck' or 'simon'
        rounds: number of cipher rounds
        n_samples: total samples (half real, half random)
        input_diff: input difference for real pairs
    
    Returns:
        X, y: features and labels (numpy arrays)
    """
    from cipher_generation import make_dataset
    
    print(f"  Generating {cipher.upper()} {rounds}-round, {n_samples:,} samples...", end='', flush=True)
    X, y = make_dataset(n_samples, cipher=cipher, rounds=rounds, input_diff=input_diff)
    
    # Convert to numpy (make_dataset returns torch tensors)
    X = X.numpy()
    y = y.numpy()
    
    print(" ✓")
    return X, y

# Main: Generate All Cached Datasets

def generate_all_datasets(specific_names=None):
    """
    Generate and cache all datasets
    
    Args:
        specific_names: list of dataset names to generate
                       If None, generates all
    """
    if specific_names is None:
        specific_names = list(DATASETS.keys())
    
    print("DATASET CACHE GENERATION")
    print(f"Cache directory: {CACHE_DIR}")
    print()
    
    # Check what's already cached
    already_cached = [name for name in specific_names if dataset_exists(name)]
    to_generate = [name for name in specific_names if not dataset_exists(name)]
    
    if already_cached:
        print(f"Already cached ({len(already_cached)}):")
        for name in already_cached:
            size_mb = get_cache_path(name).stat().st_size / (1024 * 1024)
            print(f"  ✓ {name} ({size_mb:.1f} MB)")
        print()
    
    if not to_generate:
        print("All requested datasets are already cached!")
        return
    
    print(f"Generating {len(to_generate)} new datasets...")
    print()
    
    for name in to_generate:
        cipher, rounds, n_samples, input_diff = DATASETS[name]
        
        print(f"[{len(to_generate) - to_generate.index(name)}/{len(to_generate)}] {name}")
        X, y = generate_dataset(cipher, rounds, n_samples, input_diff)
        save_dataset(X, y, name)
    
    print()
    print("CACHE GENERATION COMPLETE")
    print()
    print("Usage in your code:")
    print("  from cache_datasets import load_dataset")
    print("  X_speck, y_speck = load_dataset('speck_7rounds_200k')")
    print()

def print_cache_info():
    """Print information about cached datasets"""
    print("CACHED DATASETS")
    print()
    
    cached = [(name, get_cache_path(name)) for name in DATASETS.keys() if dataset_exists(name)]
    
    if not cached:
        print("No cached datasets found. Run: python cache_datasets.py")
        return
    
    total_size = 0
    for name, path in cached:
        size_mb = path.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"  {name:<30} {size_mb:>8.1f} MB")
    
    print()
    print(f"Total cached: {total_size:.1f} MB")
    print()

# CLI Interface

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate and manage dataset caches for WDN training'
    )
    parser.add_argument(
        '--generate', 
        nargs='*',
        metavar='NAME',
        help='Generate specific datasets (leave empty for all)'
    )
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show cached dataset information'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available datasets'
    )
    parser.add_argument(
        '--clear',
        action='store_true',
        help='Delete all cached datasets'
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("AVAILABLE DATASETS")
        print()
        by_cipher = {}
        for name, (cipher, rounds, n_samples, _) in DATASETS.items():
            cipher_key = f"{cipher.upper()} {rounds}-round"
            if cipher_key not in by_cipher:
                by_cipher[cipher_key] = []
            by_cipher[cipher_key].append((n_samples, name))
        
        for cipher_info in sorted(by_cipher.keys()):
            print(f"{cipher_info}:")
            for n_samples, name in sorted(by_cipher[cipher_info]):
                status = "✓ cached" if dataset_exists(name) else "  pending"
                print(f"  {status}  {name:<35} ({n_samples:>7,} samples)")
            print()
    
    elif args.clear:
        import shutil
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            CACHE_DIR.mkdir(exist_ok=True, parents=True)
            print("✓ Cache cleared")
        else:
            print("Cache directory not found")
    
    elif args.info:
        print_cache_info()
    
    elif args.generate is not None:
        if len(args.generate) == 0:
            # Generate all
            generate_all_datasets()
        else:
            # Generate specific datasets
            for name in args.generate:
                if name not in DATASETS:
                    print(f"Error: Unknown dataset '{name}'")
                    print(f"Available: {list(DATASETS.keys())}")
                    exit(1)
            generate_all_datasets(args.generate)
    
    else:
        # Default: show info
        print_cache_info()
