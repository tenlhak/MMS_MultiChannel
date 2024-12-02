# scripts/data_preparation.py

import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, random_split

def load_data(data_dir):
    """Load data from .npy files."""
    X_original = np.load(os.path.join(data_dir, 'X_t.npy'), allow_pickle=True)
    y_original = np.load(os.path.join(data_dir, 'y_t.npy'), allow_pickle=True)
    return X_original, y_original

def prepare_datasets(X_original, y_original, train_ratio=0.7, val_ratio=0.15):
    """Prepare train, validation, and test datasets."""
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_original, dtype=torch.float32)
    y_tensor = torch.tensor(y_original, dtype=torch.long)
    
    # Create a dataset
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # Define split sizes
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    return train_dataset, val_dataset, test_dataset
