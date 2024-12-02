# utils/helpers.py

import torch
from torch.utils.data import DataLoader

def get_device():
    """Check and return the available device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=64):
    """Create DataLoaders for datasets."""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
