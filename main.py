# main.py

import os
from utils.seed import set_seed
from utils.helpers import get_device, create_dataloaders
from models.multichannel_cnn import MultiChannel1DCNN
from scripts.data_preparation import load_data, prepare_datasets
from scripts.train import train_model
from scripts.test import test_model
from scripts.plot_results import plot_loss, plot_accuracy, plot_per_class_predictions
import torch
import torch.nn as nn

def main():
    # Set random seeds for reproducibility
    set_seed()

    # Check device
    device = get_device()
    print(f"Using device: {device}")

    # Data Preparation
    data_dir = "/home/dapgrad/tenzinl2/TFPred/raw_data"
    X_original, y_original = load_data(data_dir)
    train_dataset, val_dataset, test_dataset = prepare_datasets(X_original, y_original)
    print(f"Dataset sizes -> Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")

    # DataLoaders
    batch_size = 64
    train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size)

    # Model Initialization
    model = MultiChannel1DCNN().to(device)
    print(model)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training
    num_epochs = 20  # Adjust as needed
    best_model_path = 'best_model_multichannel.pth'
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, device, train_loader, val_loader, criterion, optimizer, num_epochs, best_model_path)

    # Load the Best Model
    model.load_state_dict(torch.load(best_model_path))

    # Testing
    classes = ['normal', 'misaligned', 'imbalanced', 'bearing fault']
    all_preds, all_labels, per_class_correct, per_class_incorrect = test_model(
        model, device, test_loader, criterion, classes)

    # Plotting and Saving Results
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    plot_loss(train_losses, val_losses, num_epochs, plot_dir)
    plot_accuracy(train_accuracies, val_accuracies, num_epochs, plot_dir)
    plot_per_class_predictions(per_class_correct, per_class_incorrect, classes, plot_dir)

if __name__ == '__main__':
    main()
