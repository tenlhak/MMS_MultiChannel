import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import accuracy_score
import random

# Set random seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a directory to save plots
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

# 1. Define the Model Architecture
class MultiChannel1DCNN(nn.Module):
    def __init__(self):
        super(MultiChannel1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, 3)
        # Adjust the output size after conv and pool layers
        self.fc1 = nn.Linear(32 * 251, 64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Shape: (Samples, 16, 505)
        x = self.pool(F.relu(self.conv2(x)))  # Shape: (Samples, 32, 251)
        x = x.view(x.size(0), -1)             # Flatten to (Samples, 32*251)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                        # Output logits
        return x

# 2. Load the Trained Model
model = MultiChannel1DCNN().to(device)
model_path = '/home/dapgrad/tenzinl2/TFPred/best_model_multichannel.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded trained model from {model_path}")
else:
    raise FileNotFoundError(f"Model file {model_path} not found. Please ensure the model is saved.")

model.eval()


X_original = np.load(os.path.join("/home/dapgrad/tenzinl2/TFPred/raw_data", 'X_t.npy'), allow_pickle=True)  # Shape: (num_samples, channels, signal_length)
y_original = np.load(os.path.join("/home/dapgrad/tenzinl2/TFPred/raw_data", 'y_t.npy'), allow_pickle=True) 

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_original, dtype=torch.float32)
y_tensor = torch.tensor(y_original, dtype=torch.long)

# Create a dataset and dataloader
test_dataset = TensorDataset(X_tensor, y_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Get all true labels
all_labels = y_tensor.numpy()

# 4. Generate Predictions on the Test Set
all_preds = []
model.eval()
with torch.no_grad():
    for batch_X, _ in test_loader:
        batch_X = batch_X.to(device)
        outputs = model(batch_X)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())

# Compute test accuracy
test_accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {test_accuracy:.4f}")

# 5. Identify Correctly Predicted Samples
correct_indices = [i for i, (pred, true) in enumerate(zip(all_preds, all_labels)) if pred == true]
print(f"Number of correctly predicted samples: {len(correct_indices)}")

# 6. Initialize Integrated Gradients
ig = IntegratedGradients(model)

# 7. Compute Channel Contributions
def compute_channel_attributions(model, ig, data_loader, device, correct_indices):
    channel_attributions = {i: 0.0 for i in range(4)}  # Assuming 4 channels
    total_samples = 0

    for batch_idx, (inputs, labels) in enumerate(data_loader):
        batch_size = inputs.size(0)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Iterate through the batch
        for i in range(batch_size):
            global_idx = batch_idx * data_loader.batch_size + i
            if global_idx in correct_indices:
                input_sample = inputs[i].unsqueeze(0)  # Add batch dimension
                label = labels[i].item()

                # Compute attributions
                attributions, _ = ig.attribute(input_sample, 
                                               target=label, 
                                               return_convergence_delta=True)

                # attributions shape: (1, 4, 1013)
                attributions = attributions.squeeze(0).cpu().numpy()  # Shape: (4, 1013)

                # Aggregate attributions per channel by summing absolute values
                for channel in range(4):
                    channel_attributions[channel] += np.sum(np.abs(attributions[channel]))

                total_samples += 1

    # Average attributions per channel
    for channel in channel_attributions:
        channel_attributions[channel] /= total_samples if total_samples > 0 else 1

    return channel_attributions

# Compute attributions
channel_attributions = compute_channel_attributions(model, ig, test_loader, device, correct_indices)

# Print aggregated attributions
print("\nAggregated Attributions per Channel (Averaged over Correct Predictions):")
for channel, attribution in channel_attributions.items():
    print(f"Channel {channel}: {attribution:.4f}")

# 8. Visualize Channel Contributions
channels = list(channel_attributions.keys())
attributions = list(channel_attributions.values())

plt.figure(figsize=(8, 6))
plt.bar(channels, attributions, color=['skyblue', 'salmon', 'lightgreen', 'violet'])
plt.xlabel('Channel')
plt.ylabel('Average Attribution')
plt.title('Average Channel Contributions to Correct Predictions')
plt.xticks(channels)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(attributions):
    plt.text(channels[i], v + max(attributions)*0.01, f"{v:.2f}", ha='center', va='bottom')

# Save the plot
channel_plot_path = os.path.join(plot_dir, 'channel_contributions.png')
plt.savefig(channel_plot_path)
plt.close()
print(f"Channel contributions plot saved to {channel_plot_path}")

# 9. Permutation Importance
def permutation_importance(model, test_loader, device, channel_to_permute, all_preds, all_labels):
    # Compute baseline accuracy
    baseline_acc = accuracy_score(all_labels, all_preds)

    # Permute the specified channel
    permuted_preds = []
    for batch_X, _ in test_loader:
        batch_X = batch_X.numpy()
        batch_X[:, channel_to_permute, :] = np.random.permutation(batch_X[:, channel_to_permute, :])
        batch_X = torch.tensor(batch_X, dtype=torch.float32).to(device)

        with torch.no_grad():
            outputs = model(batch_X)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            permuted_preds.extend(preds)

    # Compute permuted accuracy
    permuted_acc = accuracy_score(all_labels, permuted_preds)

    # Importance is the decrease in accuracy
    importance = baseline_acc - permuted_acc
    return importance

# Compute permutation importance for each channel
permutation_importances = {}
for channel in range(4):
    importance = permutation_importance(model, test_loader, device, channel, all_preds, all_labels)
    permutation_importances[channel] = importance
    print(f"Channel {channel} Permutation Importance: {importance:.4f}")

# Plot permutation importances
channels = list(permutation_importances.keys())
importances = list(permutation_importances.values())

plt.figure(figsize=(8, 6))
plt.bar(channels, importances, color=['skyblue', 'salmon', 'lightgreen', 'violet'])
plt.xlabel('Channel')
plt.ylabel('Decrease in Accuracy')
plt.title('Permutation Importance per Channel')
plt.xticks(channels)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(importances):
    plt.text(channels[i], v + max(importances)*0.01, f"{v:.4f}", ha='center', va='bottom')

# Save the plot
permutation_plot_path = os.path.join(plot_dir, 'permutation_importance.png')
plt.savefig(permutation_plot_path)
plt.close()
print(f"Permutation importance plot saved to {permutation_plot_path}")

# 10. Ablation Study
def ablation_study(model, test_loader, device, channels_to_remove, all_preds, all_labels):
    # Compute baseline accuracy
    baseline_acc = accuracy_score(all_labels, all_preds)

    # Modify data by removing channels (set to zero)
    modified_preds = []
    for batch_X, _ in test_loader:
        batch_X = batch_X.numpy()
        for channel in channels_to_remove:
            batch_X[:, channel, :] = 0  # Zero out the channel
        batch_X = torch.tensor(batch_X, dtype=torch.float32).to(device)

        with torch.no_grad():
            outputs = model(batch_X)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            modified_preds.extend(preds)

    # Compute modified accuracy
    modified_acc = accuracy_score(all_labels, modified_preds)

    # Importance is the decrease in accuracy
    importance = baseline_acc - modified_acc
    return importance

# Perform ablation study for each channel
ablation_importances = {}
for channel in range(4):
    importance = ablation_study(model, test_loader, device, [channel], all_preds, all_labels)
    ablation_importances[channel] = importance
    print(f"Channel {channel} Ablation Importance: {importance:.4f}")

# Plot ablation importances
channels = list(ablation_importances.keys())
importances = list(ablation_importances.values())

plt.figure(figsize=(8, 6))
plt.bar(channels, importances, color=['skyblue', 'salmon', 'lightgreen', 'violet'])
plt.xlabel('Channel')
plt.ylabel('Decrease in Accuracy')
plt.title('Ablation Importance per Channel')
plt.xticks(channels)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(importances):
    plt.text(channels[i], v + max(importances)*0.01, f"{v:.4f}", ha='center', va='bottom')

# Save the plot
ablation_plot_path = os.path.join(plot_dir, 'ablation_importance.png')
plt.savefig(ablation_plot_path)
plt.close()
print(f"Ablation importance plot saved to {ablation_plot_path}")

# 11. Visualize Attributions for a Sample
def visualize_sample_attributions(model, ig, sample_index, test_dataset, device, plot_dir):
    model.eval()
    sample_X, sample_y = test_dataset[sample_index]
    input_tensor = sample_X.unsqueeze(0).to(device)  # Add batch dimension

    # Compute attributions
    attributions, delta = ig.attribute(input_tensor, 
                                       target=sample_y.item(), 
                                       return_convergence_delta=True)
    attributions = attributions.squeeze(0).cpu().detach().numpy()  # Shape: (4, 1013)

    # Plot attributions per channel
    plt.figure(figsize=(12, 8))
    for channel in range(4):
        plt.subplot(2, 2, channel+1)
        plt.plot(attributions[channel], color='red')
        plt.title(f'Channel {channel} Attribution')
        plt.xlabel('Feature Index')
        plt.ylabel('Attribution')
        plt.tight_layout()

    plt.suptitle(f'Attribution Maps for Sample {sample_index} - True Class {sample_y.item()}', y=1.02)
    sample_plot_path = os.path.join(plot_dir, f'sample_{sample_index}_attributions.png')
    plt.savefig(sample_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Attribution maps for sample {sample_index} saved to {sample_plot_path}")

# Example: Visualize attributions for the first correctly predicted sample
if len(correct_indices) > 0:
    first_correct_idx = correct_indices[0]
    visualize_sample_attributions(model, ig, first_correct_idx, test_dataset, device, plot_dir)

# 12. Completion Message
print("\nChannel contribution analysis completed.")
