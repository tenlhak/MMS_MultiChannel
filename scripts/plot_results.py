# scripts/plot_results.py

import matplotlib.pyplot as plt
import os
import numpy as np

def plot_loss(train_losses, val_losses, num_epochs, plot_dir):
    """Plot and save the loss over epochs."""
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(plot_dir, 'loss_plot.png')
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Loss plot saved to {loss_plot_path}")

def plot_accuracy(train_accuracies, val_accuracies, num_epochs, plot_dir):
    """Plot and save the accuracy over epochs."""
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    accuracy_plot_path = os.path.join(plot_dir, 'accuracy_plot.png')
    plt.savefig(accuracy_plot_path)
    plt.close()
    print(f"Accuracy plot saved to {accuracy_plot_path}")

def plot_per_class_predictions(per_class_correct, per_class_incorrect, classes, plot_dir):
    """Plot and save per-class correct and incorrect predictions."""
    correct_counts = [per_class_correct[classes.index(cls)] for cls in classes]
    incorrect_counts = [per_class_incorrect[classes.index(cls)] for cls in classes]
    x = np.arange(len(classes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, correct_counts, width, label='Correct')
    rects2 = ax.bar(x + width/2, incorrect_counts, width, label='Incorrect')
    ax.set_ylabel('Number of Predictions')
    ax.set_xlabel('Class')
    ax.set_title('Correct vs Incorrect Predictions per Class on Test Set')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.grid(axis='y')

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    per_class_plot_path = os.path.join(plot_dir, 'per_class_predictions.png')
    plt.savefig(per_class_plot_path)
    plt.close()
    print(f"Per-class prediction plot saved to {per_class_plot_path}")
