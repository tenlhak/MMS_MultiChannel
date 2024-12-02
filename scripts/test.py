# scripts/test.py

import torch
from collections import defaultdict

def test_model(model, device, test_loader, criterion, classes):
    """Test the model and perform per-class prediction analysis."""
    model.eval()
    test_running_loss = 0.0
    correct_test = 0
    total_test = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_running_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_test += (predicted == batch_y).sum().item()
            total_test += batch_y.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    test_loss = test_running_loss / total_test
    test_accuracy = correct_test / total_test
    print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Per-Class Analysis
    per_class_correct = defaultdict(int)
    per_class_incorrect = defaultdict(int)

    for pred, true in zip(all_preds, all_labels):
        if pred == true:
            per_class_correct[true] += 1
        else:
            per_class_incorrect[true] += 1

    print("\nPer-Class Prediction Results on Test Set:")
    for cls in classes:
        cls_index = classes.index(cls)
        correct = per_class_correct[cls_index]
        incorrect = per_class_incorrect[cls_index]
        print(f"Class {cls}: Correct Predictions = {correct}, Incorrect Predictions = {incorrect}")

    return all_preds, all_labels, per_class_correct, per_class_incorrect
