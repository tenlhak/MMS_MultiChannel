# models/multichannel_cnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiChannel1DCNN(nn.Module):
    def __init__(self):
        super(MultiChannel1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 251, 64)  # Adjust based on signal length after conv and pool
        self.fc2 = nn.Linear(64, 4)         # Assuming 4 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Shape: (Samples, 16, 505)
        x = self.pool(F.relu(self.conv2(x)))  # Shape: (Samples, 32, 251)
        x = x.view(x.size(0), -1)             # Flatten to (Samples, 32*251)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                       # Output logits
        return x
