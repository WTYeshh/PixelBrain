import torch
import torch.nn as nn
import torch.nn.functional as F

# We are naming this MNISTCNN so app.py can import it perfectly!
class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        # First convolutional layer: looks for simple edges
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        # Second convolutional layer: combines edges into shapes
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Max pooling shrinks the image dimensions
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layer that outputs probabilities for digits 0-9
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7) # Flatten matrix into a line
        x = self.fc(x)
        return x