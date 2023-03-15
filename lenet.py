import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    """
    Convolutional neural network based on LeNet architecture.
    """
    def __init__(self, in_channels, out_classes):
        """
        Args:
            in_channels: Number of channels in the input images e.g. 3 for RBG, 1 for grayscale.
            out_classes: Number of output classes.
        """
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(4 * 4 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_classes)
    
    def forward(self, x):
        """
        Args:
            x: Input image tensors of shape (batch_size, in_channels, height, width).
        
        Returns:
            Outputs of shape (batch_size, out_classes).
        """
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)

        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        
        x = torch.flatten(x, 1)
    
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
