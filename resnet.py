import numpy as np
import torch.nn as nn

class Block(nn.Module):
    """
    A ResNet block containing 2 convolutional layers (with batch norm and relu), and a skip connection.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of channels produced by the convolutional layer.
            stride: Stride of the convolution (default=1).
        """
        super(Block, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if in_channels == out_channels and stride == 1:
            self.skip_flag = False
        else:
            self.skip_flag = True
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels)
            
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Args:
            x: Input tensors of shape (batch_size, in_channels, height, width).
        
        Returns:
            Outputs of shape (batch_size, out_channels).
        """
        identity = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.skip_flag:
            identity = self.skip_connection(identity)
            identity = self.bn3(identity)
        x += identity
    
        x = self.relu2(x)
        
        return x


class BlockGroup(nn.Module):
    """
    A group of ResNet blocks containing same number of output channels.
    """
    def __init__(self, in_channels, out_channels, n_blocks, stride=1):
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of channels produced by the convolutional layer.
            n_blocks: Number of blocks in the group.
            stride: Stride of the convolution (default=1).
        """
        super(BlockGroup, self).__init__()

        first_block = Block(in_channels, out_channels, stride)
        next_blocks = [Block(out_channels, out_channels) for _ in range(1, n_blocks)]
        self.block_group = nn.Sequential(first_block, *next_blocks)

    def forward(self, x):
        """
        Args:
            x: Input tensors of shape (batch_size, in_channels, height, width).
        
        Returns:
            Outputs of shape (batch_size, out_channels).
        """
        return self.block_group(x)


class ResNet(nn.Module):
    """
    Convolutional neural network based on ResNet architecture.
    """
    def __init__(self, in_channels, out_classes, out_channels=64, n_blocks=[2, 2, 2]):
        """
        Args:
            in_channels: Number of channels in the input images e.g. 3 for RBG, 1 for grayscale.
            out_classes: Number of output classes.
            out_channels: Basis for the number of channels produced by the convolutional layer  (default=64).
            n_blocks: Number of blocks in each block group (default=[2, 2, 2]).
        """
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.group1 = BlockGroup(out_channels, out_channels, n_blocks[0])
        self.group2 = BlockGroup(out_channels, 2 * out_channels, n_blocks[1], stride=2)
        self.group3 = BlockGroup(2 * out_channels, 4 * out_channels, n_blocks[2], stride=2)

        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=1)
        self.fc = nn.Linear(4 * out_channels, out_classes)

        # initialize weightss
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, x):
        """
        Args:
            x: Input image tensors of shape (batch_size, in_channels, height, width).
        
        Returns:
            Outputs of shape (batch_size, out_classes).
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)

        x = self.avgpool(x)

        x = x.view(-1, self.fc.in_features)
        x = self.fc(x)

        return x
