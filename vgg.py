import torch.nn as nn

class ConvLayer(nn.Module):
    """
    A convolutional layer with batch normalization and relu non-linearity.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """
        Args:
            in_channels: Number of input channels in the input.
            out_channels: Number of channels produced by the convolution.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution (default=1).
            padding: Number of pixels to be padded on all sides of the input (default=0).
        """
        super(ConvLayer, self).__init__()

        self.layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
    
    def forward(self, x):
        """
        Args:
            x: Input tensors of shape (batch_size, in_channels, height, width).
        
        Returns:
            Outputs of shape (batch_size, out_channels).
        """
        x = self.layer(x)

        return x


class VGGNet(nn.Module):
    """
    Convolutional neural network based on VGG architecture.
    """
    def __init__(self, in_channels, out_classes, out_channels=20):
        """
        Args:
            in_channels: Number of channels in the input images e.g. 3 for RBG, 1 for grayscale.
            out_classes: Number of output classes.
            out_channels: Basis for the number of channels produced by the convolutional layer (default=20).
        """
        super(VGGNet, self).__init__()

        self.block1 = nn.Sequential(
            ConvLayer(in_channels, out_channels, 3, padding=1),
            ConvLayer(out_channels, out_channels, 3, padding=1),
            ConvLayer(out_channels, out_channels, 3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.block2 = nn.Sequential(
            ConvLayer(out_channels, 2 * out_channels, 3, padding=1),
            ConvLayer(2 * out_channels, 2 * out_channels, 3, padding=1),
            ConvLayer(2 * out_channels, 2 * out_channels, 3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.block3 = nn.Sequential(
            ConvLayer(2 * out_channels, 3 * out_channels, 3),
            ConvLayer(3 * out_channels, 2 * out_channels, 1),
            ConvLayer(2 * out_channels, out_channels, 1),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(out_channels, out_classes)


    def forward(self, x):
        """
        Args:
            x: Input image tensors of shape (batch_size, in_channels, height, width).
        
        Returns:
            Outputs of shape (batch_size, out_classes).
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.avgpool(x)
        x = x.view(-1, self.fc.in_features)
        x = self.fc(x)

        return x
