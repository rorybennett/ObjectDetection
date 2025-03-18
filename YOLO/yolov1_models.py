"""
YOLO models set up as originally published (as close as possible anyways).
    - YOLOv1 based on this paper: https://arxiv.org/pdf/1506.02640. With some help (and additional settings)
    from: https://www.youtube.com/watch?v=n9_XyCGr-MI
"""
import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, LeakyReLU, MaxPool2d, Sequential, Module, Flatten, Linear, Dropout


class YOLOv1(Module):
    def __init__(self, S=7, B=2, C=2, input_channels=3):
        """
        YOLOv1 Model. Batch normalisation seems necessary and was not a part of the original paper..

        Args:
        - S (int): Grid size (default 7x7).
        - B (int): Number of bounding boxes per grid cell (default 2).
        - C (int): Number of classes (default 1).
        - input_channels (int): Input channels of original image (RGB or 1-channel greyscale)
        """
        super(YOLOv1, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.input_channels = input_channels

        # CNN Backbone
        self.darknet_layers = Sequential(
            # First convolutional layer (based on paper diagram).
            Conv2d(self.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            BatchNorm2d(64),
            LeakyReLU(0.1),
            MaxPool2d(kernel_size=2, stride=2),
            # Second convolutional layer.
            Conv2d(64, 192, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(192),
            LeakyReLU(0.1),
            MaxPool2d(kernel_size=2, stride=2),
            # Third batch of convolutional layers.
            Conv2d(192, 128, kernel_size=1, bias=False),
            BatchNorm2d(128),
            LeakyReLU(0.1),
            Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(256),
            LeakyReLU(0.1),
            Conv2d(256, 256, kernel_size=1, stride=1, bias=False),
            BatchNorm2d(256),
            LeakyReLU(0.1),
            Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(512),
            LeakyReLU(0.1),
            MaxPool2d(kernel_size=2, stride=2),
            # Fourth batch of convolutional layers (repeating the first 2 4 times).
            Conv2d(512, 256, kernel_size=1, stride=1, bias=False),
            BatchNorm2d(256),
            LeakyReLU(0.1),
            Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(512),
            LeakyReLU(0.1),
            Conv2d(512, 256, kernel_size=1, stride=1, bias=False),
            BatchNorm2d(256),
            LeakyReLU(0.1),
            Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(512),
            LeakyReLU(0.1),
            Conv2d(512, 256, kernel_size=1, stride=1, bias=False),
            BatchNorm2d(256),
            LeakyReLU(0.1),
            Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(512),
            LeakyReLU(0.1),
            Conv2d(512, 256, kernel_size=1, stride=1, bias=False),
            BatchNorm2d(256),
            LeakyReLU(0.1),
            Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(512),
            LeakyReLU(0.1),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm2d(512),
            LeakyReLU(0.1),
            Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(1024),
            LeakyReLU(0.1),
            MaxPool2d(kernel_size=2, stride=2),
            # Fifth batch of convolutional layers.
            Conv2d(1024, 512, kernel_size=1, stride=1, bias=False),
            BatchNorm2d(512),
            LeakyReLU(0.1),
            Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(1024),
            LeakyReLU(0.1),
            Conv2d(1024, 512, kernel_size=1, stride=1, bias=False),
            BatchNorm2d(512),
            LeakyReLU(0.1),
            Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(1024),
            LeakyReLU(0.1),
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(1024),
            LeakyReLU(0.1),
            Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(1024),
            LeakyReLU(0.1),
            # Final convolutional layers.
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(1024),
            LeakyReLU(0.1),
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(1024),
            LeakyReLU(0.1)
        )

        # Fully Connected Layers
        self.fc_layers = Sequential(
            Flatten(),
            Linear(1024 * 7 * 7, 4096),
            Dropout(0.5),
            LeakyReLU(0.1),
            Linear(4096, S * S * (B * 5 + C)),
        )

    def forward(self, x):
        x = self.darknet_layers(x)
        x = self.fc_layers(x)
        x = x.view(-1, self.S, self.S, self.B * 5 + self.C)
        return x


class YOLOv1Fast(Module):
    def __init__(self, S=7, B=2, C=1, input_channels=3):
        """
        YOLOv1Fast Model (similar to YOLOv1, with fewer (9) convolutional layers).

        Args:
        - S (int): Grid size (default 7x7).
        - B (int): Number of bounding boxes per grid cell (default 2).
        - C (int): Number of classes (default 1).
        - input_channels (int): Input channels of original image (RGB or 1-channel greyscale)
        """
        super(YOLOv1Fast, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.input_channels = input_channels

        # CNN Backbone
        self.darknet_fast_layers = Sequential(
            # First Conv Block
            Conv2d(self.input_channels, 16, kernel_size=7, stride=2, padding=3, bias=False),
            BatchNorm2d(16),
            LeakyReLU(0.1),
            MaxPool2d(2, 2),

            # Second Conv Block
            Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(32),
            LeakyReLU(0.1),
            MaxPool2d(2, 2),

            # Third Conv Block
            Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(64),
            LeakyReLU(0.1),
            MaxPool2d(2, 2),

            # Fourth Conv Block
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(128),
            LeakyReLU(0.1),
            MaxPool2d(2, 2),

            # Fifth Conv Block
            Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(256),
            LeakyReLU(0.1),
            MaxPool2d(2, 2),

            # Sixth Conv Block
            Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(512),
            LeakyReLU(0.1),
            # Remove this maxpool to ensure 7x7 feature map output from dark_net_fast layers.
            # MaxPool2d(2, 2),

            # Final Convolutions (Keep Feature Map 7x7)
            Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(1024),
            LeakyReLU(0.1),
            Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(512),
            LeakyReLU(0.1),
            Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(1024),
            LeakyReLU(0.1)
        )

        # Fully Connected Layers
        self.fc_layers = Sequential(
            Flatten(),
            Linear(1024 * 7 * 7, 2048),
            LeakyReLU(0.1),
            Dropout(0.5),
            Linear(2048, S * S * (B * 5 + C)),
        )

    def forward(self, x):
        x = self.darknet_fast_layers(x)
        x = self.fc_layers(x)
        x = x.view(-1, self.S, self.S, self.B * 5 + self.C)
        return x


########################################################################################################################
# Model loading tests.
########################################################################################################################
# import torch
#
# model = YOLOv1Fast(S=7, B=2, C=1, input_channels=3)
# test_input = torch.randn(1, 3, 448, 448)
# print(model(test_input).shape)
