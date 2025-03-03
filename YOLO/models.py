"""
YOLOv1 models set up as originally published (as close as possible anyways).

YOLOv1: 24 Convolutional Layers

YOLOv1_fast: 9 Convolutional Layers.

Based on this paper: https://arxiv.org/pdf/1506.02640

Assumes an RGB input.
"""
import torch
import torch.nn as nn


class YOLOv1(nn.Module):
    def __init__(self, S=7, B=2, C=1):
        """
        YOLOv1 Model.

        Args:
        - S (int): Grid size (default 7x7).
        - B (int): Number of bounding boxes per grid cell (default 2).
        - C (int): Number of classes (default 1).
        """
        super(YOLOv1, self).__init__()
        self.S = S
        self.B = B
        self.C = C

        # CNN Backbone
        self.darknet_layers = nn.Sequential(
            # First convolutional layer (based on paper diagram).
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Second convolutional layer.
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Third batch of convolutional layers.
            nn.Conv2d(192, 128, kernel_size=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Fourth batch of convolutional layers (repeating the first 2 4 times).
            nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Fifth batch of convolutional layers.
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            # Final convolutional layers.
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1)
        )

        # Fully Connected Layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, S * S * (B * 5 + C)),
        )

    def forward(self, x):
        x = self.darknet_layers(x)
        x = self.fc_layers(x)  # Move to fully connected layers
        x = x.view(-1, self.S, self.S, self.B * 5 + self.C)
        return x


class YOLOv1Fast(nn.Module):
    def __init__(self, S=7, B=2, C=1):
        """
        YOLOv1Fast Model (similar to YOLOv1, with fewer (9) convolutional layers).

        Args:
        - S (int): Grid size (default 7x7).
        - B (int): Number of bounding boxes per grid cell (default 2).
        - C (int): Number of classes (default 1).
        """
        super(YOLOv1Fast, self).__init__()
        self.S = S
        self.B = B
        self.C = C

        # CNN Backbone
        self.darknet_layers = nn.Sequential(
            # First Conv Block
            nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3), nn.LeakyReLU(0.1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            # Second Conv Block
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            # Third Conv Block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            # Fourth Conv Block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            # Fifth Conv Block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            # Sixth Conv Block
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            # Final Convolutions (Keep Feature Map 7x7)
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.1),
        )

        # Fully Connected Layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 2048),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(2048, S * S * (B * 5 + C)),
        )

    def forward(self, x):
        x = self.darknet_layers(x)
        x = self.fc_layers(x)
        x = x.view(-1, self.S, self.S, self.B * 5 + self.C)

        return x


class YOLOv1Special(nn.Module):
    def __init__(self, S=7, B=2, C=1):
        """
        YOLOv1 Model.

        Args:
        - S (int): Grid size (default 7x7).
        - B (int): Number of bounding boxes per grid cell (default 2).
        - C (int): Number of classes (default 1).
        """
        super(YOLOv1Special, self).__init__()
        self.S = S
        self.B = B
        self.C = C

        # CNN Backbone
        self.darknet_layers = nn.Sequential(
            # First convolutional layer (based on paper diagram).
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Second convolutional layer.
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Third batch of convolutional layers.
            nn.Conv2d(192, 128, kernel_size=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Fourth batch of convolutional layers (repeating the first 2 4 times).
            nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Fifth batch of convolutional layers.
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            # Final convolutional layers.
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1)
        )

        # Fully Connected Layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (B * 5 + C)),
        )

    def forward(self, x):
        x = self.darknet_layers(x)
        x = self.fc_layers(x)  # Move to fully connected layers
        # x = x.view(-1, self.S, self.S, self.B * 5 + self.C)
        return x


########################################################################################################################
# Model tests.
########################################################################################################################
# model = YOLOv1Special(S=7, B=2, C=1)
# test_input = torch.randn(1, 3, 448, 448)
# print(model(test_input).shape)
