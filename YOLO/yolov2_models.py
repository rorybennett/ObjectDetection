"""
YOLO models set up as originally published (as close as possible anyways).
    - YOLOv2 based on this paper: https://arxiv.org/abs/1612.08242.
"""
import torch
from torch.nn import Conv2d, LeakyReLU, MaxPool2d, BatchNorm2d, Module, Sequential


class YOLOv2(Module):
    def __init__(self, num_classes, anchors, input_channels=3):
        """
        YOLOv1 Model. Batch normalisation seems necessary and was not a part of the original paper..

        Args:
        - num_classes (int): Number of classes to be detected.
        - anchors (list(list)): List of anchor coordinates from k-means clustering.
        - input_channels (int): Input channels of original image (RGB or 1-channel greyscale)
        """
        super(YOLOv2, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.input_channels = input_channels

        # CNN Backbone Darknet19.
        # Stage 1.
        self.stage1_conv1 = Sequential(Conv2d(3, 32, 3, 1, 1, bias=False), BatchNorm2d(32),
                                       LeakyReLU(0.1, inplace=True), MaxPool2d(2, 2))
        self.stage1_conv2 = Sequential(Conv2d(32, 64, 3, 1, 1, bias=False), BatchNorm2d(64),
                                       LeakyReLU(0.1, inplace=True), MaxPool2d(2, 2))
        self.stage1_conv3 = Sequential(Conv2d(64, 128, 3, 1, 1, bias=False), BatchNorm2d(128),
                                       LeakyReLU(0.1, inplace=True))
        self.stage1_conv4 = Sequential(Conv2d(128, 64, 1, 1, 0, bias=False), BatchNorm2d(64),
                                       LeakyReLU(0.1, inplace=True))
        self.stage1_conv5 = Sequential(Conv2d(64, 128, 3, 1, 1, bias=False), BatchNorm2d(128),
                                       LeakyReLU(0.1, inplace=True), MaxPool2d(2, 2))
        self.stage1_conv6 = Sequential(Conv2d(128, 256, 3, 1, 1, bias=False), BatchNorm2d(256),
                                       LeakyReLU(0.1, inplace=True))
        self.stage1_conv7 = Sequential(Conv2d(256, 128, 1, 1, 0, bias=False), BatchNorm2d(128),
                                       LeakyReLU(0.1, inplace=True))
        self.stage1_conv8 = Sequential(Conv2d(128, 256, 3, 1, 1, bias=False), BatchNorm2d(256),
                                       LeakyReLU(0.1, inplace=True), MaxPool2d(2, 2))
        self.stage1_conv9 = Sequential(Conv2d(256, 512, 3, 1, 1, bias=False), BatchNorm2d(512),
                                       LeakyReLU(0.1, inplace=True))
        self.stage1_conv10 = Sequential(Conv2d(512, 256, 1, 1, 0, bias=False), BatchNorm2d(256),
                                        LeakyReLU(0.1, inplace=True))
        self.stage1_conv11 = Sequential(Conv2d(256, 512, 3, 1, 1, bias=False), BatchNorm2d(512),
                                        LeakyReLU(0.1, inplace=True))
        self.stage1_conv12 = Sequential(Conv2d(512, 256, 1, 1, 0, bias=False), BatchNorm2d(256),
                                        LeakyReLU(0.1, inplace=True))
        self.stage1_conv13 = Sequential(Conv2d(256, 512, 3, 1, 1, bias=False), BatchNorm2d(512),
                                        LeakyReLU(0.1, inplace=True))

        # Stage 2.
        self.stage2_a_maxpl = MaxPool2d(2, 2)
        self.stage2_a_conv1 = Sequential(Conv2d(512, 1024, 3, 1, 1, bias=False),
                                         BatchNorm2d(1024), LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv2 = Sequential(Conv2d(1024, 512, 1, 1, 0, bias=False), BatchNorm2d(512),
                                         LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv3 = Sequential(Conv2d(512, 1024, 3, 1, 1, bias=False), BatchNorm2d(1024),
                                         LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv4 = Sequential(Conv2d(1024, 512, 1, 1, 0, bias=False), BatchNorm2d(512),
                                         LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv5 = Sequential(Conv2d(512, 1024, 3, 1, 1, bias=False), BatchNorm2d(1024),
                                         LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv6 = Sequential(Conv2d(1024, 1024, 3, 1, 1, bias=False), BatchNorm2d(1024),
                                         LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv7 = Sequential(Conv2d(1024, 1024, 3, 1, 1, bias=False), BatchNorm2d(1024),
                                         LeakyReLU(0.1, inplace=True))

        self.stage2_b_conv = Sequential(Conv2d(512, 64, 1, 1, 0, bias=False), BatchNorm2d(64),
                                        LeakyReLU(0.1, inplace=True))

        self.stage3_conv1 = Sequential(Conv2d(256 + 1024, 1024, 3, 1, 1, bias=False), BatchNorm2d(1024),
                                       LeakyReLU(0.1, inplace=True))
        self.stage3_conv2 = Conv2d(1024, len(self.anchors) * (5 + num_classes), 1, 1, 0, bias=False)

    def forward(self, x):
        x = self.stage1_conv1(x)
        x = self.stage1_conv2(x)
        x = self.stage1_conv3(x)
        x = self.stage1_conv4(x)
        x = self.stage1_conv5(x)
        x = self.stage1_conv6(x)
        x = self.stage1_conv7(x)
        x = self.stage1_conv8(x)
        x = self.stage1_conv9(x)
        x = self.stage1_conv10(x)
        x = self.stage1_conv11(x)
        x = self.stage1_conv12(x)
        x = self.stage1_conv13(x)

        residual = x

        x_1 = self.stage2_a_maxpl(x)
        x_1 = self.stage2_a_conv1(x_1)
        x_1 = self.stage2_a_conv2(x_1)
        x_1 = self.stage2_a_conv3(x_1)
        x_1 = self.stage2_a_conv4(x_1)
        x_1 = self.stage2_a_conv5(x_1)
        x_1 = self.stage2_a_conv6(x_1)
        x_1 = self.stage2_a_conv7(x_1)

        x_2 = self.stage2_b_conv(residual)
        batch_size, num_channel, height, width = x_2.data.size()
        x_2 = x_2.view(batch_size, int(num_channel / 4), height, 2, width, 2).contiguous()
        x_2 = x_2.permute(0, 3, 5, 1, 2, 4).contiguous()
        x_2 = x_2.view(batch_size, -1, int(height / 2), int(width / 2))

        x = torch.cat((x_1, x_2), 1)
        x = self.stage3_conv1(x)
        x = self.stage3_conv2(x)

        return x
