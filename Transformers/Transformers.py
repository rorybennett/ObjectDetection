"""
Transforms used during training and validation.

Can be altered to suit specific needs. Since this project was created for ultrasound images, the main transforms
deal with physical transformations, with only Gaussian transformations done on the "colour" level.
"""

import torch
from torchvision.transforms import v2


def get_training_transforms(image_size=(600, 600)):
    return v2.Compose([
        v2.Resize(image_size),
        # Original transforms.
        v2.RandomAffine(degrees=30, shear=15, translate=(0.1, 0.1), scale=(0.6, 1.2)),
        v2.RandomHorizontalFlip(p=0.2),
        v2.RandomErasing(0.5, scale=(0.02, 0.08)),
        v2.ToDtype(torch.float32, scale=True),
        v2.GaussianNoise(mean=0, sigma=0.2),
        v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2)),
        v2.Grayscale(num_output_channels=3),
        v2.ToImage()
    ])


def get_validation_transforms(image_size=(600, 600)):
    return v2.Compose([
        v2.Resize(image_size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Grayscale(num_output_channels=3),
        v2.ToImage()
    ])
