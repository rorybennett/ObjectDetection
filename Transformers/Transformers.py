"""
Transforms used during training and validation.

Can be altered to suit specific needs.
"""

import torch
from torchvision.transforms import v2


class TrainingTransformers:
    def __init__(self, image_size=600):
        self.transformers = v2.Compose([
            v2.Resize((image_size, image_size)),
            v2.RandomAffine(degrees=30, shear=15, translate=(0.1, 0.1), scale=(0.6, 1.2)),
            v2.RandomHorizontalFlip(p=0.2),
            v2.RandomErasing(0.5, scale=(0.02, 0.08)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

    def get_transformers(self):
        return self.transformers


class ValidationTransformers:
    def __init__(self, image_size=600):
        self.transformers = v2.Compose([
            v2.Resize((image_size, image_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

    def get_transformers(self):
        return self.transformers
