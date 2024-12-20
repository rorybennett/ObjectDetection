"""
Transforms used during training.

Can be altered to suit specific needs. Since this project was created for ultrasound images, the main transforms
deal with physical transformations, with only Gaussian transformations done on the "colour" level.
"""

from torchvision.transforms import v2


def get_training_transforms():
    return v2.Compose([
        # Original transforms.
        v2.RandomErasing(0.5, scale=(0.02, 0.08)),
        v2.RandomAffine(degrees=30, shear=15, translate=(0.1, 0.1), scale=(0.8, 1.1)),
        v2.RandomHorizontalFlip(p=0.2),
        v2.GaussianNoise(mean=0, sigma=0.1, clip=False),
        v2.GaussianBlur(5)
    ])
