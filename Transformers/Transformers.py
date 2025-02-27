"""
Transforms used during training.

Can be altered to suit specific needs. Since this project was created for ultrasound images, the main transforms
deal with physical transformations, with only Gaussian transformations done on the "colour" level.
"""

from torchvision.transforms import v2


def get_retinanet_transforms():
    """
    Transformations used by RetinaNet pipeline. Resizing and normalising take place within the RetinaNet model, so
    it does not need to happen here.

    Returns: Transformations.
    """
    return v2.Compose([
        # Original transforms.
        v2.RandomErasing(0.5, scale=(0.02, 0.08)),
        v2.RandomAffine(degrees=30, shear=15, translate=(0.1, 0.1), scale=(0.8, 1.1)),
        v2.RandomHorizontalFlip(p=0.2),
        v2.GaussianNoise(mean=0, sigma=0.1, clip=False),
        v2.GaussianBlur(5)
    ])


def get_fasterrcnn_transforms():
    """
    Transformations used by FasterRCNN pipeline. Resizing and normalising take place within the FasterRCNN model, so
    it does not need to happen here.

    Returns: Transformations.
    """
    return v2.Compose([
        # Original transforms.
        v2.RandomErasing(0.5, scale=(0.02, 0.08)),
        v2.RandomAffine(degrees=30, shear=15, translate=(0.1, 0.1), scale=(0.8, 1.1)),
        v2.RandomHorizontalFlip(p=0.2),
        v2.GaussianNoise(mean=0, sigma=0.1, clip=False),
        v2.GaussianBlur(5)
    ])


def get_yolov1_transforms():
    """
    Transformations used by YOLO pipeline. Resizing and normalising take place within the dataset class, so
    it does not need to happen here.

    Returns: Transformations.
    """
    return v2.Compose([
        # Original transforms.
        v2.RandomErasing(0.5, scale=(0.02, 0.08)),
        v2.RandomAffine(degrees=30, shear=15, translate=(0.1, 0.1), scale=(0.8, 1.1)),
        v2.RandomHorizontalFlip(p=0.2),
        v2.GaussianNoise(mean=0, sigma=0.2, clip=False),
        v2.GaussianBlur(5)
    ])
