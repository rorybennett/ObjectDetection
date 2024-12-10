"""
Custom FasterRCNN class. Makes use of a ResNet50 with fpn backbone, from the fasterrcnn_resnet50_fpn_v2 setup.

I think this is set up correctly, but I could be grossly mistaken.

No pretrained weights are used, and the number of classes is equal to detection classes + 1 (for background).
"""
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2


class FasterRCNN:
    def __init__(self, num_classes):
        self.model = fasterrcnn_resnet50_fpn_v2(weights=None, num_classes=num_classes)

    def forward(self, images, targets=None):
        """
        Altered forward pass. If in training mode then the standard forward function is called, returning only
        losses. If in evaluation mode then the detections and losses are returned if targets is not None.

        :param images: Input images.
        :param targets: Desired targets, can be None.
        :return: losses or losses + detections or detections.
        """
        # If model.train() call standard training function.
        if self.model.training:
            return self.model(images, targets)
        else:
            if targets is not None:
                # Set the model to training mode temporarily to get losses (with torch.no_grad(), should already be set
                # in the calling script).
                self.model.train()
                losses = self.model.forward(images, targets)
                # Set the model back to evaluation mode
                self.model.eval()
                detections = self.model.forward(images)
                return losses, detections
            else:
                # Model already in evaluation mode.
                return self.model.forward(images)
