"""
Custom RetinaNet class. Makes use of a ResNet50 with fpn backbone, from the retinanet_resnet50_fpn_v2 setup.

I think this is set up correctly, but I could be grossly mistaken.

No pretrained weights are used, and the number of classes is equal to detection classes.
"""
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from . import retinanet_backbones as backbones


class CustomRetinaNet:
    def __init__(self, backbone_type, weights=None, num_classes=None, **kwargs):
        if backbone_type == backbones['retinanet_resnet50_fpn_v2']:
            self.model = retinanet_resnet50_fpn_v2(weights=weights, num_classes=num_classes, **kwargs)
        elif backbone_type == backbones['mobilenet_v2']:
            backbone = torchvision.models.mobilenet_v2(weights=weights).features
            backbone.out_channels = 1280
            anchor_generator = AnchorGenerator(
                sizes=((32, 64, 128, 256, 512),),
                aspect_ratios=((0.5, 1.0, 2.0),)
            )
            self.model = RetinaNet(backbone,
                                   num_classes=num_classes,
                                   anchor_generator=anchor_generator, **kwargs)
        else:
            print(f"\n\n{backbone_type} is not available, choose between:\n{backbones.keys()}\n\n")
            exit()

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
                # Set the model to training mode temporarily to get losses (with torch.no_grad() should already be set
                # in the calling script.
                self.model.train()
                losses = self.model.forward(images, targets)
                # Set the model back to evaluation mode
                self.model.eval()
                detections = self.model.forward(images)
                return losses, detections
            else:
                # Model already in evaluation mode.
                return self.model.forward(images)
