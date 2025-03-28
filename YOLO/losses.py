import torch
from torch.nn import Module, MSELoss, BCELoss, CrossEntropyLoss


class YOLOv1Loss(Module):
    def __init__(self, S=7, B=2, C=1):
        """
        YOLOv1 Loss Function, created by GPT, based on the original paper's loss criterion.
        There was a problem with a negative sqrt, but that seems to have been sorted out.

        Args:
        - S (int): Grid size (default 7x7).
        - B (int): Number of bounding boxes per grid cell.
        - C (int): Number of classes.
        """
        super(YOLOv1Loss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.mse = MSELoss(reduction="sum")

    def forward(self, predictions, target):
        """
        Compute YOLOv1 Loss.

        Args:
        - predictions (Tensor): Shape (batch, S, S, B*5 + C).
        - target (Tensor): Shape (batch, S, S, B*5 + C).

        Returns:
        - loss (dict): Dictionary of individual loss values.
        """
        batch_size = predictions.shape[0]

        # Split into components
        pred_boxes = predictions[..., :self.B * 5].reshape(batch_size, self.S, self.S, self.B, 5)  # (x, y, w, h, conf)
        pred_classes = predictions[..., self.B * 5:]  # Class probabilities

        target_boxes = target[..., :self.B * 5].reshape(batch_size, self.S, self.S, self.B, 5)
        target_classes = target[..., self.B * 5:]

        # Object presence mask
        obj_mask = target[..., 4].unsqueeze(-1)  # (batch, 7, 7, 1)
        noobj_mask = 1 - obj_mask  # Opposite of object mask

        # Find best bbox (highest IoU)
        ious = self.iou(pred_boxes[..., :4], target_boxes[..., :4])  # Compute IoU
        best_box = torch.argmax(ious, dim=-1, keepdim=True)  # (batch, S, S, 1)
        best_box_mask = torch.zeros_like(ious, dtype=torch.bool).scatter_(-1, best_box, 1)

        # Select best bbox
        pred_best_boxes = pred_boxes[best_box_mask]  # (batch, S, S, 5)
        target_best_boxes = target_boxes[best_box_mask]

        # Localization Loss (Only for object cells)
        xy_loss = self.mse(pred_best_boxes[..., :2], target_best_boxes[..., :2])  # (x, y)

        wh_loss = self.mse(torch.flatten(torch.sign(pred_best_boxes[..., 2:4]) * torch.sqrt(torch.abs(pred_best_boxes[..., 2:4] + 1e-6)), end_dim=-2),
                           torch.flatten(torch.sqrt(target_best_boxes[..., 2:4]), end_dim=-2))

        # Confidence Loss
        obj_conf_loss = self.mse(pred_best_boxes[..., 4], target_best_boxes[..., 4])  # Only for objects
        noobj_conf_loss = self.mse(pred_boxes[..., 4] * noobj_mask, target_boxes[..., 4] * noobj_mask)  # No-object penalty

        # Classification Loss
        class_loss = self.mse(pred_classes * obj_mask, target_classes * obj_mask)

        return {"xy_loss": xy_loss / batch_size,
                "wh_loss": wh_loss / batch_size,
                "obj_conf_loss": obj_conf_loss / batch_size,
                "noobj_conf_loss": noobj_conf_loss / batch_size,
                "class_loss": class_loss / batch_size}

    def iou(self, box1, box2):
        """
        Compute Intersection over Union (IoU) between two bounding boxes. Keep all previous dimensions when selecting
        box corner points so when you select the best IoU, you can get the rest of the data immediately.

        Args:
        - box1 (Tensor): (batch, S, S, B, 4) -> (x, y, w, h)
        - box2 (Tensor): (batch, S, S, B, 4) -> (x, y, w, h)

        Returns:
        - IoU score (Tensor): (batch, S, S, B)
        """
        box1_x1 = box1[..., 0] - box1[..., 2] / 2
        box1_y1 = box1[..., 1] - box1[..., 3] / 2
        box1_x2 = box1[..., 0] + box1[..., 2] / 2
        box1_y2 = box1[..., 1] + box1[..., 3] / 2

        box2_x1 = box2[..., 0] - box2[..., 2] / 2
        box2_y1 = box2[..., 1] - box2[..., 3] / 2
        box2_x2 = box2[..., 0] + box2[..., 2] / 2
        box2_y2 = box2[..., 1] + box2[..., 3] / 2
        # Intersection area box.
        inter_x1 = torch.max(box1_x1, box2_x1)
        inter_y1 = torch.max(box1_y1, box2_y1)
        inter_x2 = torch.min(box1_x2, box2_x2)
        inter_y2 = torch.min(box1_y2, box2_y2)
        # Clamp is for when there is no intersection.
        intersection_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

        return intersection_area / (box1_area + box2_area - intersection_area + 1e-6)


class YOLOv2Loss(Module):
    def __init__(self, anchors, num_classes):
        """
        YOLOv2 Loss Function, created by GPT and a mixture of github repos. The original paper did not explicitly state the loss function so there
        is a bit of guesswork here. There was a problem with the assumed predictions outputs not matching with the targets, but that seems to have been
        sorted out. Sometimes the predictions show massive errors, to the point that when the box is plot on the image the result is so large it
        cannot be saved. Not sure what that is about at the moment.

        Args:
        - anchors (list): Anchors calculated using the training dataset and k-means clustering.
        - num_classes (int): Number of classes for detection.
        """
        super(YOLOv2Loss, self).__init__()
        self.anchors = torch.tensor(anchors).float()
        self.num_classes = num_classes
        self.mse_loss = MSELoss(reduction='sum')  # Sum over elements
        self.bce_loss = BCELoss(reduction='sum')  # Sum over elements
        self.ce_loss = CrossEntropyLoss(reduction='sum')  # Sum over elements

    def forward(self, predictions, targets):
        """
        Compute YOLOv2 loss.

        - predictions: (B, num_anchors, 5 + num_classes, H, W)
        - targets: (B, num_anchors, 5 + num_classes, H, W)
        - grid_size: Feature map size (e.g., 13 for 416x416 input)
        """
        B, _, _, _ = predictions.shape

        ########################################################################################################################################################
        # Extract individual components from predictions.
        ########################################################################################################################################################
        tx = predictions[:, :, 0, :, :]  # Center x
        ty = predictions[:, :, 1, :, :]  # Center y
        tw = predictions[:, :, 2, :, :]  # Width
        th = predictions[:, :, 3, :, :]  # Height
        objectness = predictions[:, :, 4, :, :]  # Objectness score

        ########################################################################################################################################################
        # Loss calculations.
        ########################################################################################################################################################
        # Create a mask that will result in calculations only being done on cells that are supposed to have boxes, otherwise losses skyrocket.
        obj_mask = (targets[:, :, 4, :, :] == 1).float()
        coord_loss = self.mse_loss(tx * obj_mask, targets[:, :, 0, :, :]) + self.mse_loss(ty * obj_mask, targets[:, :, 1, :, :])

        size_loss = self.mse_loss(tw * obj_mask, targets[:, :, 2, :, :]) + self.mse_loss(th * obj_mask, targets[:, :, 3, :, :])

        obj_loss = self.bce_loss(torch.sigmoid(objectness) * obj_mask, targets[:, :, 4, :, :])

        class_loss = self.ce_loss(predictions[:, :, 5:, :, :], targets[:, :, 5:, :, :])

        # Normalize by batch size.
        coord_loss /= B
        size_loss /= B
        obj_loss /= B
        class_loss /= B

        return {'coord_loss': coord_loss,
                'size_loss': size_loss,
                'obj_loss': obj_loss,
                'class_loss': class_loss}
