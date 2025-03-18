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
        - loss (Tensor): Total loss value.
        """
        batch_size = predictions.shape[0]

        # # Reshape tensors for clarity
        # predictions = predictions.view(batch_size, self.S, self.S, self.B * 5 + self.C)
        # target = target.view(batch_size, self.S, self.S, self.B * 5 + self.C)

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
        xy_loss = self.mse(pred_best_boxes[..., :2],
                           target_best_boxes[..., :2])  # (x, y)

        wh_loss = self.mse(
            torch.flatten(torch.sign(pred_best_boxes[..., 2:4]) * torch.sqrt(
                torch.abs(pred_best_boxes[..., 2:4] + 1e-6)), end_dim=-2),
            torch.flatten(torch.sqrt(target_best_boxes[..., 2:4]), end_dim=-2)
        )

        # Confidence Loss
        obj_conf_loss = self.mse(pred_best_boxes[..., 4], target_best_boxes[..., 4])  # Only for objects
        noobj_conf_loss = self.mse(pred_boxes[..., 4] * noobj_mask,
                                   target_boxes[..., 4] * noobj_mask)  # No-object penalty

        # Classification Loss
        class_loss = self.mse(pred_classes * obj_mask, target_classes * obj_mask)

        return {
            "xy_loss": xy_loss / batch_size,
            "wh_loss": wh_loss / batch_size,
            "obj_conf_loss": obj_conf_loss / batch_size,
            "noobj_conf_loss": noobj_conf_loss / batch_size,
            "class_loss": class_loss / batch_size
        }

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
        super(YOLOv2Loss, self).__init__()
        self.anchors = torch.tensor(anchors).float()
        self.num_classes = num_classes

    def forward(self, predictions, targets, grid_size):
        """
        - predictions: (B, num_anchors, 5 + num_classes, H, W)
        - targets: Ground truth tensor (B, num_anchors, 5 + num_classes, H, W)
        - grid_size: The spatial size of the feature map (e.g., 13x13 for 416x416 input)
        """
        device = predictions.device
        self.anchors = self.anchors.to(device)

        # Extract individual predictions
        tx = predictions[:, :, 0, :, :]
        ty = predictions[:, :, 1, :, :]
        tw = predictions[:, :, 2, :, :]
        th = predictions[:, :, 3, :, :]
        objectness = predictions[:, :, 4, :, :]
        class_scores = predictions[:, :, 5:, :, :]

        # Apply transformations
        tx = torch.sigmoid(tx)  # Center x offset
        ty = torch.sigmoid(ty)  # Center y offset
        objectness = torch.sigmoid(objectness)  # Objectness score
        class_scores = torch.softmax(class_scores, dim=2)  # Class probabilities

        # Compute actual bounding box coordinates
        bx = tx + torch.arange(grid_size, device=device).float().view(1, 1, 1, -1)
        by = ty + torch.arange(grid_size, device=device).float().view(1, 1, -1, 1)
        bw = torch.exp(tw) * self.anchors[:, 0].view(1, -1, 1, 1)
        bh = torch.exp(th) * self.anchors[:, 1].view(1, -1, 1, 1)

        # Compute loss components
        coord_loss = MSELoss()(bx, targets[:, :, 0, :, :]) + MSELoss()(by, targets[:, :, 1, :, :])
        size_loss = MSELoss()(bw, targets[:, :, 2, :, :]) + MSELoss()(bh, targets[:, :, 3, :, :])
        obj_loss = BCELoss()(objectness, targets[:, :, 4, :, :])
        class_loss = CrossEntropyLoss()(class_scores.permute(0, 1, 3, 4, 2).contiguous().view(-1, self.num_classes),
                                        targets[:, :, 5:, :, :].argmax(dim=2).view(-1))

        return coord_loss, size_loss, obj_loss, class_loss
