from pprint import pprint
from os.path import join

import cv2
import numpy as np
import torch
from matplotlib import patches, pyplot as plt
from torchvision import tv_tensors


def draw_yolo_boxes_on_axis(B, S, ax, boxes, title, box_format):
    """
    Draws bounding boxes directly on the provided axis.

    - For 'xyxy' format: boxes are directly drawn.
    - For 'yolov1' format: boxes have been converted to XYXY before drawing.

    :param B: Boxes per grid cell.
    :param S: Grid size (SxS).
    :param ax: Matplotlib axis to draw on.
    :param boxes: Bounding boxes in either XYXY or YOLOv1 format.
    :param title: Title of the image.
    :param box_format: 'xyxy' for standard boxes, 'yolov1' for YOLO grid format.
    """
    ax.set_title(title)

    if box_format == 'xyxy':
        # Standard XYXY format (boxes.shape = (N, 4))
        for box in boxes:
            if len(box) != 4:
                continue
            x_min, y_min, x_max, y_max = box
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

    elif box_format == 'yolov1':
        # YOLOv1 format (boxes.shape = (S, S, B*5 + C))
        for i in range(S):
            for j in range(S):
                for b in range(B):
                    conf = boxes[i, j, 4 + b * 5]  # Object confidence
                    if conf > 0:
                        # Extract (x, y, w, h)
                        x, y, w, h = boxes[i, j, b * 5: b * 5 + 4]

                        # Convert relative (to cell) to absolute (image) coordinates
                        cell_size = 448 / S
                        abs_x = (j + x) * cell_size
                        abs_y = (i + y) * cell_size
                        abs_w = w * 448
                        abs_h = h * 448

                        # Convert to XYXY format
                        x_min = abs_x - abs_w / 2
                        y_min = abs_y - abs_h / 2
                        x_max = abs_x + abs_w / 2
                        y_max = abs_y + abs_h / 2

                        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                                 linewidth=2, edgecolor='blue', facecolor='none')
                        ax.add_patch(rect)
    return


def read_image(image_path):
    """
    Read an image at the given path for use by the yolo_dataset.ProstateBladderDataset class.

    Images are read as 1-channel greyscale, then the range is changed to [0, 1], then converted to 3-channel greyscale.

    :param image_path: Path to image that must be read from disk.
    :return: cv2 img.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) / 255
    img = np.stack([img, img, img], axis=-1)
    return img


def get_label_data(label_path, img_size, idx):
    """
    Get label data from label_path file.

    :param label_path: Path to label.txt file.
    :param img_size: Size of image.
    :param idx: Index of image.

    :return: Target dict, varies based on model type.
    """
    # Read the label file assuming YOLO format.
    boxes = []
    labels = []
    with open(label_path) as f:
        img_height, img_width = img_size[:2]
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])

            # Convert from YOLOv8 format to (x_min, y_min, x_max, y_max) format.
            x_min = (x_center - width / 2) * img_width
            y_min = (y_center - height / 2) * img_height
            x_max = (x_center + width / 2) * img_width
            y_max = (y_center + height / 2) * img_height

            boxes.append([x_min, y_min, x_max, y_max])

            labels.append(class_id)

    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.int64)

    image_id = torch.tensor([idx])
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    target = {
        "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(img_height, img_width)),
        "labels": labels,
        "image_id": image_id,
        "area": area
    }

    return target


def convert_xyxy_to_yolov1(target, S=7, B=2, C=1):
    """
    Converts target labels from XYXY format to YOLOv1 format, assuming an image size of (448, 448). Returns the target
    as a torch.tensor.

    :param target: Dictionary containing target data in XYXY format.
    :param S: Grid size (SxS).
    :param B: Number of bounding boxes per grid cell.
    :param C: Number of classes.
    :return: YOLOv1 label tensor of shape (S, S, B*5 + C).
    """
    img_width, img_height = 448, 448
    yolo_v1_label = np.zeros((S, S, B * 5 + C), dtype=np.float32)  # (7,7,11) for 1 class

    boxes = target["boxes"]
    labels = target["labels"]

    for i in range(len(boxes)):
        x_min, y_min, x_max, y_max = boxes[i]
        class_id = labels[i].item()

        # Convert absolute XYXY to YOLO (x_center, y_center, width, height)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        # Normalize by image size
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height

        # Determine which grid cell (row, col) the box belongs to
        grid_x = int(x_center * S)
        grid_y = int(y_center * S)

        # Compute relative coordinates within the grid cell
        x_rel = (x_center * S) - grid_x
        y_rel = (y_center * S) - grid_y
        w_rel = width
        h_rel = height

        # Assign values to YOLOv1 label tensor
        for b in range(B):
            conf_index = b * 5 + 4  # Confidence score index in the tensor
            if yolo_v1_label[grid_y, grid_x, conf_index] == 0:  # Check if this slot is empty
                yolo_v1_label[grid_y, grid_x, b * 5: (b + 1) * 5] = [x_rel, y_rel, w_rel, h_rel, 1]
                break  # Assign only one box per object

        # Assign class probabilities (One-Hot Encoding)
        class_start_idx = B * 5  # After all bounding boxes
        yolo_v1_label[grid_y, grid_x, class_start_idx:] = 0  # Reset class probabilities
        yolo_v1_label[grid_y, grid_x, class_start_idx + class_id] = 1  # Set class

    return torch.tensor(yolo_v1_label, dtype=torch.float32)


def plot_losses(best_epoch, training_losses, validation_losses, training_learning_rates, save_path):
    """
    Plot the training losses and validation losses along with the learning rates for YOLOv1.
    The figure will be saved at save_path/losses.png. The losses and rates should be in a list that grows as the
    epochs increase.

    :param best_epoch: Best epoch for special marker.
    :param training_losses: List of training losses (weighted).
    :param validation_losses: List of validation losses (weighted).
    :param training_learning_rates: List of optimiser learning rates.
    :param save_path: Save directory.
    """
    train_xy_loss = [i[1] for i in training_losses]
    train_wh_loss = [i[2] for i in training_losses]
    train_obj_conf_loss = [i[3] for i in training_losses]
    train_noobj_conf_loss = [i[4] for i in training_losses]
    train_class_loss = [i[5] for i in training_losses]
    train_training_combined_losses = [i[0] for i in training_losses]

    val_xy_loss = [i[1] for i in validation_losses]
    val_wh_loss = [i[2] for i in validation_losses]
    val_obj_conf_loss = [i[3] for i in validation_losses]
    val_noobj_conf_loss = [i[4] for i in validation_losses]
    val_class_loss = [i[5] for i in validation_losses]
    val_training_combined_losses = [i[0] for i in validation_losses]
    # Epochs start at 0.
    epochs = range(1, len(training_losses) + 1)
    _, ax = plt.subplots(nrows=2, ncols=6, layout='constrained', figsize=(27, 9), dpi=200)
    # Plot training xy losses.
    ax[0, 0].set_title('Training xy Losses')
    ax[0, 0].plot(epochs, train_xy_loss, marker='*')
    ax[0, 0].axvline(x=best_epoch, color='green', linestyle='--')
    ax[0, 0].set_xlabel('Epoch')
    ax[0, 0].set_ylabel('Loss')
    # Plot training wh losses.
    ax[0, 1].set_title('Training wh Losses')
    ax[0, 1].plot(epochs, train_wh_loss, marker='*')
    ax[0, 1].axvline(x=best_epoch, color='green', linestyle='--')
    # Plot training obj_conf losses.
    ax[0, 2].set_title('Training obj_conf Losses')
    ax[0, 2].plot(epochs, train_obj_conf_loss, marker='*')
    ax[0, 2].axvline(x=best_epoch, color='green', linestyle='--')
    # Plot training noobj_conf losses.
    ax[0, 3].set_title('Training noobj_conf Losses')
    ax[0, 3].plot(epochs, train_noobj_conf_loss, marker='*')
    ax[0, 3].axvline(x=best_epoch, color='green', linestyle='--')
    # Plot training class losses.
    ax[0, 4].set_title('Training Class Losses')
    ax[0, 4].plot(epochs, train_class_loss, marker='*')
    ax[0, 4].axvline(x=best_epoch, color='green', linestyle='--')
    # Plot weighted training losses with learning rates.
    ax[0, 5].set_title('Training Losses (weighted)\n'
                       'with Learning Rate')
    ax[0, 5].plot(epochs, train_training_combined_losses, marker='*')
    ax_lr = ax[0, 5].twinx()
    ax_lr.plot(epochs, [i * 1000 for i in training_learning_rates], color='red', label='learning rate')
    ax[0, 5].axvline(x=best_epoch, color='green', linestyle='--')
    ax_lr.set_ylabel('Learning Rate x10$^{-3}$')
    ax_lr.legend(loc='upper right')

    # Plot validation xy losses.
    ax[1, 0].set_title('Validation xy Losses')
    ax[1, 0].plot(epochs, val_xy_loss, marker='*')
    ax[1, 0].axvline(x=best_epoch, color='green', linestyle='--')
    ax[1, 0].set_xlabel('Epoch')
    ax[1, 0].set_ylabel('Loss')
    # Plot validation wh losses.
    ax[1, 1].set_title('Validation wh Losses')
    ax[1, 1].plot(epochs, val_wh_loss, marker='*')
    ax[1, 1].axvline(x=best_epoch, color='green', linestyle='--')
    # Plot validation obj_conf losses.
    ax[1, 2].set_title('Validation obj_conf Losses')
    ax[1, 2].plot(epochs, val_obj_conf_loss, marker='*')
    ax[1, 2].axvline(x=best_epoch, color='green', linestyle='--')
    # Plot validation noobj_conf losses.
    ax[1, 3].set_title('Validation noobj_conf Losses')
    ax[1, 3].plot(epochs, val_noobj_conf_loss, marker='*')
    ax[1, 3].axvline(x=best_epoch, color='green', linestyle='--')
    # Plot validation class losses.
    ax[1, 4].set_title('Validation Class Losses')
    ax[1, 4].plot(epochs, val_class_loss, marker='*')
    ax[1, 4].axvline(x=best_epoch, color='green', linestyle='--')
    # Plot weighted validation losses with learning rates.
    ax[1, 5].set_title('Validation Losses (weighted)\n'
                       'with Learning Rate')
    ax[1, 5].plot(epochs, val_training_combined_losses, marker='*')
    ax[1, 5].axvline(x=best_epoch, color='green', linestyle='--')

    plt.savefig(join(save_path, 'losses.png'))
    plt.close()


def plot_validation_results(validation_detections, validation_images, S, B, threshold, counter, train_mean, train_std,
                            save_path):
    """
    Draw input images with detected bounding boxes on them. Only the top scoring box of each label/class
    is displayed. Since FasterRCNN using label 0 for background and RetinaNet using label 0 for the first
    class, there is an offset that is set using starting_label (for selecting box colour).

    :param validation_detections: Detection returned by the model in eval() mode.
    :param validation_images: Images that were given to the model for detection.
    :param detection_count: Maximum number of detections (boxes) per class to be displayed.
    :param starting_label: Lowest label value (RetinaNet = 0, FasterRCNN = 1 since 0 is background).
    :param counter: Image counter, based on batch_size, for saving images with unique names while maintaining
                    validation dataset size.
    :param save_path: Save directory.
    """
    batch_number = counter
    for index, (predictions, image) in enumerate(zip(validation_detections, validation_images)):
        image = image.cpu().numpy().transpose((1, 2, 0)).copy()

        image = (image * train_std) + train_mean

        # Convert to uint8 format (0-255 range)
        image = (image * 255).astype(np.uint8) if image.dtype != np.uint8 else image

        height, width, _ = image.shape
        cell_size = width / S

        for i in range(S):
            for j in range(S):
                for b in range(B):
                    confidence = predictions[i, j, b * 5 + 4]
                    if confidence > threshold:
                        x, y, w, h = predictions[i, j, b * 5: b * 5 + 4]
                        x = (j + x.item()) * cell_size  # Convert to absolute x
                        y = (i + y.item()) * cell_size  # Convert to absolute y
                        w = w.item() * width  # Scale width to image size
                        h = h.item() * height  # Scale height to image size
                        x1 = int(x - w / 2)
                        y1 = int(y - h / 2)
                        x2 = int(x + w / 2)
                        y2 = int(y + h / 2)
                        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)  # Ensure valid image format
        plt.imshow(image, cmap='gray')
        plt.savefig(join(save_path, f'val_result_{batch_number}.png'))
        plt.close()

        batch_number += 1

    return

# def intersection_over_union()
