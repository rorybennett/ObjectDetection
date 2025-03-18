from os.path import join

import cv2
import numpy as np
import torch
from matplotlib import patches, pyplot as plt
from torchvision import tv_tensors

from Utils import box_colours




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
        "area": area,
        "img_size": img_size
    }

    return target


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


def plot_top_validation_results(validation_detections, validation_images, S, B, counter, train_mean, train_std,
                                save_path):
    """
    Draw validation images with only the highest scoring bounding box for each class.

    :param validation_detections: Model outputs in eval() mode.
    :param validation_images: Input images to the model.
    :param S: Grid size.
    :param B: Number of bounding boxes per cell.
    :param counter: For unique image filenames.
    :param train_mean: Mean for denormalisation.
    :param train_std: Std for denormalisation.
    :param save_path: Directory to save images.
    """
    batch_number = counter
    for index, (predictions, image) in enumerate(zip(validation_detections, validation_images)):
        image = image.cpu().numpy().transpose((1, 2, 0)).copy()
        image = (image * train_std) + train_mean
        image = (image * 255).astype(np.uint8)

        height, width, _ = image.shape
        cell_size = width / S

        # Dictionary to store top box per class: {class_id: (confidence, (x1, y1, x2, y2))}
        top_boxes = {}

        for i in range(S):
            for j in range(S):
                # Class probabilities are shared per cell (not per box)
                class_probs = predictions[i, j, B * 5:]

                for b in range(B):
                    offset = b * 5
                    x, y, w, h, confidence = predictions[i, j, offset:offset + 5]

                    class_scores = confidence.item() * class_probs
                    class_id = class_scores.argmax().item()
                    class_score = class_scores[class_id].item()

                    x = (j + x.item()) * cell_size
                    y = (i + y.item()) * cell_size
                    w = w.item() * width
                    h = h.item() * height
                    x1 = int(x - w / 2)
                    y1 = int(y - h / 2)
                    x2 = int(x + w / 2)
                    y2 = int(y + h / 2)

                    if class_id not in top_boxes or class_score > top_boxes[class_id][0]:
                        top_boxes[class_id] = (class_score, (x1, y1, x2, y2))

        # Draw top boxes
        _, ax = plt.subplots(1)
        ax.imshow(image)
        for class_id, (score, (x1, y1, x2, y2)) in top_boxes.items():
            patch = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=box_colours[class_id],
                                      facecolor='none')
            ax.add_patch(patch)
            ax.text(x1, y1, f'{class_id}: {score:.1f}', ha='left', color=box_colours[class_id], weight='bold',
                    va='bottom')

        plt.axis('off')
        plt.savefig(join(save_path, f'val_result_{batch_number}.png'), bbox_inches='tight', pad_inches=0)
        plt.close()

        batch_number += 1

    return
