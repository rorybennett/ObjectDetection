import platform
from collections import defaultdict
from os.path import join

import numpy as np
import torch
from matplotlib import pyplot as plt, patches

from . import box_colours


def plot_losses_retinanet(best_epoch, training_losses, validation_losses, training_learning_rates, save_path):
    """
    Plot the training losses (combined weighted, cls, and bbox) and validation losses (combined weights, cls, and bbox)
    along with the learning rates for RetinaNet. The figure will be saved at save_path/losses.png. The losses and rates
    should be in a list that grows as the epochs increase.

    :param best_epoch: Best epoch for special marker.
    :param training_losses: List of training losses (weighted), [combined, classification, regression].
    :param validation_losses: List of validation losses (weighted), [combined, classification, regression].
    :param training_learning_rates: List of optimiser learning rates.
    :param save_path: Save directory.
    """
    training_cls_losses = [i[1] for i in training_losses]
    training_bbox_losses = [i[2] for i in training_losses]
    training_combined_losses = [i[0] for i in training_losses]
    val_cls_losses = [i[1] for i in validation_losses]
    val_bbox_losses = [i[2] for i in validation_losses]
    val_combined_losses = [i[0] for i in validation_losses]
    # Epochs start at 0.
    epochs = range(1, len(training_losses) + 1)
    _, ax = plt.subplots(nrows=2, ncols=3, layout='constrained', figsize=(16, 9), dpi=200)
    # Plot training classification losses.
    ax[0, 0].set_title('Training Classification Losses')
    ax[0, 0].plot(epochs, training_cls_losses, marker='*')
    ax[0, 0].axvline(x=best_epoch, color='green', linestyle='--')
    ax[0, 0].set_xlabel('Epoch')
    ax[0, 0].set_ylabel('Loss')
    # Plot bounding box regression losses.
    ax[0, 1].set_title('Training Box Regression Losses')
    ax[0, 1].plot(epochs, training_bbox_losses, marker='*')
    ax[0, 1].axvline(x=best_epoch, color='green', linestyle='--')
    # Plot weighted training losses with learning rates.
    ax[0, 2].set_title('Training Losses (weighted)\n'
                       'with Learning Rate')
    ax[0, 2].plot(epochs, training_combined_losses, marker='*')
    ax_lr = ax[0, 2].twinx()
    ax_lr.plot(epochs, [i * 1000 for i in training_learning_rates], color='red', label='learning rate')
    ax[0, 2].axvline(x=best_epoch, color='green', linestyle='--')
    ax_lr.set_ylabel('Learning Rate x10$^{-3}$')
    ax_lr.legend(loc='upper right')
    # Plot validation classification losses.
    ax[1, 0].set_title('Validation Classification Losses')
    ax[1, 0].plot(epochs, val_cls_losses, marker='*')
    ax[1, 0].axvline(x=best_epoch, color='green', linestyle='--')
    ax[1, 0].set_xlabel('Epoch')
    ax[1, 0].set_ylabel('Loss')
    # Plot validation bounding box regression losses.
    ax[1, 1].set_title('Validation Box Regression Losses')
    ax[1, 1].plot(epochs, val_bbox_losses, marker='*')
    ax[1, 1].axvline(x=best_epoch, color='green', linestyle='--')
    ax[1, 1].set_xlabel('Epoch')
    # Plot unweighted validation losses.
    ax[1, 2].set_title('Validation Losses (weighted)')
    ax[1, 2].plot(epochs, val_combined_losses, marker='*')
    ax[1, 2].axvline(x=best_epoch, color='green', linestyle='--', label='Best Validation Epoch')
    ax[1, 2].set_xlabel('Epoch')
    ax[1, 2].legend(loc='upper right')

    plt.savefig(join(save_path, 'losses.png'))
    plt.close()


def plot_losses_fasterrcnn(best_epoch, training_losses, validation_losses, training_learning_rates, save_path):
    """
    Plot the training losses (combined weighted, cls, bbox, objectness, rpn_box) and validation losses
    (combined weighted, cls, bbox, objectness, rpn_box) along with the learning rates for FasterRCNN.
    The figure will be saved at save_path/losses.png. The losses and rates should be in a list that grows as the
    epochs increase.

    :param best_epoch: Best epoch for special marker.
    :param training_losses: List of training losses (weighted), [combined, classification, regression, objectness,
                            rpn_box].
    :param validation_losses: List of validation losses (weighted), [combined, classification, regression, objectness,
                              rpn_box].
    :param training_learning_rates: List of optimiser learning rates.
    :param save_path: Save directory.
    """
    training_cls_losses = [i[1] for i in training_losses]
    training_bbox_losses = [i[2] for i in training_losses]
    training_objectness_losses = [i[3] for i in training_losses]
    training_rpn_box_losses = [i[4] for i in training_losses]
    training_combined_losses = [i[0] for i in training_losses]
    val_cls_losses = [i[1] for i in validation_losses]
    val_bbox_losses = [i[2] for i in validation_losses]
    val_objectness_losses = [i[3] for i in validation_losses]
    val_rpn_box_losses = [i[4] for i in validation_losses]
    val_combined_losses = [i[0] for i in validation_losses]
    # Epochs start at 0.
    epochs = range(1, len(training_losses) + 1)
    _, ax = plt.subplots(nrows=2, ncols=5, layout='constrained', figsize=(22, 9), dpi=200)
    # Plot training classification losses.
    ax[0, 0].set_title('Training Classification Losses')
    ax[0, 0].plot(epochs, training_cls_losses, marker='*')
    ax[0, 0].axvline(x=best_epoch, color='green', linestyle='--')
    ax[0, 0].set_xlabel('Epoch')
    ax[0, 0].set_ylabel('Loss')
    # Plot training bounding box regression losses.
    ax[0, 1].set_title('Training Box Regression Losses')
    ax[0, 1].plot(epochs, training_bbox_losses, marker='*')
    ax[0, 1].axvline(x=best_epoch, color='green', linestyle='--')
    # Plot training objectness losses.
    ax[0, 2].set_title('Training Objectness Losses')
    ax[0, 2].plot(epochs, training_objectness_losses, marker='*')
    ax[0, 2].axvline(x=best_epoch, color='green', linestyle='--')
    # Plot training rpn box regression losses.
    ax[0, 3].set_title('Training RPN Box Losses')
    ax[0, 3].plot(epochs, training_rpn_box_losses, marker='*')
    ax[0, 3].axvline(x=best_epoch, color='green', linestyle='--')
    # Plot weighted training losses with learning rates.
    ax[0, 4].set_title('Training Losses (weighted)\n'
                       'with Learning Rate')
    ax[0, 4].plot(epochs, training_combined_losses, marker='*')
    ax_lr = ax[0, 4].twinx()
    ax_lr.plot(epochs, [i * 1000 for i in training_learning_rates], color='red', label='learning rate')
    ax[0, 4].axvline(x=best_epoch, color='green', linestyle='--')
    ax_lr.set_ylabel('Learning Rate x10$^{-3}$')
    ax_lr.legend(loc='upper right')
    # Plot validation classification losses.
    ax[1, 0].set_title('Validation Classification Losses')
    ax[1, 0].plot(epochs, val_cls_losses, marker='*')
    ax[1, 0].axvline(x=best_epoch, color='green', linestyle='--')
    ax[1, 0].set_xlabel('Epoch')
    ax[1, 0].set_ylabel('Loss')
    # Plot validation bounding box regression losses.
    ax[1, 1].set_title('Validation Box Regression Losses')
    ax[1, 1].plot(epochs, val_bbox_losses, marker='*')
    ax[1, 1].axvline(x=best_epoch, color='green', linestyle='--')
    ax[1, 1].set_xlabel('Epoch')
    # Plot validation objectness losses.
    ax[1, 2].set_title('Validation Objectness Losses')
    ax[1, 2].plot(epochs, val_objectness_losses, marker='*')
    ax[1, 2].axvline(x=best_epoch, color='green', linestyle='--')
    # Plot validation rpn box regression losses.
    ax[1, 3].set_title('Validation RPN Box Losses')
    ax[1, 3].plot(epochs, val_rpn_box_losses, marker='*')
    ax[1, 3].axvline(x=best_epoch, color='green', linestyle='--')
    # Plot unweighted validation losses.
    ax[1, 4].set_title('Validation Losses (weighted)')
    ax[1, 4].plot(epochs, val_combined_losses, marker='*')
    ax[1, 4].axvline(x=best_epoch, color='green', linestyle='--', label='Best Validation Epoch')
    ax[1, 4].set_xlabel('Epoch')
    ax[1, 4].legend(loc='upper right')

    plt.savefig(join(save_path, 'losses.png'))
    plt.close()


def plot_validation_results(validation_detections, validation_images, starting_label, detection_count, counter,
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
    # Since batches are used, detections per image are delt with incrementally.
    for index, output in enumerate(validation_detections):
        # Highest scoring box per label.
        highest_scoring_boxes = defaultdict(lambda: {'scores': [], 'boxes': []})

        labels = output['labels'].cpu().tolist()
        scores = output['scores'].cpu().tolist()
        boxes = output['boxes'].cpu().tolist()

        # Group detections by class
        class_detections = defaultdict(list)
        for label, score, box in zip(labels, scores, boxes):
            class_detections[label].append((score, box))

        # Sort and select top x boxes for each class
        for label, items in class_detections.items():
            # Sort items by score in descending order
            sorted_items = sorted(items, key=lambda item: item[0], reverse=True)
            # Select the top scoring items.
            top_items = sorted_items[:detection_count]
            # Store the scores and boxes in the dictionary.
            highest_scoring_boxes[label]['scores'] = [item[0] for item in top_items]
            highest_scoring_boxes[label]['boxes'] = [item[1] for item in top_items]

        _, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(np.transpose(validation_images[index].to('cpu'), (1, 2, 0))[:, :, 0], cmap='gray')
        for label, label_results in highest_scoring_boxes.items():
            scores = label_results['scores']
            boxes = label_results['boxes']
            for j, s in enumerate(scores):
                box = boxes[j]
                patch = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1,
                                          edgecolor=box_colours[label - starting_label], facecolor='none')
                ax.add_patch(patch)
                ax.text(box[0], box[1], f'{label}: {s:0.1f}', ha='left', color=box_colours[label - starting_label],
                        weight='bold', va='bottom')

        plt.savefig(join(save_path, f'val_result_{batch_number}.png'))
        plt.close()
        batch_number += 1


def get_device_name():
    """
    Return the name of the device being used by torch (GPU name or CPU name.

    :return: Name of torch device.
    """
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(torch.device('cuda'))
    else:
        return platform.processor()
