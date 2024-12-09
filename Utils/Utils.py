from os.path import join

import numpy as np
from matplotlib import pyplot as plt, patches

from . import box_colours


def plot_losses(best_epoch, training_losses, validation_losses, training_learning_rates, save_path):
    """
    Plot the training losses (combined weighted, cls, and bbox) and validation losses (combined, cls, and bbox)
    along with the learning rates. The figure will be saved at save_path/losses.png. The losses and rates should
    be in a list that grows as the epochs increase.

    :param best_epoch: Best epoch for special marker.
    :param training_losses: List of training losses (weighted), [combined, classification, regression].
    :param validation_losses: List of validation losses, [combined, classification, regression].
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
    ax_lr.set_ylabel('Learning Rate x10$^{-2}$')
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
    ax[1, 2].set_title('Validation Losses (unweighted)')
    ax[1, 2].plot(epochs, val_combined_losses, marker='*')
    ax[1, 2].axvline(x=best_epoch, color='green', linestyle='--', label='Best Validation Epoch')
    ax[1, 2].set_xlabel('Epoch')
    ax[1, 2].legend(loc='upper right')

    plt.savefig(join(save_path, 'losses.png'))
    plt.close()


def plot_validation_results(validation_detections, validation_images, starting_label, counter, save_path):
    """
    Draw input images with detected bounding boxes on them. Only the top scoring box of each label/class
    is displayed. Since FasterRCNN using label 0 for background and RetinaNet using label 0 for the first
    class, there is an offset that is set using starting_label (for selecting box colour).
    
    :param validation_detections: Detection returned by the model in eval() mode.
    :param validation_images: Images that were given to the model for detection.
    :param starting_label: Lowest label value (RetinaNet = 0, FasterRCNN = 1 since 0 is background).
    :param counter: Image counter, based on batch_size, for saving images with unique names while maintaining
                    validation dataset size.
    :param save_path: Save directory.
    """
    batch_number = counter
    # Since batches are used, detections per image are delt with incrementally.
    for index, output in enumerate(validation_detections):
        # Highest scoring box per label.
        highest_scoring_boxes = {}

        for i in range(len(output['scores'])):
            label = output['labels'][i].item()
            score = output['scores'][i].item()
            box = output['boxes'][i].tolist()

            if label not in highest_scoring_boxes:
                highest_scoring_boxes[label] = {'score': score, 'box': box}
            else:
                if score > highest_scoring_boxes[label]['score']:
                    highest_scoring_boxes[label] = {'score': score, 'box': box}

        # Sort by label.
        sorted_highest_scoring_boxes = dict(sorted(highest_scoring_boxes.items()))

        _, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(np.transpose(validation_images[index].to('cpu'), (1, 2, 0)))

        for label, result in sorted_highest_scoring_boxes.items():
            box = result['box']
            patch = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1,
                                      edgecolor=box_colours[label - starting_label], facecolor='none')
            ax.add_patch(patch)
            ax.text(box[0], box[1], f'{label}', ha='left', color=box_colours[label - starting_label], weight='bold',
                    va='bottom')

        plt.savefig(join(save_path, f'val_result_{batch_number}.png'))
        plt.close()
        batch_number += 1
