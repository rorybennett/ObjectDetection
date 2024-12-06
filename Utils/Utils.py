from os.path import join

import numpy as np
from matplotlib import pyplot as plt, patches


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
    ax_lr.plot(epochs, [i * 100 for i in training_learning_rates], color='red', label='learning rate')
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


def plot_validation_results(validation_detections, validation_images, counter, save_path):
    """
    Draw input images with detected bounding boxes on them. Only the top scoring box is drawn for each label class.
    todo Fix how this works, need to see exactly how the vlaidatoin detections looks
    :param validation_detections: Detection returned by the model in eval() mode.
    :param validation_images: Images that were given to the model for detection.
    :param counter: Image counter, based on batch_size.
    :param save_path: Save directory.
    """
    batch_number = counter
    for index, output in enumerate(validation_detections):
        top_box_prostate = None
        top_box_bladder = None
        boxes = output['boxes']
        labels = output['labels']

        for i in range(len(labels)):
            if labels[i] == 1 and top_box_prostate is None:
                top_box_prostate = boxes[i].to('cpu')
            elif labels[i] == 2 and top_box_bladder is None:
                top_box_bladder = boxes[i].to('cpu')

            # Break the loop if both top boxes are found.
            if top_box_prostate is not None and top_box_bladder is not None:
                break

        _, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(np.transpose(validation_images[index].to('cpu'), (1, 2, 0)))
        if top_box_prostate is not None:
            prostate_patch = patches.Rectangle((top_box_prostate[0], top_box_prostate[1]),
                                               top_box_prostate[2] - top_box_prostate[0],
                                               top_box_prostate[3] - top_box_prostate[1], linewidth=1,
                                               edgecolor='g', facecolor='none')
            ax.add_patch(prostate_patch)
        if top_box_bladder is not None:
            bladder_patch = patches.Rectangle((top_box_bladder[0], top_box_bladder[1]),
                                              top_box_bladder[2] - top_box_bladder[0],
                                              top_box_bladder[3] - top_box_bladder[1], linewidth=1,
                                              edgecolor='blue', facecolor='none')
            ax.add_patch(bladder_patch)

        plt.savefig(join(save_path, f'val_result_{batch_number}.png'))
        plt.close()
        batch_number += 1
