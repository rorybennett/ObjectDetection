from os.path import join

from matplotlib import pyplot as plt


def plot_losses(best_epoch, training_losses, validation_losses, training_learning_rates, save_path):
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
