from os.path import join

from matplotlib import pyplot as plt


def plot_losses(best_epoch, training_losses, validation_losses, training_learning_rates, save_path):
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
