import os
from datetime import datetime
from os.path import join

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_

from Datasets.ProstateBladderDataset import ProstateBladderDataset
from DetectionModels.RetinaNet import RetinaNet
from EarlyStopping.EarlyStopping import EarlyStopping
from Transformers import Transformers
from Utils import ArgsParser, Utils

########################################################################################################################
# Track run time of script.
########################################################################################################################
script_start = datetime.now()

########################################################################################################################
# Set seeds for semi-reproducibility.
########################################################################################################################
torch.manual_seed(2024)

########################################################################################################################
# Set device and empty cuda cache.
########################################################################################################################
torch.cuda.empty_cache()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

########################################################################################################################
# Fetch parser and set variables from parser args.
########################################################################################################################
args = ArgsParser.get_arg_parser()
train_images_path = args.train_images_path  # Path to training images directory.
train_labels_path = args.train_labels_path  # Path to training labels directory.
val_images_path = args.val_images_path  # Path to validation images directory.
val_labels_path = args.val_labels_path  # Path to validation labels directory.
save_path = args.save_path  # Path to saving directory, where models, loss plots, and validation results are stored.
if not os.path.isdir(save_path):
    os.makedirs(save_path)  # Make save_path into dir.
image_size = args.image_size  # Image size for resizing, used in training and validation.
batch_size = args.batch_size  # Batch size for loader.
total_epochs = args.epochs  # Training epochs.
warmup_epochs = args.warmup_epochs  # Epochs before early stop checks are done.
patience = args.patience  # Early stopping patience.
patience_delta = args.patience_delta  # Early stopping delta.
learning_rate = args.learning_rate  # Optimiser learning rate.
learning_restart = args.learning_restart  # Learning rate schedular restart frequency.
momentum = args.momentum  # Optimiser momentum.
weight_decay = args.weight_decay  # Optimiser weight decay.
box_weight = args.box_weight  # Weight applied to box loss.
cls_weight = args.box_weight  # Weight applied to classification loss.
oversampling_factor = args.oversampling_factor  # Oversampling factor.
save_latest = args.save_latest  # Save latest model as well as the best model.pth.

########################################################################################################################
# Transformers and datasets used by models.
########################################################################################################################
train_transforms = Transformers.get_training_transforms(image_size=image_size)
val_transforms = Transformers.get_validation_transforms(image_size=image_size)
train_dataset = ProstateBladderDataset(images_root=train_images_path, labels_root=train_labels_path,
                                       transforms=train_transforms, oversampling_factor=oversampling_factor)
val_dataset = ProstateBladderDataset(images_root=val_images_path, labels_root=val_labels_path,
                                     transforms=train_transforms)
# If you want to validate the dataset transforms visually, you can do it here.

########################################################################################################################
# Set up dataloaders.
########################################################################################################################
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                           collate_fn=lambda x: tuple(zip(*x)))
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                         collate_fn=lambda x: tuple(zip(*x)))

########################################################################################################################
# Set up model, optimiser, and learning rate scheduler (RetinaNet).
########################################################################################################################
print(f'Loading RetinaNet model...', end=' ')
custom_model = RetinaNet()
custom_model.model.to(device)
params = [p for p in custom_model.model.parameters() if p.requires_grad]
optimiser = torch.optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
lr_schedular = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0=learning_restart, eta_min=0)
print(f'Model loaded.')

########################################################################################################################
# Save training parameters to file and display to screen.
########################################################################################################################
with open(join(save_path, 'training_parameters.txt'), 'w') as save_file:
    save_file.write(f'Start time: {script_start}\n'
                    f'Training images path: {train_images_path}\n'
                    f'Training labels path: {train_labels_path}\n'
                    f'Validation image path: {val_images_path}\n'
                    f'Validation labels path: {val_labels_path}\n'
                    f'Save path: {save_path}\n'
                    f'Batch size: {batch_size}\n'
                    f'Epochs: {total_epochs}\n'
                    f'Warmup Epochs: {warmup_epochs}\n'
                    f'Device: {device}\n'
                    f'Patience: {patience}\n'
                    f'Patience delta: {patience_delta}\n'
                    f'Image Size: {image_size}\n'
                    f'Optimiser learning rate: {learning_rate}.\n'
                    f'Optimiser learning rate restart frequency: {learning_restart}\n'
                    f'Optimiser momentum: {momentum}\n'
                    f'Optimiser weight decay: {weight_decay}\n'
                    f'Oversampling factor: {oversampling_factor}\n'
                    f'Total training images in dataset (excluding dataset oversampling): {train_dataset.get_image_count()}\n'
                    f'Total training images in dataset (including dataset oversampling): {train_dataset.__len__()}\n'
                    f'Total validation images in dataset: {val_dataset.__len__()}\n'
                    f'Training Transformer count: {len(train_transforms.transforms)}\n'
                    f'Optimiser: {optimiser.__class__.__name__}\n'
                    f'Learning rate schedular: {lr_schedular.__class__.__name__}\n')

print('=====================================================================================================\n'
      f'Start time: {script_start}\n'
      f'Device: {device}\n'
      f'Training images path: {train_images_path}\n'
      f'Validation image path: {val_images_path}\n'
      f'Save path: {save_path}\n'
      f'Batch size: {batch_size}\n'
      f'Epochs: {total_epochs}\n'
      f'Total training images in dataset (including dataset oversampling): {train_dataset.__len__()}\n'
      f'Total validation images in dataset: {val_dataset.__len__()}\n')

########################################################################################################################
# Training and validation loop.
########################################################################################################################
def main():
    print(f'Starting training:')
    early_stopping = EarlyStopping(patience=patience, delta=patience_delta)
    training_losses = []
    val_losses = []
    training_learning_rates = []
    final_epoch_reached = 0
    for epoch in range(total_epochs):
        ################################################################################################################
        # Training step within epoch.
        ################################################################################################################
        custom_model.model.train()
        epoch_train_loss = [0, 0, 0]
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # Zero the gradients.
            optimiser.zero_grad()
            # Forward pass.
            loss_dict = custom_model.forward(images, targets)
            # Extract each loss.
            cls_loss = loss_dict['classification']
            bbox_loss = loss_dict['bbox_regression']
            # Calculate total loss, can apply weights here.
            losses = cls_loss * cls_weight + bbox_loss * box_weight
            # Check for NaNs or Infs.
            if torch.isnan(losses).any() or torch.isinf(losses).any():
                print("Loss has NaNs or Infs, skipping this batch")
                continue
            # Calculate gradients.
            losses.backward()
            # Apply gradient clipping.
            clip_grad_norm_(custom_model.model.parameters(), 2)
            optimiser.step()
            # Epoch loss per batch.
            epoch_train_loss[0] += losses.item()
            epoch_train_loss[1] += cls_loss.item()
            epoch_train_loss[2] += bbox_loss.item()
        # Step schedular once per epoch.
        lr_schedular.step()
        # Average epoch loss per image for all images.
        epoch_train_loss = [loss / len(train_loader) for loss in epoch_train_loss]
        training_losses.append(epoch_train_loss)
        training_learning_rates.append(lr_schedular.get_last_lr()[0])

        ################################################################################################################
        # Validation step within epoch.
        ################################################################################################################
        custom_model.model.eval()
        epoch_val_loss = [0, 0, 0]
        # No gradient calculations.
        with torch.no_grad():
            for images, targets in val_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                # Forward pass.
                loss_dict, _ = custom_model.forward(images, targets)
                # Extract each loss.
                cls_loss = loss_dict['classification']
                bbox_loss = loss_dict['bbox_regression']
                # Calculate total loss.
                losses = cls_loss + bbox_loss
                # Epoch loss per batch.
                epoch_val_loss[0] += losses.item()
                epoch_val_loss[1] += cls_loss.item()
                epoch_val_loss[2] += bbox_loss.item()

                # Average epoch loss per image for all images.
                epoch_val_loss = [loss / len(train_loader) for loss in epoch_val_loss]
                val_losses.append(epoch_val_loss)

        ################################################################################################################
        # Display training and validation losses (combined loss - training is weighted, validation is not).
        ################################################################################################################
        time_now = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
        print(f"\t{time_now}  -  Epoch {epoch + 1}/{total_epochs}, "
              f"Train Loss: {training_losses[-1][0]:0.3f}, "
              f"Val Loss: {val_losses[-1][0]:0.3f}, "
              f"Learning Rate: {lr_schedular.get_last_lr()[0]:0.6f},", end=' ', flush=True)

        ################################################################################################################
        # Check for early stopping. If patience reached, model is saved and final plots are made.
        ################################################################################################################
        final_epoch_reached = epoch
        if final_epoch_reached + 1 > warmup_epochs:
            early_stopping(epoch_val_loss, custom_model.model, epoch, optimiser, save_path)

        Utils.plot_losses(early_stopping.best_epoch + 1, training_losses, val_losses, training_learning_rates,
                          save_path)
        if early_stopping.early_stop:
            print('Patience reached, stopping early.')
            break
        else:
            print()

    ####################################################################################################################
    # On training complete, pass through validation images and plot them using best model (must be reloaded).
    ####################################################################################################################
    custom_model.model.load_state_dict(torch.load(join(save_path, 'model_best.pth'))['model_state_dict'])
    custom_model.model.eval()
    with torch.no_grad():
        counter = 0
        for images, targets in val_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            _, detections = custom_model.forward(images, targets)

            Utils.plot_validation_results(detections, images, 0, counter, save_path)

            counter += batch_size

    ####################################################################################################################
    # Save extra parameters to file.
    ####################################################################################################################
    script_end = datetime.now()
    run_time = script_end - script_start
    with open(join(save_path, 'training_parameters.txt'), 'a') as save_file:
        save_file.write(f'Final Epoch Reached: {final_epoch_reached}\n'
                        f'Best Epoch: {early_stopping.best_epoch}\n'
                        f"End time: {script_end.strftime('%Y-%m-%d  %H:%M:%S')}\n"
                        f'Total run time: {run_time}')

    print('Training completed.\n'
          f'Best Epoch: {early_stopping.best_epoch + 1}.\n'
          f"End time: {script_end.strftime('%Y-%m-%d  %H:%M:%S')}.\n"
          f'Total run time: {run_time}.')
    print('=====================================================================================================')

