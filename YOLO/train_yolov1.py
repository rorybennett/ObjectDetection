import os
from datetime import datetime
from os.path import join
from pprint import pprint, pformat

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from Datasets.yolo_dataset import ProstateBladderDataset as PBD
from EarlyStopping.EarlyStopping import EarlyStopping
from Transformers import Transformers
from Utils import general_utils, yolo_utils
from Utils.arg_parsers import YOLOv1ArgParser
from YOLO.losses import YOLOv1Loss
from YOLO.models import YOLOv1, YOLOv1Special

########################################################################################################################
# Track run time of script.
########################################################################################################################
script_start = datetime.now()

########################################################################################################################
# Set seeds for semi-reproducibility.
########################################################################################################################
seed = 2025
torch.manual_seed(seed)

########################################################################################################################
# Set device and empty cuda cache.
########################################################################################################################
torch.cuda.empty_cache()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

########################################################################################################################
# Fetch parser and set variables from parser args.
########################################################################################################################
arg_parser = YOLOv1ArgParser()
args = arg_parser.parse_args()
train_images_path = args.train_images_path  # Path to training images directory.
train_labels_path = args.train_labels_path  # Path to training labels directory.
val_images_path = args.val_images_path  # Path to validation images directory.
val_labels_path = args.val_labels_path  # Path to validation labels directory.
save_path = args.save_path  # Path to saving directory, where models, loss plots, and validation results are stored.
if not os.path.isdir(save_path):
    os.makedirs(save_path)  # Make save_path into dir.
batch_size = args.batch_size  # Batch size for loader.
num_classes = args.number_of_classes  # Number of classes present in training dataset.
total_epochs = args.epochs  # Training epochs.
warmup_epochs = args.warmup_epochs  # Epochs before early stop checks are done.
patience = args.patience  # Early stopping patience.
patience_delta = args.patience_delta  # Early stopping delta.
learning_rate = args.optimiser_learning_rate  # Optimiser learning rate.
learning_restart = args.optimiser_learning_rate_restart  # Learning rate scheduler restart frequency.
momentum = args.optimiser_momentum  # Optimiser momentum.
weight_decay = args.optimiser_weight_decay  # Optimiser weight decay.
oversampling_factor = args.oversampling_factor  # Oversampling factor.
save_latest = args.save_latest  # Save latest model as well as the best model.pth.
S = args.yolo_s  # YOLOv1 grid size.
B = args.yolo_b  # YOLOv1 cell box prediction count.
loss_weight = args.loss_weight  # Weight for localization loss.
conf_weight = args.conf_weight  # Weight for confidence loss on no-object cells.

########################################################################################################################
# Transformers and datasets used by models.
########################################################################################################################
train_transforms = Transformers.get_yolov1_transforms()

train_mean, train_std = PBD(images_root=train_images_path, labels_root=train_labels_path).get_mean_and_std()
train_dataset = PBD(images_root=train_images_path, labels_root=train_labels_path, transforms=train_transforms,
                    oversampling_factor=oversampling_factor, train_mean=train_mean, train_std=train_std, S=S, B=B,
                    C=num_classes)
val_dataset = PBD(images_root=val_images_path, labels_root=val_labels_path, train_mean=train_mean, train_std=train_std,
                  S=S, B=B, C=num_classes)
# If you want to validate the dataset transforms visually, you can do it here.
# for i in range(len(train_dataset)):
#     train_dataset.display_transforms(i)
#
# exit()
########################################################################################################################
# Set up dataloaders.
########################################################################################################################
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

########################################################################################################################
# Set up model, optimiser, learning rate scheduler, and loss criterion.
########################################################################################################################
print(f'Loading YOLOv1 model...', end=' ')
# Loss function
criterion = YOLOv1Loss(S=S, B=B, C=num_classes).to(device)
# Initialize model
yolo_model = YOLOv1(S=S, B=B, C=num_classes).to(device)
# Optimizer
params = [p for p in yolo_model.parameters() if p.requires_grad]
optimiser = torch.optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
lr_schedular = CosineAnnealingWarmRestarts(optimiser, T_0=learning_restart, eta_min=0)
print(f'Model loaded.')

########################################################################################################################
# Save training parameters to file and display to screen.
########################################################################################################################
arg_parser.save_args(save_path, args, script_start=script_start, seed=seed, device=device,
                     device_name=general_utils.get_device_name(), train_mean=train_mean, train_std=train_std,
                     train_dataset_len=train_dataset.__len__(), val_dataset_len=val_dataset.__len__(),
                     transformer_count=len(train_transforms.transforms), optimiser_name=optimiser.__class__.__name__,
                     lr_schedular_name=lr_schedular.__class__.__name__)

print('=====================================================================================================\n'
      f'Start time: {script_start}\n'
      f'Device: {device}\n'
      f'Training images path: {train_images_path}\n'
      f'Validation images path: {val_images_path}\n'
      f'Save path: {save_path}\n'
      f'Batch size: {batch_size}\n'
      f'Epochs: {total_epochs}\n'
      f'Total training images in dataset (including dataset oversampling): {train_dataset.__len__()}\n'
      f'Total validation images in dataset: {val_dataset.__len__()}\n')


########################################################################################################################
# Training and validation loop.
########################################################################################################################
# def reset_weights(m):
#     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#         torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
#         if m.bias is not None:
#             torch.nn.init.zeros_(m.bias)
#
#
# yolo_model.apply(reset_weights)  # Reset before training


def main():
    print(f'Starting training:')
    early_stopping = EarlyStopping(patience=patience, delta=patience_delta, save_latest=save_latest)
    training_losses = []
    val_losses = []
    training_learning_rates = []
    final_epoch_reached = 0
    for epoch in range(total_epochs):
        ################################################################################################################
        # Training step within epoch.
        ################################################################################################################
        yolo_model.train()
        epoch_train_loss = [0, 0, 0, 0, 0, 0]
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            # Zero the gradients.
            optimiser.zero_grad()
            # Forward pass
            predictions = yolo_model(images)
            # Calculate loss.
            losses_dict = criterion(predictions, targets)
            # Extract each loss.
            xy_loss = losses_dict['xy_loss']
            wh_loss = losses_dict['wh_loss']
            obj_conf_loss = losses_dict['obj_conf_loss']
            noobj_conf_loss = losses_dict['noobj_conf_loss']
            class_loss = losses_dict['class_loss']
            # Calculate total loss, can apply weights here.
            losses = (xy_loss + wh_loss) * loss_weight + obj_conf_loss + noobj_conf_loss * conf_weight + class_loss
            # Check for NaNs or Infs.
            if torch.isnan(losses).any() or torch.isinf(losses).any():
                print("Loss has NaNs or Infs, skipping this batch")
                continue
            # Calculate gradients.
            losses.backward()
            # Apply gradient clipping.
            clip_grad_norm_(yolo_model.parameters(), 2)
            optimiser.step()
            # Epoch loss per batch.
            epoch_train_loss[0] += losses.item()
            epoch_train_loss[1] += xy_loss.item()
            epoch_train_loss[2] += wh_loss.item()
            epoch_train_loss[3] += obj_conf_loss.item()
            epoch_train_loss[4] += noobj_conf_loss.item()
            epoch_train_loss[5] += class_loss.item()
        # Step schedular once per epoch.
        lr_schedular.step()
        # Average epoch loss per image for all images.
        epoch_train_loss = [loss / len(train_loader) for loss in epoch_train_loss]
        training_losses.append(epoch_train_loss)
        training_learning_rates.append(lr_schedular.get_last_lr()[0])

        ################################################################################################################
        # Validation step within epoch.
        ################################################################################################################
        yolo_model.eval()
        epoch_val_loss = [0, 0, 0, 0, 0, 0]
        # No gradient calculations.
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                # Forward pass
                predictions = yolo_model(images)
                # Calculate loss.
                losses_dict = criterion(predictions, targets)
                # Extract each loss.
                xy_loss = losses_dict['xy_loss']
                wh_loss = losses_dict['wh_loss']
                obj_conf_loss = losses_dict['obj_conf_loss']
                noobj_conf_loss = losses_dict['noobj_conf_loss']
                class_loss = losses_dict['class_loss']
                # Calculate total loss, can apply weights here.
                losses = (xy_loss + wh_loss) * loss_weight + obj_conf_loss + noobj_conf_loss * conf_weight + class_loss
                # Epoch loss per batch.
                epoch_val_loss[0] += losses.item()
                epoch_val_loss[1] += xy_loss.item()
                epoch_val_loss[2] += wh_loss.item()
                epoch_val_loss[3] += obj_conf_loss.item()
                epoch_val_loss[4] += noobj_conf_loss.item()
                epoch_val_loss[5] += class_loss.item()

            # Average epoch loss per image for all images.
            epoch_val_loss = [loss / len(val_loader) for loss in epoch_val_loss]
            val_losses.append(epoch_val_loss)

        ################################################################################################################
        # Display training and validation losses (combined loss - weighted).
        ################################################################################################################
        time_now = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
        print(f"\t{time_now}  -  Epoch {epoch + 1}/{total_epochs}, "
              f"Train Loss: {training_losses[-1][0]:0.3f}, "
              f"Val Loss: {val_losses[-1][0]:0.3f}, "
              f"Learning Rate: {lr_schedular.get_last_lr()[0]:0.3f},", end=' ', flush=True)

        ################################################################################################################
        # Check for early stopping. If patience reached, model is saved and final plots are made.
        ################################################################################################################
        final_epoch_reached = epoch
        if final_epoch_reached + 1 > warmup_epochs:
            early_stopping(epoch_val_loss[0], yolo_model, epoch, optimiser, save_path)

        yolo_utils.plot_losses(early_stopping.best_epoch + 1, training_losses, val_losses, training_learning_rates,
                               save_path)
        if early_stopping.early_stop:
            print('Patience reached, stopping early.')
            break
        else:
            print()

    ####################################################################################################################
    # On training complete, pass through validation images and plot them using best model (must be reloaded).
    ####################################################################################################################
    yolo_model.load_state_dict(torch.load(join(save_path, 'model_latest.pth'), weights_only=True)['model_state_dict'])
    yolo_model.eval()
    val_start = datetime.now()
    with torch.no_grad():
        counter = 0
        for index, (images, _) in enumerate(val_loader):
            images = images.to(device)
            detections = yolo_model(images)
            yolo_utils.plot_validation_results(detections, images, S, B, 0.1, counter, train_mean, train_std, save_path)

            counter += batch_size
    inference_time = datetime.now() - val_start

    ####################################################################################################################
    # Save extra parameters to file.
    ####################################################################################################################
    script_end = datetime.now()
    run_time = script_end - script_start
    with open(join(save_path, 'training_parameters.txt'), 'a') as save_file:
        save_file.write(f'\n\nFinal Epoch Reached: {final_epoch_reached + 1}\n'
                        f'Best Epoch: {early_stopping.best_epoch + 1}\n'
                        f"End time: {script_end.strftime('%Y-%m-%d  %H:%M:%S')}\n"
                        f'Total run time: {run_time}\n'
                        f'Validation Inference Time Total: {inference_time}\n'
                        f'Validation Inference Time Per Image: {inference_time / val_dataset.__len__()}')

    print('Training completed.\n'
          f'Best Epoch: {early_stopping.best_epoch + 1}.\n'
          f"End time: {script_end.strftime('%Y-%m-%d  %H:%M:%S')}.\n"
          f'Total run time: {run_time}.')
    print('=====================================================================================================')


if __name__ == '__main__':
    main()
