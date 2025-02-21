import os
from datetime import datetime
from os.path import join

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_

import Datasets
from Datasets.ProstateBladderDataset import ProstateBladderDataset as PBD
from EarlyStopping.EarlyStopping import EarlyStopping
from FasterRCNN.CustomFasterRCNN import CustomFasterRCNN
from Transformers import Transformers
from Utils import CustomArgParser, Utils
from . import backbones

########################################################################################################################
# Track run time of script.
########################################################################################################################
script_start = datetime.now()

########################################################################################################################
# Set seeds for semi-reproducibility.
########################################################################################################################
seed = 2024
torch.manual_seed(seed)

########################################################################################################################
# Set device and empty cuda cache.
########################################################################################################################
torch.cuda.empty_cache()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

########################################################################################################################
# Fetch parser and set variables from parser args.
########################################################################################################################
CAP = CustomArgParser.CustomArgParser()
args = CAP.parser.parse_args()
train_images_path = args.train_images_path  # Path to training images directory.
train_labels_path = args.train_labels_path  # Path to training labels directory.
val_images_path = args.val_images_path  # Path to validation images directory.
val_labels_path = args.val_labels_path  # Path to validation labels directory.
save_path = args.save_path  # Path to saving directory, where models, loss plots, and validation results are stored.
if not os.path.isdir(save_path):
    os.makedirs(save_path)  # Make save_path into dir.
image_size = args.image_size  # Image size for resizing, used in training and validation.
batch_size = args.batch_size  # Batch size for loader.
num_classes = args.number_of_classes  # Number of classes present in training dataset.
total_epochs = args.epochs  # Training epochs.
warmup_epochs = args.warmup_epochs  # Epochs before early stop checks are done.
patience = args.patience  # Early stopping patience.
patience_delta = args.patience_delta  # Early stopping delta.
learning_rate = args.learning_rate  # Optimiser learning rate.
learning_restart = args.learning_restart  # Learning rate schedular restart frequency.
momentum = args.momentum  # Optimiser momentum.
weight_decay = args.weight_decay  # Optimiser weight decay.
box_weight = args.box_weight  # Weight applied to box loss.
cls_weight = args.cls_weight  # Weight applied to classification loss.
objectness_weight = args.objectness_weight  # Weight applied to objectness loss.
rpn_weight = args.rpn_box_weight  # Weight applied to rpn box loss.
oversampling_factor = args.oversampling_factor  # Oversampling factor.
backbone_type = args.backbone_type  # Type of backbone model should use.
save_latest = args.save_latest  # Save latest model as well as the best model.pth.

########################################################################################################################
# Transformers and datasets used by models.
########################################################################################################################
train_transforms = Transformers.get_training_transforms()

train_mean, train_std = PBD(images_root=train_images_path, labels_root=train_labels_path,
                            model_type=Datasets.model_fasterrcnn).get_mean_and_std()
train_dataset = PBD(images_root=train_images_path, labels_root=train_labels_path, model_type=Datasets.model_fasterrcnn,
                    optional_transforms=train_transforms, oversampling_factor=oversampling_factor)
val_dataset = PBD(images_root=val_images_path, labels_root=val_labels_path, model_type=Datasets.model_fasterrcnn)
# If you want to validate the dataset transforms visually, you can do it here.
# for i in range(len(train_dataset)):
#     train_dataset.display_transforms(i)
#
# exit()
########################################################################################################################
# Set up dataloaders.
########################################################################################################################
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                           collate_fn=lambda x: tuple(zip(*x)))
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                         collate_fn=lambda x: tuple(zip(*x)))

########################################################################################################################
# Set up model, optimiser, and learning rate scheduler (FasterRCNN).
########################################################################################################################
train_mean = [train_mean] * 3
train_std = [train_std] * 3
print(f'Loading FasterRCNN model {backbone_type}...', end=' ')
custom_fasterrcnn = CustomFasterRCNN(num_classes=num_classes, backbone_type=backbones[backbone_type],
                                     min_size=image_size, max_size=image_size, image_mean=train_mean,
                                     image_std=train_std)
custom_fasterrcnn.model.to(device)
params = [p for p in custom_fasterrcnn.model.parameters() if p.requires_grad]
optimiser = torch.optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
lr_schedular = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0=learning_restart, eta_min=0)
print(f'Model loaded.')

########################################################################################################################
# Save training parameters to file and display to screen.
########################################################################################################################
CAP.save_args(save_path, script_start=script_start, seed=seed, device=device, device_name=Utils.get_device_name(),
              train_mean=train_mean, train_std=train_std, train_dataset_len=train_dataset.__len__(),
              val_dataset_len=val_dataset.__len__(), transformer_count=len(train_transforms.transforms),
              optimiser_name=optimiser.__class__.__name__, lr_schedular_name=lr_schedular.__class__.__name__)

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
        custom_fasterrcnn.model.train()
        epoch_train_loss = [0, 0, 0, 0, 0]
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # Zero the gradients.
            optimiser.zero_grad()
            # Forward pass.
            loss_dict = custom_fasterrcnn.forward(images, targets)
            # Extract each loss.
            cls_loss = loss_dict['loss_classifier']
            bbox_loss = loss_dict['loss_box_reg']
            objectness_loss = loss_dict['loss_objectness']
            rpn_box_loss = loss_dict['loss_rpn_box_reg']
            # Calculate total loss, can apply weights here.
            losses = cls_loss * cls_weight + bbox_loss * box_weight + objectness_loss * objectness_weight + rpn_box_loss * rpn_weight
            # Check for NaNs or Infs.
            if torch.isnan(losses).any() or torch.isinf(losses).any():
                print("Loss has NaNs or Infs, skipping this batch")
                continue
            # Calculate gradients.
            losses.backward()
            # Apply gradient clipping.
            clip_grad_norm_(custom_fasterrcnn.model.parameters(), 2)
            optimiser.step()
            # Epoch loss per batch.
            epoch_train_loss[0] += losses.item()
            epoch_train_loss[1] += cls_loss.item()
            epoch_train_loss[2] += bbox_loss.item()
            epoch_train_loss[3] += objectness_loss.item()
            epoch_train_loss[4] += rpn_box_loss.item()
        # Step schedular once per epoch.
        lr_schedular.step()
        # Average epoch loss per image for all images.
        epoch_train_loss = [loss / len(train_loader) for loss in epoch_train_loss]
        training_losses.append(epoch_train_loss)
        training_learning_rates.append(lr_schedular.get_last_lr()[0])

        ################################################################################################################
        # Validation step within epoch.
        ################################################################################################################
        custom_fasterrcnn.model.eval()
        epoch_val_loss = [0, 0, 0, 0, 0]
        # No gradient calculations.
        with torch.no_grad():
            for images, targets in val_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                # Forward pass.
                loss_dict, _ = custom_fasterrcnn.forward(images, targets)
                # Extract each loss.
                cls_loss = loss_dict['loss_classifier']
                bbox_loss = loss_dict['loss_box_reg']
                objectness_loss = loss_dict['loss_objectness']
                rpn_box_loss = loss_dict['loss_rpn_box_reg']
                # Calculate total loss.
                losses = cls_loss * cls_weight + bbox_loss * box_weight + objectness_loss * objectness_weight + rpn_box_loss * rpn_weight
                # Epoch loss per batch.
                epoch_val_loss[0] += losses.item()
                epoch_val_loss[1] += cls_loss.item()
                epoch_val_loss[2] += bbox_loss.item()
                epoch_val_loss[3] += objectness_loss.item()
                epoch_val_loss[4] += rpn_box_loss.item()

            # Average epoch loss per image for all images.
            epoch_val_loss = [loss / len(val_loader) for loss in epoch_val_loss]
            val_losses.append(epoch_val_loss)

        ################################################################################################################
        # Display training and validation losses (combined loss - training is weighted, validation is not).
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
            early_stopping(epoch_val_loss[0], custom_fasterrcnn.model, epoch, optimiser, save_path)

        Utils.plot_losses_fasterrcnn(early_stopping.best_epoch + 1, training_losses, val_losses,
                                     training_learning_rates, save_path)
        if early_stopping.early_stop:
            print('Patience reached, stopping early.')
            break
        else:
            print()

    ####################################################################################################################
    # On training complete, pass through validation images and plot them using best model (must be reloaded).
    ####################################################################################################################
    custom_fasterrcnn.model.load_state_dict(
        torch.load(join(save_path, 'model_best.pth'), weights_only=True)['model_state_dict'])
    custom_fasterrcnn.model.eval()
    val_start = datetime.now()
    with torch.no_grad():
        counter = 0
        for images, targets in val_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            _, detections = custom_fasterrcnn.forward(images, targets)

            Utils.plot_validation_results(detections, images, 1, 1, counter, save_path)

            counter += batch_size
    inference_time = datetime.now() - val_start
    ####################################################################################################################
    # Save extra parameters to file.
    ####################################################################################################################
    script_end = datetime.now()
    run_time = script_end - script_start
    with open(join(save_path, 'training_parameters.txt'), 'a') as save_file:
        save_file.write(f'Final Epoch Reached: {final_epoch_reached + 1}\n'
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
