import os
from datetime import datetime
from os.path import join

import torch
from torch import optim

import DetectionModels
from Datasets.ProstateBladderDataset import ProstateBladderDataset
from DetectionModels.FasterRCNN import FasterRCNN
from DetectionModels.RetinaNet import RetinaNet
from Transformers import Transformers
from Utils import ArgsParser

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
model = args.model  # Type of model to load for training.
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
# Set up model, optimiser, and learning rate scheduler (FasterRCNN/RetinaNet).
########################################################################################################################
print(f'Loading {model} model...', end=' ')
custom_model = RetinaNet() if model == DetectionModels.model_types['RetinaNet'] else FasterRCNN(num_classes=3)
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
                    f'Model: {model}\n'
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
      f'Model: {model}\n'
      f'Training images path: {train_images_path}\n'
      f'Validation image path: {val_images_path}\n'
      f'Save path: {save_path}\n'
      f'Batch size: {batch_size}\n'
      f'Epochs: {total_epochs}\n'
      f'Device: {device}\n'
      f'Total training images in dataset (including dataset oversampling): {train_dataset.__len__()}\n'
      f'Total validation images in dataset: {val_dataset.__len__()}\n')
