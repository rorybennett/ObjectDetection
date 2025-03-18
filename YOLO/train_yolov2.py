########################################################################################################################
# Track run time of script.
########################################################################################################################
import os
from datetime import datetime

import torch

from Datasets.yolov2_dataset import ProstateBladderDataset as PBD
from Transformers import Transformers
from Utils.arg_parsers import YOLOv2ArgParser

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
arg_parser = YOLOv2ArgParser()
args = arg_parser.parse_args()
train_images_path = args.train_images_path  # Path to training images directory.
train_labels_path = args.train_labels_path  # Path to training labels directory.
val_images_path = args.val_images_path  # Path to validation images directory.
val_labels_path = args.val_labels_path  # Path to validation labels directory.
save_dir = args.save_path  # Path to saving directory, where models, loss plots, and validation results are stored.
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)  # Make save_path into dir.
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
loss_weight = args.loss_weight  # Weight for localization loss.
conf_weight = args.conf_weight  # Weight for confidence loss on no-object cells.

########################################################################################################################
# Transformers and datasets used by models.
########################################################################################################################
train_transforms = Transformers.get_yolov1_transforms()

train_mean, train_std, train_anchors = PBD(images_root=train_images_path,
                                           labels_root=train_labels_path, verbose=True).get_mean_std_and_anchors()
train_dataset = PBD(images_root=train_images_path, labels_root=train_labels_path, transforms=train_transforms,
                    oversampling_factor=oversampling_factor, train_mean=train_mean, train_std=train_std,
                    num_classes=num_classes, anchors=train_anchors, verbose=True)
val_dataset = PBD(images_root=val_images_path, labels_root=val_labels_path, train_mean=train_mean, train_std=train_std,
                  num_classes=num_classes, anchors=train_anchors)
# If you want to validate the dataset transforms visually, you can do it here.
for i in range(len(train_dataset)):
    train_dataset.display_transforms(i)

exit()
########################################################################################################################
# Set up dataloaders.
########################################################################################################################
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
