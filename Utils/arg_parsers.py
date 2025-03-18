"""
The Argument Parsers used to collect terminal command inputs when running the main scripts. Each model has its own
parser class.
"""

import argparse
import os
from os.path import join


class BaseArgParser:
    def __init__(self):
        """
        Set up and return the base arg parser. Variables here are shared between all child arg parsers.

        :return: parser.parse_args()
        """
        self.parser = argparse.ArgumentParser()
        ################################################################################################################
        # Training parameters
        ################################################################################################################
        self.parser.add_argument('-e', '--epochs', type=int, default=1000,
                                 help='Maximum number of training epochs')
        self.parser.add_argument('-we', '--warmup_epochs', type=int, default=1,
                                 help='Number of epochs to train before early stopping checks are run')
        self.parser.add_argument('-bs', '--batch_size', type=int, default=8,
                                 help='Training and validation batch size')
        self.parser.add_argument('-p', '--patience', type=int, default=0,
                                 help='Number of epochs without validation improvement before early stopping is applied')
        self.parser.add_argument('-pd', '--patience_delta', type=int, default=0.001,
                                 help='Minimum improvement amount to prevent early stopping')
        self.parser.add_argument('-olr', '--optimiser_learning_rate', type=float, default=0.01,
                                 help='Optimiser initial learning rate')
        self.parser.add_argument('-olrr', '--optimiser_learning_rate_restart', type=int, default=100,
                                 help='Learning rate schedular restart frequency')
        self.parser.add_argument('-om', '--optimiser_momentum', type=float, default=0.9,
                                 help='Optimiser momentum')
        self.parser.add_argument('-owd', '--optimiser_weight_decay', type=float, default=0.005,
                                 help='Optimiser weight decay')
        self.parser.add_argument('-sl', '--save_latest', type=bool, default=True,
                                 help='Save the latest trained model')
        ################################################################################################################
        # Dataset parameters
        ################################################################################################################
        self.parser.add_argument('-of', '--oversampling_factor', type=int, default=1,
                                 help='How much oversampling is desired (multiply the number of training images by this'
                                      ' factor, transforms are only applied to oversampled images)')

    def parse_args(self):
        return self.parser.parse_args()

    @staticmethod
    def save_args(save_dir, args, **extra_kwargs):
        """
        Save all parser args to text file at given path. Parameters not present in parser must be given as extra_kwargs.

        :param save_dir: Directory to save .txt file.
        :param args:
        """
        os.makedirs(save_dir, exist_ok=True)
        with open(join(save_dir, 'training_parameters.txt'), 'w') as save_file:
            for key, value in extra_kwargs.items():
                save_file.write(f'{key}: {value}\n')

            for arg in vars(args):
                value = getattr(args, arg)
                save_file.write(f'{arg}: {value}\n')


class FasterRCNNArgParser(BaseArgParser):
    def __init__(self):
        """
        Set up and return the parser with additional arguments for FasterRCNN.
        """
        super().__init__()
        ################################################################################################################
        # Path parameters
        ################################################################################################################
        self.parser.add_argument('-tip', '--train_images_path', type=str, required=True,
                                 help='Path to training images directory')
        self.parser.add_argument('-tlp', '--train_labels_path', type=str, required=True,
                                 help='Path to training labels directory')
        self.parser.add_argument('-vip', '--val_images_path', type=str, required=True,
                                 help='Path to validation images directory')
        self.parser.add_argument('-vlp', '--val_labels_path', type=str, required=True,
                                 help='Path to validation images directory')
        self.parser.add_argument('-sp', '--save_path', type=str, required=True,
                                 help='Path to save directory where model_best.pth and other files are stored')

        self.parser.add_argument('-is', '--image_size', type=int, default=600,
                                 help='Scaled image size, applied to all images, aspect ratio maintained')
        self.parser.add_argument('-bbt', '--backbone_type', type=str, default='fasterrcnn_resnet50_fpn_v2',
                                 choices=["fasterrcnn_resnet50_fpn", "fasterrcnn_resnet50_fpn_v2",
                                          "fasterrcnn_mobilenet_v3_large_fpn",
                                          "fasterrcnn_mobilenet_v3_large_320_fpn"],
                                 help='Model backbone (resnet50_fpn_v2 seems to be the best)')
        ################################################################################################################
        # Model parameters
        ################################################################################################################
        self.parser.add_argument('-nc', '--number_of_classes', type=int, required=True,
                                 help='Number of classes being considered (FasterRCNN has +1 due to background class)')
        self.parser.add_argument('-bw', '--box_weight', type=float, default=1,
                                 help='Weight applied to box loss')
        self.parser.add_argument('-cw', '--class_weight', type=float, default=1,
                                 help='Weight applied to classification loss')
        self.parser.add_argument('-ow', '--objectness_weight', type=float, default=1,
                                 help='Weight applied to objectness loss')
        self.parser.add_argument('-rpw', '--rpn_box_weight', type=float, default=1,
                                 help='Weight applied to rpn box loss')


class RetinaNetArgParser(BaseArgParser):
    def __init__(self):
        """
        Set up and return the parser with additional arguments for RetinaNet.
        """
        super().__init__()
        ################################################################################################################
        # Path parameters
        ################################################################################################################
        self.parser.add_argument('-tip', '--train_images_path', type=str, required=True,
                                 help='Path to training images directory')
        self.parser.add_argument('-tlp', '--train_labels_path', type=str, required=True,
                                 help='Path to training labels directory')
        self.parser.add_argument('-vip', '--val_images_path', type=str, required=True,
                                 help='Path to validation images directory')
        self.parser.add_argument('-vlp', '--val_labels_path', type=str, required=True,
                                 help='Path to validation images directory')
        self.parser.add_argument('-sp', '--save_path', type=str, required=True,
                                 help='Path to save directory where model_best.pth and other files are stored')
        ################################################################################################################
        # Model parameters
        ################################################################################################################
        self.parser.add_argument('-nc', '--number_of_classes', type=int, required=True,
                                 help='Number of classes being considered (FasterRCNN has +1 due to background class)')

        self.parser.add_argument('-is', '--image_size', type=int, default=600,
                                 help='Scaled image size, applied to all images, aspect ratio maintained')
        self.parser.add_argument('-bbt', '--backbone_type', type=str, default='retinanet_resnet50_fpn_v2',
                                 choices=['retinanet_resnet50_fpn_v2', 'mobilenet_v2'],
                                 help='Model backbone (resnet50_fpn_v2 seems to be the best)')
        self.parser.add_argument('-bw', '--box_weight', type=float, default=1,
                                 help='Weight applied to box loss')
        self.parser.add_argument('-cw', '--class_weight', type=float, default=1,
                                 help='Weight applied to classification loss')


class YOLOv1ArgParser(BaseArgParser):
    def __init__(self):
        """
        Set up and return the parser with additional arguments for YOLOv1 and YOLOv1_fast.
        """
        super().__init__()
        self.parser.add_argument('-nc', '--number_of_classes', type=int, required=True,
                                 help='Number of classes being considered')
        ################################################################################################################
        # Path parameters
        ################################################################################################################
        self.parser.add_argument('-tip', '--train_images_path', type=str, required=True,
                                 help='Path to training images directory')
        self.parser.add_argument('-tlp', '--train_labels_path', type=str, required=True,
                                 help='Path to training labels directory')
        self.parser.add_argument('-vip', '--val_images_path', type=str, required=True,
                                 help='Path to validation images directory')
        self.parser.add_argument('-vlp', '--val_labels_path', type=str, required=True,
                                 help='Path to validation images directory')
        self.parser.add_argument('-sp', '--save_path', type=str, required=True,
                                 help='Path to save directory where model_best.pth and other files are stored')
        ################################################################################################################
        # Model parameters
        ################################################################################################################
        self.parser.add_argument('-ys', '--yolo_s', type=int, default=7,
                                 help='YOLOv1 grid size (SxS)')
        self.parser.add_argument('-yb', '--yolo_b', type=int, default=2,
                                 help='YOLOv1 bounding boxes per grid cell')
        self.parser.add_argument('-lw', '--loss_weight', type=float, default=5,
                                 help='YOLOv1 loss weight')
        self.parser.add_argument('-cw', '--conf_weight', type=float, default=0.5,
                                 help='YOLOv1 confidence weight')
        self.parser.add_argument('-mt', '--model_type', choices=['normal', 'fast'], default='normal',
                                 help='YOLOv1 model type, either normal or fast')


class YOLOv2ArgParser(BaseArgParser):
    def __init__(self):
        """
        Set up and return the parser with additional arguments for YOLOv1 and YOLOv1_fast.
        """
        super().__init__()
        self.parser.add_argument('-nc', '--number_of_classes', type=int, required=True,
                                 help='Number of classes being considered')
        ################################################################################################################
        # Path parameters
        ################################################################################################################
        self.parser.add_argument('-tip', '--train_images_path', type=str, required=True,
                                 help='Path to training images directory')
        self.parser.add_argument('-tlp', '--train_labels_path', type=str, required=True,
                                 help='Path to training labels directory')
        self.parser.add_argument('-vip', '--val_images_path', type=str, required=True,
                                 help='Path to validation images directory')
        self.parser.add_argument('-vlp', '--val_labels_path', type=str, required=True,
                                 help='Path to validation images directory')
        self.parser.add_argument('-sp', '--save_path', type=str, required=True,
                                 help='Path to save directory where model_best.pth and other files are stored')
        ################################################################################################################
        # Model parameters
        ################################################################################################################
        self.parser.add_argument('-yb', '--yolo_b', type=int, default=2,
                                 help='YOLOv1 bounding boxes per grid cell')
        self.parser.add_argument('-lw', '--loss_weight', type=float, default=5,
                                 help='YOLOv1 loss weight')
        self.parser.add_argument('-cw', '--conf_weight', type=float, default=0.5,
                                 help='YOLOv1 confidence weight')
        self.parser.add_argument('-mt', '--model_type', choices=['normal', 'fast'], default='normal',
                                 help='YOLOv1 model type, either normal or fast')


class YOLOv8ArgParser(BaseArgParser):
    def __init__(self):
        """
        Set up and return the parser with additional arguments for YOLOv8.
        """
        super().__init__()
        ################################################################################################################
        # Path parameters
        ################################################################################################################
        self.parser.add_argument('-pp', '--project_path', type=str, required=True,
                                 help='Project path, where training output are saved')
        self.parser.add_argument('-n', '--name', type=str, required=True,
                                 help='Name of training run')
        self.parser.add_argument('-ms', '--model_size', type=str, default='x',
                                 choices=['n', 's', 'm', 'l', 'x'],
                                 help='YOLOv8 model size (n, s, m, l, x)')
        self.parser.add_argument('-dyp', '--dataset_yaml_path', type=str, required=True,
                                 help='Path to dataset.yaml file')
        self.parser.add_argument('-flr', '--flip_lr', type=float, default=0.2,
                                 help='Probability of flipping image lr during training')
