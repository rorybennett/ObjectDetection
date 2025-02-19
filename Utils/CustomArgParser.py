"""
The Argument Parser used to collect terminal command inputs when running the main scripts.
"""

import argparse
from os.path import join


class CustomArgParser:
    def __init__(self):
        """
        Set up and return the parser.

        :return: parser.parse_args()
        """
        self.parser = argparse.ArgumentParser()
        ################################################################################################################
        # Path parameters
        ################################################################################################################
        self.parser.add_argument('-tip',
                                 '--train_images_path',
                                 type=str,
                                 required=True,
                                 help='Path to training images directory')
        self.parser.add_argument('-tlp',
                                 '--train_labels_path',
                                 type=str,
                                 required=True,
                                 help='Path to training labels directory')
        self.parser.add_argument('-vip',
                                 '--val_images_path',
                                 type=str,
                                 required=True,
                                 help='Path to validation images directory')
        self.parser.add_argument('-vlp',
                                 '--val_labels_path',
                                 type=str,
                                 required=True,
                                 help='Path to validation images directory')
        self.parser.add_argument('-sp',
                                 '--save_path',
                                 type=str,
                                 required=True,
                                 help='Path to save directory where model_best.pth and other files are stored')

        ################################################################################################################
        # Model parameters
        ################################################################################################################
        self.parser.add_argument('-nc',
                                 '--number_of_classes',
                                 type=int,
                                 required=True,
                                 help='Number of classes being considered (FasterRCNN has +1 due to background class)')
        self.parser.add_argument('-bbt',
                                 '--backbone_type',
                                 type=str,
                                 required=True,
                                 help='Model backbone, renet50_fpn_v2 seems to be the best')

        ################################################################################################################
        # Training parameters
        ################################################################################################################
        self.parser.add_argument('-e',
                                 '--epochs',
                                 type=int,
                                 default=1000,
                                 help='Maximum number of training epochs')
        self.parser.add_argument('-we',
                                 '--warmup_epochs',
                                 type=int,
                                 default=1,
                                 help='Number of epochs to train before early stopping checks are run')
        self.parser.add_argument('-bs',
                                 '--batch_size',
                                 type=int,
                                 default=8,
                                 help='Training and validation batch size')
        self.parser.add_argument('-p',
                                 '--patience',
                                 type=int,
                                 default=0,
                                 help='Number of epochs without validation improvement before early stopping is applied')
        self.parser.add_argument('-pd',
                                 '--patience_delta',
                                 type=int,
                                 default=0.001,
                                 help='Minimum improvement amount to prevent early stopping')
        self.parser.add_argument('-lr',
                                 '--learning_rate',
                                 type=float,
                                 default=0.01,
                                 help='Optimiser starting learning rate')
        self.parser.add_argument('-lres',
                                 '--learning_restart',
                                 type=int,
                                 default=100,
                                 help='Learning rate schedular restart frequency')
        self.parser.add_argument('-m',
                                 '--momentum',
                                 type=float,
                                 default=0.9,
                                 help='Optimiser momentum')
        self.parser.add_argument('-wd',
                                 '--weight_decay',
                                 type=float,
                                 default=0.005,
                                 help='Optimiser weight decay')
        self.parser.add_argument('-bw',
                                 '--box_weight',
                                 type=float,
                                 default=1,
                                 help='Weight applied to box loss')
        self.parser.add_argument('-cw',
                                 '--cls_weight',
                                 type=float,
                                 default=1,
                                 help='Weight applied to classification loss')
        # These two are for FasterRCNN.
        self.parser.add_argument('-ow',
                                 '--objectness_weight',
                                 type=float,
                                 default=1,
                                 help='Weight applied to objectness loss')
        self.parser.add_argument('-rpw',
                                 '--rpn_box_weight',
                                 type=float,
                                 default=1,
                                 help='Weight applied to rpn box loss')
        self.parser.add_argument('-sl',
                                 '--save_latest',
                                 type=bool,
                                 default=True,
                                 help='Save the latest model as well as the best model.pth')

        ################################################################################################################
        # Dataset parameters
        ################################################################################################################
        self.parser.add_argument('-is',
                                 '--image_size',
                                 type=int,
                                 default=600,
                                 help='Scaled image size, applied to all images, aspect ratio maintained')
        self.parser.add_argument('-of',
                                 '--oversampling_factor',
                                 type=int,
                                 default=1,
                                 help='How much oversampling is desired (multiply the number of training '
                                      'images by this factor)')

    def save_args(self, save_path, **extra_kwargs):
        """
        Save all parser args to text file at given path. Parameters not present in parser must be given as extra_kwargs.

        :param save_path:
        :return:
        """
        args = self.parser.parse_args()
        with open(join(save_path, 'training_parameters.txt'), 'w') as save_file:
            for key, value in extra_kwargs.items():
                save_file.write(f'{key}: {value}\n')

            for arg in vars(args):
                value = getattr(args, arg)
                save_file.write(f'{arg}: {value}\n')
