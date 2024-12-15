"""
The Argument Parser used to collect terminal command inputs when running the main scripts.
"""

import argparse


def get_arg_parser():
    """
    Set up and return the parser.

    :return: parser.parse_args()
    """
    # Path parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-tip',
                        '--train_images_path',
                        type=str,
                        required=True,
                        help='Path to training images directory')
    parser.add_argument('-tlp',
                        '--train_labels_path',
                        type=str,
                        required=True,
                        help='Path to training labels directory')
    parser.add_argument('-vip',
                        '--val_images_path',
                        type=str,
                        required=True,
                        help='Path to validation images directory')
    parser.add_argument('-vlp',
                        '--val_labels_path',
                        type=str,
                        required=True,
                        help='Path to validation images directory')
    parser.add_argument('-sp',
                        '--save_path',
                        type=str,
                        required=True,
                        help='Path to save directory where model_best.pth and other files are stored')
    # Training parameters.
    parser.add_argument('-nc',
                        '--number_of_classes',
                        type=int,
                        required=True,
                        help='Number of classes being considered')
    parser.add_argument('-e',
                        '--epochs',
                        type=int,
                        default=1000,
                        help='Maximum number of training epochs')
    parser.add_argument('-we',
                        '--warmup_epochs',
                        type=int,
                        default=15,
                        help='Number of epochs to train before early stopping checks are run')
    parser.add_argument('-is',
                        '--image_size',
                        type=int,
                        default=(600, 600),
                        help='Scaled image size, applied to all images using the resize transform')
    parser.add_argument('-bs',
                        '--batch_size',
                        type=int,
                        default=8,
                        help='Training and validation batch size')
    parser.add_argument('-p',
                        '--patience',
                        type=int,
                        default=100,
                        help='Number of epochs without validation improvement before early stopping is applied')
    parser.add_argument('-pd',
                        '--patience_delta',
                        type=int,
                        default=0.005,
                        help='Minimum improvement amount to prevent early stopping')
    parser.add_argument('-lr',
                        '--learning_rate',
                        type=float,
                        default=0.01,
                        help='Optimiser starting learning rate')
    parser.add_argument('-lres',
                        '--learning_restart',
                        type=int,
                        default=100,
                        help='Learning rate schedular restart frequency')
    parser.add_argument('-m',
                        '--momentum',
                        type=float,
                        default=0.9,
                        help='Optimiser momentum')
    parser.add_argument('-wd',
                        '--weight_decay',
                        type=float,
                        default=0.005,
                        help='Optimiser weight decay')
    parser.add_argument('-bw',
                        '--box_weight',
                        type=float,
                        default=7.5,
                        help='Weight applied to box loss')
    parser.add_argument('-cw',
                        '--cls_weight',
                        type=float,
                        default=0.5,
                        help='Weight applied to classification loss')
    parser.add_argument('-of',
                        '--oversampling_factor',
                        type=int,
                        default=1,
                        help='How much oversampling is desired (multiply the number of training images by this factor)')
    parser.add_argument('-sl',
                        '--save_latest',
                        type=bool,
                        default=True,
                        help='Save the latest model as well as the best model.pth')

    return parser.parse_args()
