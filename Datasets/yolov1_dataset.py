import os

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from natsort import natsorted
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from Utils import yolo_utils


class ProstateBladderDataset(Dataset):
    def __init__(self, images_root, labels_root, label_names=None, transforms=None, train_mean=None,
                 train_std=None, oversampling_factor=1, S=7, B=2, C=1, verbose=False):
        """
        Initialise the ProstateBladderDataset class. Set the images' and labels' root directories, the transforms
        that are to be used when __getitem__() is called, and perform minor dataset validation.

        :param images_root: Directory containing images.
        :param labels_root: Directory containing labels.
        :param label_names: List of names for associated labels.
        :param transforms: Transformers to be used on this dataset.
        :param oversampling_factor: Increase dataset size by this amount (oversampling).
        :param train_mean: Training dataset mean, in 3 channel format.
        :param train_std: Training dataset standard deviation, in 3 channel format.
        :param S: Grid size (SxS).
        :param B: Box detection per grid cell.
        :param C: Number of classes.
        :param verbose: Print details to screen.
        """
        # Default values for when the mean and std are calculated without using the dataset.
        train_mean = train_mean or 0
        train_std = train_std or 1

        self.images_root = images_root
        self.labels_root = labels_root
        self.label_names = label_names
        self.transforms = transforms
        self.oversampling_factor = oversampling_factor
        self.S = S
        self.B = B
        self.C = C
        self.verbose = verbose
        self.images = list(natsorted(os.listdir(images_root)))
        self.labels = list(natsorted(os.listdir(labels_root)))
        self.required_transforms = v2.Compose([
            v2.Resize((448, 448)),
            v2.Normalize(mean=[train_mean], std=[train_std]),
            v2.Grayscale(num_output_channels=3)
        ])

        if self.verbose:
            print(f'Initialising the ProstateBladder dataset'
                  f'\tImages root: {self.images_root}.\n'
                  f'\tLabels root: {self.labels_root}.\n'
                  f'\tTotal Images: {len(self.images)}.\n'
                  f'\tTotal Labels: {len(self.labels)}.\n'
                  f'\tOversampling count: {self.oversampling_factor}.')

        self._validate_dataset()

    def __len__(self):
        # Total images in dataset = input images * oversampling factor.
        return len(self.images) * self.oversampling_factor

    def __getitem__(self, idx):
        original_idx = idx // self.oversampling_factor
        img_path = os.path.join(self.images_root, self.images[original_idx])
        label_path = os.path.join(self.labels_root, self.labels[original_idx])
        # Load image as 3-channel greyscale.
        img = yolo_utils.read_image(img_path)

        # Get targets (bounding boxes, labels, etc.)
        target = yolo_utils.get_label_data(label_path, img.shape, idx)

        # Convert img to the format expected by the model.
        img = torch.tensor(img, dtype=torch.float32)
        img = tv_tensors.Image(img).permute(2, 0, 1)

        # Only apply optional transforms on oversampled images.
        if idx % self.oversampling_factor != 0:
            img, target = self.transforms(img, target)

        # Apply required transforms (resize, normalise, 3-channel greyscale).
        img, target = self.required_transforms(img, target)

        # Convert target (labels) to YOLOv1 format
        lbl = yolo_utils.convert_xyxy_to_yolov1(target, S=self.S, B=self.B, C=self.C)

        return img, lbl

    def get_mean_and_std(self):
        """
        Return the mean and std of the current dataset, scaled between (0, 1). The assumption is that the input images
        are scaled to (0, 255), so if the mean and std seem wildly off just make sure the input pixels are scaled
        correctly. These values are required for the normalising transform, and the training values should be used on
        the validation dataset. It is also assumed that the images are greyscale.

        :return: dataset_mean, dataset_std.
        """
        means = []
        stds = []
        for image in self.images:
            # Read image using cv2.
            img = cv2.imread(os.path.join(self.images_root, image), cv2.IMREAD_GRAYSCALE) / 255
            # Calculate mean and std using cv2.
            m, s = cv2.meanStdDev(img)
            means.append(m)
            stds.append(s)

        dataset_mean = np.mean(means)
        dataset_std = np.mean(stds)

        if self.verbose:
            print(f'Calculated Dataset Mean: {dataset_mean}.\n'
                  f'Calculated Dataset Standard Deviation: {dataset_std}.')

        return dataset_mean, dataset_std

    def _validate_dataset(self):
        """
        Minor dataset validation:
            1. Ensure there are the same amount of images as labels.
            2. Ensure the labels and images have the same naming convention.
        """
        if self.verbose:
            print(f'Running dataset validation...')
        # Validation 1.
        assert len(self.images) == len(self.labels), "Dataset input images and labels are of different length."
        # Validation 2.
        for i in range(len(self.images)):
            if self.images[i].split('.')[0] != self.labels[i].split('.')[0]:
                assert False, f"There is a mismatch between imgs and labels at {self.images[i]} and {self.labels[i]}."

        if self.verbose:
            print(f'Dataset validation passed.')

        return

    def display_transforms(self, idx):
        """
        Visualizes the original and transformed image with bounding boxes.

        :param idx: Index of the sample to visualize.
        """
        original_idx = idx // self.oversampling_factor
        img_path = os.path.join(self.images_root, self.images[original_idx])
        label_path = os.path.join(self.labels_root, self.labels[original_idx])

        # Load original image and label in XYXY format
        img_original = yolo_utils.read_image(img_path)
        target_original = yolo_utils.get_label_data(label_path, img_size=img_original.shape, idx=idx)

        img_transformed, target_transformed = self.__getitem__(idx)

        # Convert images to displayable format
        img_transformed = img_transformed.permute(1, 2, 0).numpy()  # (C, H, W) → (H, W, C)
        img_transformed_display = (img_transformed - img_transformed.min()) / (
                img_transformed.max() - img_transformed.min()
        )

        # Create plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(img_original, cmap="gray", vmin=-2, vmax=2)
        yolo_utils.draw_yolo_boxes_on_axis(self.B, self.S, axes[0], target_original["boxes"],
                                           "Original Image with Bounding Boxes", box_format='xyxy')

        axes[1].imshow(img_transformed_display, cmap="gray", vmin=-2, vmax=2)
        yolo_utils.draw_yolo_boxes_on_axis(self.B, self.S, axes[1], target_transformed,
                                           "Transformed Image with Bounding Boxes", box_format='yolov1')

        plt.show()

    @staticmethod
    def convert_xyxy_to_yolov1(target, S=7, B=2, C=1):
        """
        Converts target labels from XYXY format to YOLOv1 format, assuming an image size of (448, 448). Returns the target
        as a torch.tensor.

        :param target: Dictionary containing target data in XYXY format.
        :param S: Grid size (SxS).
        :param B: Number of bounding boxes per grid cell.
        :param C: Number of classes.
        :return: YOLOv1 label tensor of shape (S, S, B*5 + C).
        """
        img_width, img_height = 448, 448
        yolo_v1_label = np.zeros((S, S, B * 5 + C), dtype=np.float32)  # (7,7,11) for 1 class

        boxes = target["boxes"]
        labels = target["labels"]

        for i in range(len(boxes)):
            x_min, y_min, x_max, y_max = boxes[i]
            class_id = labels[i].item()

            # Convert absolute XYXY to YOLO (x_center, y_center, width, height)
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min

            # Normalize by image size
            x_center /= img_width
            y_center /= img_height
            width /= img_width
            height /= img_height

            # Determine which grid cell (row, col) the box belongs to
            grid_x = int(x_center * S)
            grid_y = int(y_center * S)

            # Compute relative coordinates within the grid cell
            x_rel = (x_center * S) - grid_x
            y_rel = (y_center * S) - grid_y
            w_rel = width
            h_rel = height

            # Assign values to YOLOv1 label tensor
            for b in range(B):
                conf_index = b * 5 + 4  # Confidence score index in the tensor
                if yolo_v1_label[grid_y, grid_x, conf_index] == 0:  # Check if this slot is empty
                    yolo_v1_label[grid_y, grid_x, b * 5: (b + 1) * 5] = [x_rel, y_rel, w_rel, h_rel, 1]
                    break  # Assign only one box per object

            # Assign class probabilities (One-Hot Encoding)
            class_start_idx = B * 5  # After all bounding boxes
            yolo_v1_label[grid_y, grid_x, class_start_idx:] = 0  # Reset class probabilities
            yolo_v1_label[grid_y, grid_x, class_start_idx + class_id] = 1  # Set class

        return torch.tensor(yolo_v1_label, dtype=torch.float32)
