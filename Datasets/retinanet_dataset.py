"""
Custom dataset class for the prostate and bladder detection. This is specifically for the RetinaNet model.
A single prostate bounding box and/or a single bladder bounding box are present for each image.
The dataset labels should be organised in the YOLO format. Images will be stored in the root/images directory
and labels in the root/labels directory.

todo Move display of result to this class perhaps, allowing for correct labelling?
"""
import os

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt, patches
from natsort import natsorted
from torchvision import tv_tensors
from torchvision.transforms import v2


class ProstateBladderDataset(torch.utils.data.Dataset):
    def __init__(self, images_root, labels_root, label_names=None, optional_transforms=None, oversampling_factor=1,
                 verbose=False):
        """
        Initialise the ProstateBladderDataset class. Set the images' and labels' root directories, the transforms
        that are to be used when __getitem__() is called, and perform minor dataset validation.

        :param images_root: Directory containing images.
        :param labels_root: Directory containing labels.
        :param label_names: List of names for associated labels.
        :param optional_transforms: Transformers to be used on this dataset.
        :param oversampling_factor: Increase dataset size by this amount (oversampling).
        :param verbose: Print details to screen.
        """

        self.verbose = verbose
        self.images_root = images_root
        self.labels_root = labels_root
        self.label_names = label_names
        self.optional_transforms = optional_transforms
        self.oversampling_factor = oversampling_factor
        self.imgs = list(natsorted(os.listdir(self.images_root)))
        self.labels = list(natsorted(os.listdir(self.labels_root)))
        # Transforms that are required (making sure it is 3-channel greyscale)
        self.required_transforms = v2.Compose([
            v2.Grayscale(num_output_channels=3)
        ])

        if self.verbose:
            print(f'Initialising the ProstateBladder dataset'
                  f'\tImages root: {self.images_root}.\n'
                  f'\tLabels root: {self.labels_root}.\n'
                  f'\tTotal Images: {len(self.imgs)}.\n'
                  f'\tTotal Labels: {len(self.labels)}.\n'
                  f'\tOversampling count: {self.oversampling_factor}.')

        self.validate_dataset()

    def __getitem__(self, idx):
        """
        Overwrite the __getitem__() method of the parent Dataset class. Images and labels are retrieved from disk
        on an "as needed" basis. The labels are converted from YOLO format, which is normalised
        (x_centre, y_centre, width, height), to (x_min, y_min, x_max, y_max) in pixel coordinates. The targets dict
        variable is created with:
            - At least:
                - boxes: tv_tensors.BoundingBoxes in the xyxy format.
                - labels: Class labels. Depending on the model, this may include background as '0'. If there is a
                  problem during training (particularly CUDA issues), check that the number of labels matches the
                  model requirements.
            - Possibly optional:
                - image_id: Unique image identifier.
                - area: Area of the bounding boxes.
                - iscrowd: Are the instances considered a crowd.
        Reads the image and mask from the disk, converts them to relative tv_tensors,
        applies necessary then optional transforms (optional only applied if over_sampling > 1), and can display
        images and labels before and after transforms if verbose == True.

        :param idx: Index of the required image-label set.

        :return: The image and the targets dict, post transformation.
        """
        original_idx = idx // self.oversampling_factor
        img_path = os.path.join(self.images_root, self.imgs[original_idx])
        label_path = os.path.join(self.labels_root, self.labels[original_idx])
        img = self._read_image(img_path)
        # Get targets (bounding boxes, labels, etc.)
        target = self.get_targets(label_path, img.shape, idx)

        # Convert img to the format expected by the model.
        img = torch.tensor(img, dtype=torch.float32)
        img = tv_tensors.Image(img)

        # Apply optional transforms.
        if self.optional_transforms:
            # Only apply optional transforms on oversampled images.
            if idx % self.oversampling_factor != 0:
                img, target = self.optional_transforms(img, target)
        # Apply required transforms.
        img, target = self.required_transforms(img, target)

        return img, target

    def __len__(self):
        # Total images in dataset = input images * oversampling factor.
        return len(self.imgs) * self.oversampling_factor

    def get_targets(self, label_path, img_size, idx):
        """
        Get target data from label_path file. Conversion from YOLO format is done here.

        :param label_path: Path to label.txt file.
        :param img_size: Size of image (width, height).
        :param idx: Index of image.

        :return: Target dict, varies based on model type.
        """
        # Read the label file assuming YOLO format.
        boxes = []
        labels = []
        with open(label_path) as f:
            img_height, img_width = img_size
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])

                # Convert from YOLO format to (x_min, y_min, x_max, y_max) format.
                x_min = (x_center - width / 2) * img_width
                y_min = (y_center - height / 2) * img_height
                x_max = (x_center + width / 2) * img_width
                y_max = (y_center + height / 2) * img_height

                boxes.append([x_min, y_min, x_max, y_max])

                labels.append(class_id)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(img_height, img_width)),
            "labels": labels
        }

        return target

    def get_label_name(self, label: int):
        """
        Return the label name associated with the given label. If no self.label_names==None, the int label is returned
        as a string.

        :param label: Label integer value (return by model).
        :return: Label string name.
        """
        if self.label_names:
            return self.label_names[label]
        else:
            if self.verbose:
                print(f'No label_names given during initialisation.')
            return f'{label}'

    def get_image_count(self):
        # Total input images, before oversampling.
        return len(self.imgs)

    def validate_dataset(self):
        """
        Minor dataset validation:
            1. Ensure there are the same amount of images as labels.
            2. Ensure the labels and images have the same naming convention.
        """
        if self.verbose:
            print(f'Running dataset validation...')
        # Validation 1.
        assert len(self.imgs) == len(self.labels), "Dataset input images and labels are of different length."
        # Validation 2.
        for i in range(len(self.imgs)):
            if self.imgs[i].split('.')[0] != self.labels[i].split('.')[0]:
                assert False, f"There is a mismatch between imgs and labels at {self.imgs[i]} and {self.labels[i]}."

        if self.verbose:
            print(f'Dataset validation passed.')

    def display_transforms(self, idx):
        """
        This function will plot the original image with bounding boxes and the transformed image with transformed
        bounding boxes. Shear + Rotate transforms may make it seem like the transformed bounding boxes are out of place,
        but it should be fine. Boxes are displayed using 1 of 4 colours, with no special order. The original dataset
        was greyscale, so the ax.imshow cmap is set to greyscale, and the output of the necessary transforms is a single
        channel greyscale image, as opposed to a 3 channel greyscale image as use din the __getitem__() method.

        :param idx: Index of image.
        """
        original_idx = idx // self.oversampling_factor
        img_path = os.path.join(self.images_root, self.imgs[original_idx])
        label_path = os.path.join(self.labels_root, self.labels[original_idx])
        img = self._read_image(img_path)

        # Get targets (bounding boxes, labels, etc.)
        target = self.get_targets(label_path, img.shape, idx)

        boxes = target['boxes']
        img = torch.tensor(img, dtype=torch.float32)
        img = tv_tensors.Image(img)

        colours = ['g', 'b', 'r', 'magenta']
        _, ax = plt.subplots(2)
        # Show original image with original bounding boxes.
        ax[0].imshow(np.transpose(img, (1, 2, 0)), cmap='grey')
        for index, b in enumerate(boxes):
            patch = patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], linewidth=1,
                                      edgecolor=colours[index], facecolor='none')
            ax[0].add_patch(patch)

            # Apply optional transforms.
            if self.optional_transforms is not None:
                # Only apply optional transforms on oversampled images.
                if idx % self.oversampling_factor != 0:
                    img, target = self.optional_transforms(img, target)
            # Apply required transforms (slightly altered from __getitem() as ax.imshow(cmap='gray') has limitations.
            required_transforms = v2.Compose([
                v2.Grayscale(num_output_channels=1)
            ])
            img, target = required_transforms(img, target)

        boxes = target['boxes']

        # Show transformed image with transformed bounding boxes.
        ax[1].imshow(np.transpose(img, (1, 2, 0)), cmap='gray')
        for index, b in enumerate(boxes):
            patch = patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], linewidth=1,
                                      edgecolor=colours[index % len(colours)], facecolor='none')
            ax[1].add_patch(patch)
        plt.show()

    def get_mean_and_std(self):
        """
        Return the mean and std of the current dataset, scaled between (0, 1). The assumption is that the input images
        are scaled to (0, 255), so if the mean and std seem wildly off just make sure the input pixels are scaled
        correctly. These values are required for the normalising transform, and the training values should be used on
        the validation dataset.

        :return: dataset_mean, dataset_std.
        """
        means = []
        stds = []
        for image in self.imgs:
            # Read image using cv2.
            image = cv2.imread(os.path.join(self.images_root, image), cv2.IMREAD_GRAYSCALE) / 255
            # Calculate mean and std using cv2.
            m, s = cv2.meanStdDev(image)
            means.append(m)
            stds.append(s)

        dataset_mean = np.mean(means)
        dataset_std = np.mean(stds)

        if self.verbose:
            print(f'Calculated Dataset Mean: {dataset_mean}.\n'
                  f'Calculated Dataset Standard Deviation: {dataset_std}.')

        return dataset_mean, dataset_std

    @staticmethod
    def _read_image(img_path):
        """
        Moved to a separate function as more than one method makes use of this call, so a single change here will
        subsequently update all other calls.

        Images are read as greyscale, then the range is changed to [0, 1].

        :param img_path: Path to image that must be read from disk.
        :return:
        """
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255
        return img
