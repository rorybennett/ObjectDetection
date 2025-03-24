import os

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt, patches
from natsort import natsorted
from sklearn.cluster import KMeans
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from Utils import yolo_utils


class ProstateBladderDataset(Dataset):
    def __init__(self, images_root, labels_root, label_names=None, transforms=None, train_mean=None,
                 train_std=None, oversampling_factor=1, img_size=416, anchors=None, num_classes=1, verbose=False):
        """
        Initialise the ProstateBladderDataset class, specifically for YOLOv2 implementation.
        Set the images' and labels' root directories, the transforms, that are to be used when __getitem__() is called,
        and perform minor dataset validation.

        :param images_root: Directory containing images.
        :param labels_root: Directory containing labels.
        :param label_names: List of names for associated labels.
        :param transforms: Optional transformers to be used on this dataset.
        :param train_mean: Training dataset mean, in 3 channel format.
        :param train_std: Training dataset standard deviation, in 3 channel format.
        :param oversampling_factor: Increase dataset size by this amount (oversampling).
        :param img_size: Transformed image size (img_size, img_size).
        :params anchors: Anchors calculated using k-means clustering.
        :param num_classes: Number of classes.
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
        self.img_size = img_size
        self.S = img_size // 32
        self.anchors = anchors
        self.num_classes = num_classes
        self.verbose = verbose

        # Sort images and labels into list.
        self.images = list(natsorted(os.listdir(images_root)))
        self.labels = list(natsorted(os.listdir(labels_root)))

        # Transforms that are always applied.
        self.required_transforms = v2.Compose([
            v2.Resize((img_size, img_size)),
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

        # Apply required transforms (resize, normalise, 3-channel greyscale). Change target['img_size'] value.
        img, target = self.required_transforms(img, target)
        target['img_size'] = (self.img_size, self.img_size)

        # Convert target (labels) to YOLOv1 format
        lbl = self.convert_xyxy_to_yolov2(target)

        return img, lbl

    def get_mean_std_and_anchors(self, k=5):
        """
        Calculate dataset mean, std, and k-means anchor boxes. Use this function on a temporary dataset, probably
        a temporary train dataset. The dataset mean, std, and anchor boxes are all returned and can then be
        used in the creation of the "proper" datasets that will be used during training and validation.

        :param k: Number of anchor boxes.

        :return: dataset_mean, dataset_std, anchor_boxes.
        """
        means = []
        stds = []
        bboxes = []

        for index, (image, label) in enumerate(zip(self.images, self.labels)):
            # Read and resize image
            img = cv2.imread(f'{self.images_root}/{image}', cv2.IMREAD_GRAYSCALE) / 255
            # Resize before processing, so anchor boxes are the correct size.
            img = cv2.resize(img, (self.img_size, self.img_size))

            # Calculate mean and std per image.
            m, s = cv2.meanStdDev(img)
            means.append(m)
            stds.append(s)

            # Read bounding box labels
            label_path = os.path.join(self.labels_root, label)
            target = yolo_utils.get_label_data(label_path, img_size=img.shape, idx=index)

            # Extract widths & heights of bounding boxes
            boxes = target["boxes"]
            widths = (boxes[:, 2] - boxes[:, 0]) / self.img_size  # Normalize width
            heights = (boxes[:, 3] - boxes[:, 1]) / self.img_size  # Normalize height

            bboxes.extend(zip(widths, heights))

        # Compute dataset mean & std.
        dataset_mean = np.mean(means)
        dataset_std = np.mean(stds)

        # Apply k-means clustering using normalised boxes.
        bboxes = np.array(bboxes)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(bboxes)

        # Get and sort anchors by area
        anchors = kmeans.cluster_centers_
        anchors = sorted(anchors, key=lambda x: x[0] * x[1])  # Sort by area

        # Scale anchors back to final training size.
        anchors = (np.array(anchors) * self.img_size).astype(int)

        if self.verbose:
            print(f'Calculated Dataset Mean: {dataset_mean}.\n'
                  f'Calculated Dataset Standard Deviation: {dataset_std}.\n'
                  f'Calculated Dataset Anchors: {anchors}.')

        return dataset_mean, dataset_std, anchors

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
        self.draw_yolo_boxes_on_axis(self.anchors, self.S, axes[0], target_original["boxes"],
                                     "Original Image with Bounding Boxes", box_format='xyxy')

        axes[1].imshow(img_transformed_display, cmap="gray", vmin=-2, vmax=2)
        self.draw_yolo_boxes_on_axis(self.anchors, self.S, axes[1], target_transformed,
                                     "Transformed Image with Bounding Boxes", box_format='yolov2')

        plt.show()

    @staticmethod
    def draw_yolo_boxes_on_axis(anchors, S, ax, boxes, title, box_format='xyxy', img_size=416):
        """
        Draws bounding boxes on the provided axis for YOLOv2.

        :param anchors: List of anchor box sizes [(w, h), (w, h), ...].
        :param S: Grid size (e.g., 13 for 416x416 input).
        :param ax: Matplotlib axis to draw on.
        :param boxes: Bounding boxes in YOLOv2 format (num_anchors, 5 + num_classes, S, S).
        :param title: Title of the image.
        :param box_format: 'xyxy' for standard boxes, 'yolov2' for YOLO format.
        :param img_size: The size of the original image (default: 416).
        """
        ax.set_title(title)

        if box_format == 'xyxy':
            # Standard XYXY format
            for box in boxes:
                if len(box) != 4:
                    continue
                x_min, y_min, x_max, y_max = box
                width = max(1, x_max - x_min)
                height = max(1, y_max - y_min)

                rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)

        elif box_format == 'yolov2':
            num_anchors = len(anchors)
            for anchor_idx in range(num_anchors):
                for i in range(S):  # Iterate over grid rows
                    for j in range(S):  # Iterate over grid cols
                        conf = torch.sigmoid(boxes[anchor_idx, 4, i, j])  # Object confidence

                        if conf > 0.5:  # Only draw boxes with confidence > 0.5
                            # Extract (tx, ty, tw, th)
                            tx = torch.sigmoid(boxes[anchor_idx, 0, i, j])  # Center x offset
                            ty = torch.sigmoid(boxes[anchor_idx, 1, i, j])  # Center y offset
                            tw = torch.exp(boxes[anchor_idx, 2, i, j]) * anchors[anchor_idx][0]  # Width
                            th = torch.exp(boxes[anchor_idx, 3, i, j]) * anchors[anchor_idx][1]  # Height

                            # Convert grid-relative coordinates to absolute image coordinates
                            cell_size = img_size / S
                            abs_x = (j + tx) * cell_size
                            abs_y = (i + ty) * cell_size
                            abs_w = tw * img_size
                            abs_h = th * img_size

                            # Convert to XYXY format
                            x_min = abs_x - abs_w / 2
                            y_min = abs_y - abs_h / 2
                            x_max = abs_x + abs_w / 2
                            y_max = abs_y + abs_h / 2

                            # Ensure valid width & height
                            width = x_max - x_min
                            height = y_max - y_min

                            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='red',
                                                     facecolor='none')
                            ax.add_patch(rect)
        return

    def convert_xyxy_to_yolov2(self, target):
        """
        Convert XYXY bounding box labels to YOLOv2 format. This seems to work now. There was a problem with
        transformed boxes not matching the transformed image, but that was due to an incorrect target['img_size'].

        :param target: Dictionary containing "boxes" (XYXY format) and "labels".

        :return: Target tensor formatted for YOLOv2.
        """
        num_anchors = len(self.anchors)
        target_tensor = torch.zeros((num_anchors, 5 + self.num_classes, self.S, self.S))

        # Convert absolute box coords → relative to image size
        img_h, img_w = target["img_size"]
        boxes = target["boxes"] / torch.tensor([img_w, img_h, img_w, img_h])

        for box, label in zip(boxes, target["labels"]):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            xc, yc = (x1 + x2) / 2, (y1 + y2) / 2  # Convert to center coordinates

            grid_x, grid_y = int(xc * self.S), int(yc * self.S)  # Find grid cell
            anchor_idx = torch.argmax(torch.tensor([w * h / (aw * ah) for aw, ah in self.anchors],
                                                   dtype=torch.float32))

            # Normalize bbox params
            tx, ty = xc * self.S - grid_x, yc * self.S - grid_y
            tw, th = torch.log(w / self.anchors[anchor_idx][0] + 1e-6), torch.log(
                h / self.anchors[anchor_idx][1] + 1e-6)

            # Populate target tensor
            target_tensor[anchor_idx, 0, grid_y, grid_x] = tx
            target_tensor[anchor_idx, 1, grid_y, grid_x] = ty
            target_tensor[anchor_idx, 2, grid_y, grid_x] = tw
            target_tensor[anchor_idx, 3, grid_y, grid_x] = th
            target_tensor[anchor_idx, 4, grid_y, grid_x] = 1  # Objectness score
            target_tensor[anchor_idx, 5 + label, grid_y, grid_x] = 1  # Class probability

        return target_tensor
