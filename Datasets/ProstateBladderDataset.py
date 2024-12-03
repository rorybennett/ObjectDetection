"""
Custom dataset class for the prostate and bladder detection. A single prostate bounding box and single bladder
bounding box are present for each image. The dataset labels should be organised in the YOLO format. Images
will be stored in the root/images directory and labels in the root/labels directory.
"""
import os

import torch
from PIL import Image
from natsort import natsorted
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

from . import datasets_model_types


class ProstateBladderDataset(torch.utils.data.Dataset):
    def __init__(self, root, label_names=None, transforms=None, oversampling_factor=1, model_type=None, verbose=False):
        """
        Initialise the ProstateBladderDataset class. Set the dataset root directory, the transforms that are to be used
        when __getitem__() is called, the type of model that will be accessing this dataset class, and perform minor
        dataset validation. The model_type parameter is necessary as different models follow different conventions.

        :param root: Directory containing images and labels folders.
        :param label_names: List of names for associated labels.
        :param transforms: Transformers to be used on this dataset.
        :param oversampling_factor: Increase dataset size by this amount (oversampling).
        :param model_type: String representation of the model type this dataset is used with.
        :param verbose: Print details to screen.
        """

        self.verbose = verbose
        self.root = root
        self.label_names = label_names
        self.transforms = transforms
        self.oversampling_factor = oversampling_factor
        self.model_type = model_type
        self.imgs = list(natsorted(os.listdir(os.path.join(root, "images"))))
        self.labels = list(natsorted(os.listdir(os.path.join(root, "labels"))))

        if self.verbose:
            print(f'Initialising the ProstateBladder dataset with root: {self.root}.\n'
                  f'Total Images: {len(self.imgs)}.\n'
                  f'Total Labels: {len(self.labels)}.\n'
                  f'Oversampling count: {self.oversampling_factor}.')

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

        :param idx: Index of the required image-label set.

        :return: The image and the targets dict, post transformation.
        """
        original_idx = idx // self.oversampling_factor
        img_path = os.path.join(self.root, "images", self.imgs[original_idx])
        label_path = os.path.join(self.root, "labels", self.labels[original_idx])
        img = Image.open(img_path).convert("RGB")

        # Read the label file.
        boxes = []
        labels = []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])

                # Convert from YOLO format to (x_min, y_min, x_max, y_max) format.
                img_width, img_height = img.size
                x_min = (x_center - width / 2) * img_width
                y_min = (y_center - height / 2) * img_height
                x_max = (x_center + width / 2) * img_width
                y_max = (y_center + height / 2) * img_height

                boxes.append([x_min, y_min, x_max, y_max])

                if self.model_type == datasets_model_types['fasterrcnn_resnet50_fpn_v2']:
                    # Add 1 as 0 is always background for fasterrcnn.
                    labels.append(class_id + 1)
                else:
                    labels.append(class_id)
        # Convert to the format expected by the model.
        img = tv_tensors.Image(img)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Create targets dict, depends on model type.
        if self.model_type == datasets_model_types['fasterrcnn_resnet50_fpn_v2']:
            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
            target = {
                "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img)),
                "labels": labels,
                "image_id": image_id,
                "area": area,
                "iscrowd": iscrowd
            }
        else:
            target = {
                "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img)),
                "labels": labels
            }

        if self.transforms is not None:
            # Apply transforms. This proved problematic, if something seems off plot the before and after.
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        # Total images = input images * oversampling factor.
        return len(self.imgs) * self.oversampling_factor

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
        # 1
        assert len(self.imgs) == len(self.labels), "Dataset input images and labels are of different length."
        # 2
        for i in range(len(self.imgs)):
            if self.imgs[i].split('.')[0] != self.labels[i].split('.')[0]:
                assert False, f"There is a mismatch between imgs and labels at {self.imgs[i]} and {self.labels[i]}."

        if self.verbose:
            print(f'Dataset validation passed.')
