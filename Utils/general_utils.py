import platform
from collections import defaultdict
from os.path import join

import numpy as np
import torch
from matplotlib import pyplot as plt, patches

from . import box_colours


def plot_validation_results(validation_detections, validation_images, starting_label, detection_count, counter,
                            save_path):
    """
    Draw input images with detected bounding boxes on them. Only the top scoring box of each label/class
    is displayed. Since FasterRCNN using label 0 for background and RetinaNet using label 0 for the first
    class, there is an offset that is set using starting_label (for selecting box colour).
    
    :param validation_detections: Detection returned by the model in eval() mode.
    :param validation_images: Images that were given to the model for detection.
    :param detection_count: Maximum number of detections (boxes) per class to be displayed.
    :param starting_label: Lowest label value (RetinaNet = 0, FasterRCNN = 1 since 0 is background).
    :param counter: Image counter, based on batch_size, for saving images with unique names while maintaining
                    validation dataset size.
    :param save_path: Save directory.
    """
    batch_number = counter
    # Since batches are used, detections per image are delt with incrementally.
    for index, output in enumerate(validation_detections):
        # Highest scoring box per label.
        highest_scoring_boxes = defaultdict(lambda: {'scores': [], 'boxes': []})

        labels = output['labels'].cpu().tolist()
        scores = output['scores'].cpu().tolist()
        boxes = output['boxes'].cpu().tolist()

        # Group detections by class
        class_detections = defaultdict(list)
        for label, score, box in zip(labels, scores, boxes):
            class_detections[label].append((score, box))

        # Sort and select top x boxes for each class
        for label, items in class_detections.items():
            # Sort items by score in descending order
            sorted_items = sorted(items, key=lambda item: item[0], reverse=True)
            # Select the top scoring items.
            top_items = sorted_items[:detection_count]
            # Store the scores and boxes in the dictionary.
            highest_scoring_boxes[label]['scores'] = [item[0] for item in top_items]
            highest_scoring_boxes[label]['boxes'] = [item[1] for item in top_items]

        _, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(np.transpose(validation_images[index].to('cpu'), (1, 2, 0))[:, :, 0], cmap='gray')
        for label, label_results in highest_scoring_boxes.items():
            scores = label_results['scores']
            boxes = label_results['boxes']
            for j, s in enumerate(scores):
                box = boxes[j]
                patch = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1,
                                          edgecolor=box_colours[label - starting_label], facecolor='none')
                ax.add_patch(patch)
                ax.text(box[0], box[1], f'{label}: {s:0.1f}', ha='left', color=box_colours[label - starting_label],
                        weight='bold', va='bottom')

        plt.savefig(join(save_path, f'val_result_{batch_number}.png'))
        plt.close()
        batch_number += 1


def get_device_name():
    """
    Return the name of the device being used by torch (GPU name or CPU name.

    :return: Name of torch device.
    """
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(torch.device('cuda'))
    else:
        return platform.processor()
