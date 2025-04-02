# Object Detection
The models in this repo were used during the completion of a PhD project looking at estimating
the size of the prostate using abdominal ultrasound images. This was then expanded to more general
object detection in medical images.

A two-step approach was followed:

1. Detect the prostate and bladder.
2. Segment the prostate and bladder.

The results of the detection step were fed into the segmentation step to substantially 
improve the segmentation results. This repo contains the setup for training and testing/validating 
the object detection models. The segmentation setup can be found [here (not yet though)].

## Object Detection Models
Three object detection models were trained, tested, validated, and compared:

1. YOLO: YOLOv1 and YOLOv2 were built from scratch, whereas YOLOv3 onwards made used of the
Ultralytics package.
2. FasterRCNN: A faster rcnn model using a Resnet50 FPN backbone (version 2). Other backbones
are also available, with some of them being faster than others.
3. RetinaNet: A retinanet model using a Resnet50 FPN backbone (version 2). Other backbones
are available, but it seems that the ResNet50 FPN backbone is the fastest.

After much back-and-forth, Resnet50_fpn_v2 seems to be the smallest backbone for RetinaNet.
Using the example (for RetinaNet) that has mobilenet_v2 as the backbone results in a save file that is far
larger than the Resnet50_fpn_v2 backbone.

## Repo contents
The packages in this repo were created under the assumption of prostate and bladder detection,
however, minor modifications should allow for other types of object detection.

1. Datasets:
   - A dataset class is made for each model, where required. It is assumed that the datasets
   are stored in the YOLOv8 format (as that is what I started with). This means that sometimes
   a conversion is required (e.g. YOLOv1 and YOLOv2 use different formats) during data loading. 
   At some point a more general dataset class will be created, but for now it is assumed that
   only prostate/bladder detection is being conducted. There is a `display_transforms()` 
   function that enables verification of bounding box transforms as there was quite a bit
   of trouble setting the transforms up in the correct manner.

2. EarlyStopping:
   - `EarlyStopping.py`: Class that checks for improvements in validation set during training.
   If no improvement is detected after a set number of epochs, training is terminated. The
   minimum improvement necessary for continued training is defined by a delta value and the
   total number of training epochs without improvement before training is stopped is set by
   the patience value. The latest model as well as the best model are both saved, with the
   option of disabling saving of the latest model. Model saving also occurs in this class. This 
   class helps prevent overfitting using the validation set, without using the validation
   set for parameter training. A patience of zero effectively disables early stopping. 
   Optimiser parameters are not saved by default (so at the moment you cannot continue training)
   but this can be changed by uncommenting certain lines in the `save_checkpoint()` function.
 
3. DetectionModels:
   - `CustomFasterRCNN.py`: Makes use of the `fasterrcnn_resnet50_fpn_v2` (by default)
   model with no pretrained weights and a slightly altered forward pass. If the model is in 
   evaluation mode and targets are given then both the losses and the detections are returned 
   (this is to enable validation during training to prevent possible overfitting). 
   - `RetinaNet.py`: Makes use of the `retinanet_resnet50_fpn_v2` (by default) model with no 
   pretrained weights and a slightly altered forward pass. If the model is in evaluation mode and 
   targets are given then both the losses and the detections are returned (this is to enable 
   validation during training to prevent possible overfitting).
   - `yolov1_models.py`: Model architecture for YOLOv1. Attempts to follow the original paper
   as closely as possible. BatchNormalisation was added, variable number of input channels
   should be possible (not 100% tested), and the CNN backbone follows the darknet approach.
   The faster version is also available as `YOLOv1Fast`. Losses are found in the `YOLO/losses.py`
   file.
   - `yolov2_models.py`: Model architecture for YOLOv2. Attempts to follow the original paper
   as closely as possible, however, the paper was sparse on some details so there may have been
   some guesswork. Activation functions are applied at the model level. Losses are found in
   the `YOLO/losses.py` file.
   - `yolov3.py` to `yolovXXX.py`: These versions of YOLO are supported by Ultralytics and
   have been used with no modifications to the underlying codebase. Training parameters are
   tuned as necessary. These have not really been used yet, but will be implemented at some 
   point.

4. Transforms:
   - `Transformers.py`: Contains the training transforms used by some of the models. Can
   be altered as desired. Some transforms do not make use of colour transformations
   as the original dataset was greyscale. Validation transforms are not found here. Certain
   models required transforms (such as resizing) that were placed in the dataset class. 
   Transforms that were not required for the model to function can be though of as extra
   transforms, and are found here.

5. Utils:
   - `args_parsers.py`: Argument parsers used by the training scripts. The default values have
   been set with the initial dataset in mind, so they may need to be changed when calling the 
   respective training script. Paths to training images, training labels, validation images, 
   and validation labels, as well as the number of object classes, are necessary inputs. 
   The rest have default options.
   - `xxx_utils.py`: Extra functions used by various classes. Typically serparated by the models
   that use them.

6. Jobs:
   - A lot of the training was done on a high performance server with access to larger GPUs.
   The job scripts used to run the code on the servers are found here.

## Sample Model Results
The image below shows some sample losses from a FasterRCNN run using mostly default parameters,
except the oversampling_factor was set to 4 in an attempt to increase training dataset
diversity. Note the cos annealing with warm restarts learning rate and the green line indicating
the epoch with the best validation performance. The model saved under "best" will correspond
to the model at this point, while the model saved under "latest" will correspond to the model
from the last epoch run before early stopping kicked in.

![Sample Losses](res/losses_sample.png)

## Example Terminal Call

The snippet below shows an example of how to call the `train_yolov2.py` script from the
terminal. All paths are required, as is the number of classes. NB: FasterRCNN will have
`number_of_classes = number of objects + 1` as it includes the background as class 0. 

```bash
 python -m YOLO.train_yolov2 
 --train_images_path="../Datasets/ObjectDetection/SingleFrame/ProspectiveData/prostateBladder_Combined/images/train_all" 
 --train_labels_path="../Datasets/ObjectDetection/SingleFrame/ProspectiveData/prostateBladder_Combined/labels/train_all" 
 --val_images_path="../Datasets/ObjectDetection/SingleFrame/ProspectiveData/prostateBladder_Combined/images/val_all" 
 --val_labels_path="../Datasets/ObjectDetection/SingleFrame/ProspectiveData/prostateBladder_Combined/labels/val_all" 
 --save_path="YOLOv2 Model Results/prostateBladder_Combined/fold_all" 
 --epochs=50 
 --batch_size=8 
 --oversampling_factor=8 
 --number_of_classes=2 
 --optimiser_learning_rate=0.001 
 --optimiser_learning_rate_restart=200

```
