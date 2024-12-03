# Object Detection
The models in this repo were used during the completion of a PhD looking at estimating
the size of the prostate using abdominal ultrasound images. 

A two-step approach was
followed:

1. Detect the prostate and bladder.
2. Segment the prostate and bladder.

The results of the detection step were fed into the segmentation step to improve the 
results. This repo contains the setup for training and testing the object detection
models. The segmentation setup can be found [here].

## Object Detection Models
Three object detection models were trained, tested, validated, and compared. 