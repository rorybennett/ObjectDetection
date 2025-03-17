from ultralytics import YOLO

from Utils.arg_parsers import YOLOv8ArgParser

########################################################################################################################
# Fetch parser and set variables from parser args.
########################################################################################################################
arg_parser = YOLOv8ArgParser()
args = arg_parser.parse_args()
project_path = args.project_path  # Project folder.
name = args.name  # Training run name.
model_size = args.model_size  # Size of model to use.
dataset_yaml_path = args.dataset_yaml_path  # Path to dataset .yaml file.
flip_lr = args.flip_lr  # Probability of left-right flipping.
epochs = args.epochs  # Maximum number of epochs.
patience = args.patience  # Training patience.

########################################################################################################################
# Fetch base model (no pretrained weights).
########################################################################################################################
model = YOLO(f"yolov8{model_size}.yaml")


########################################################################################################################
# Train model.
########################################################################################################################
def main():
    model.train(data=dataset_yaml_path,
                project=project_path,
                name=name,
                epochs=epochs,
                imgsz=600,
                cache=False,
                pretrained=False,
                verbose=False,
                workers=0,
                batch=0.6,
                cos_lr=True,
                plots=True,
                patience=patience,
                val=True,
                shear=15,
                mosaic=0,
                fliplr=flip_lr,
                scale=0.3,
                dropout=0.2,
                hsv_h=0,
                hsv_s=0,
                hsv_v=0,
                degrees=30,
                translate=0.2,
                erasing=0.5)


if __name__ == '__main__':
    main()
