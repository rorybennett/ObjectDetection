#$ -pe smp 8            # Number of cores.
#$ -l h_vmem=11G        # Memory per core (max 11G per GPU).
#$ -l h_rt=240:0:0        # Requested runtime.
#$ -cwd                 # Change to current directory.
#$ -j y                 # Join output and error files.
#$ -o outputs/          # Change default output directory.
#$ -l gpu=1             # Request GPU usage.
#$ -l gpu_type=ampere   # GPU type
#$ -t 1-30               # Array job.
#$ -tc 6                # Concurrent jobs.
#$ -m bea               # Email beginning, end, and aborted.
#$ -l rocky

module load python

# Activate virtualenv
source .venv/bin/activate

inputs=(
    "prostate_Transverse" "0" "n"
    "prostate_Transverse" "1" "n"
    "prostate_Transverse" "2" "n"
    "prostate_Transverse" "3" "n"
    "prostate_Transverse" "4" "n"
    "prostate_Transverse" "all" "n"
    "prostate_Transverse" "0" "m"
    "prostate_Transverse" "1" "m"
    "prostate_Transverse" "2" "m"
    "prostate_Transverse" "3" "m"
    "prostate_Transverse" "4" "m"
    "prostate_Transverse" "all" "m"
    "prostate_Transverse" "0" "s"
    "prostate_Transverse" "1" "s"
    "prostate_Transverse" "2" "s"
    "prostate_Transverse" "3" "s"
    "prostate_Transverse" "4" "s"
    "prostate_Transverse" "all" "s"
    "prostate_Transverse" "0" "l"
    "prostate_Transverse" "1" "l"
    "prostate_Transverse" "2" "l"
    "prostate_Transverse" "3" "l"
    "prostate_Transverse" "4" "l"
    "prostate_Transverse" "all" "l"
    "prostate_Transverse" "0" "x"
    "prostate_Transverse" "1" "x"
    "prostate_Transverse" "2" "x"
    "prostate_Transverse" "3" "x"
    "prostate_Transverse" "4" "x"
    "prostate_Transverse" "all" "x"
)

objects=${inputs[$((3 * SGE_TASK_ID - 3))]}
fold=${inputs[$((3 * SGE_TASK_ID - 2))]}
model_size=${inputs[$((3 * SGE_TASK_ID - 1))]}


python -m YOLO.train_yolov8 \
  -dyp="/data/home/exx851/ObjectDetection/ObjectDetectionDatasets/3D/ProspectiveData/$objects/dataset_$fold.yaml" \
  -pp="/data/scratch/exx851/YOLOv8/3D/ProspectiveData" \
  -n="$objects/fold$fold _$model_size" \
  -ms="$model_size" \
  -flr=0.2 \
  -e=1000 \
  -p=100 \
