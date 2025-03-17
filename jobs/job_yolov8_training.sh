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
    "prostateBladder_Transverse" "0" "n"
    "prostateBladder_Transverse" "1" "n"
    "prostateBladder_Transverse" "2" "n"
    "prostateBladder_Transverse" "3" "n"
    "prostateBladder_Transverse" "4" "n"
    "prostateBladder_Transverse" "all" "n"
    "prostateBladder_Transverse" "0" "m"
    "prostateBladder_Transverse" "1" "m"
    "prostateBladder_Transverse" "2" "m"
    "prostateBladder_Transverse" "3" "m"
    "prostateBladder_Transverse" "4" "m"
    "prostateBladder_Transverse" "all" "m"
    "prostateBladder_Transverse" "0" "s"
    "prostateBladder_Transverse" "1" "s"
    "prostateBladder_Transverse" "2" "s"
    "prostateBladder_Transverse" "3" "s"
    "prostateBladder_Transverse" "4" "s"
    "prostateBladder_Transverse" "all" "s"
    "prostateBladder_Transverse" "0" "l"
    "prostateBladder_Transverse" "1" "l"
    "prostateBladder_Transverse" "2" "l"
    "prostateBladder_Transverse" "3" "l"
    "prostateBladder_Transverse" "4" "l"
    "prostateBladder_Transverse" "all" "l"
    "prostateBladder_Transverse" "0" "x"
    "prostateBladder_Transverse" "1" "x"
    "prostateBladder_Transverse" "2" "x"
    "prostateBladder_Transverse" "3" "x"
    "prostateBladder_Transverse" "4" "x"
    "prostateBladder_Transverse" "all" "x"
)

objects=${inputs[$((3 * SGE_TASK_ID - 3))]}
fold=${inputs[$((3 * SGE_TASK_ID - 2))]}
model_size=${inputs[$((3 * SGE_TASK_ID - 1))]}


python -m YOLO.train_yolov8 \
  -dyp="/data/home/exx851/ObjectDetection/ObjectDetectionDatasets/3D/ProspectiveData/$objects/dataset_$fold.yaml" \
  -pp="/data/scratch/exx851/YOLOv8/3D/ProspectiveData" \
  -n="$objects/fold$fold_$model_size" \
  -ms="$model_size" \
  -flr=0.2 \
  -e=1000 \
  -p=100 \
