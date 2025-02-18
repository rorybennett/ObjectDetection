#$ -pe smp 8            # Number of cores.
#$ -l h_vmem=11G        # Memory per core (max 11G per GPU).
#$ -l h_rt=240:0:0        # Requested runtime.
#$ -cwd                 # Change to current directory.
#$ -j y                 # Join output and error files.
#$ -o outputs/          # Change default output directory.
#$ -l gpu=1             # Request GPU usage.
#$ -l gpu_type=ampere   # GPU type
#$ -t 1-2               # Array job.
#$ -tc 2                # Concurrent jobs.
#$ -m bea               # Email beginning, end, and aborted.

module load python

# Activate virtualenv
source .venv/bin/activate

inputs=(
    "ProspectiveData/prostate_Combined/images/train_4"
    "ProspectiveData/prostate_Combined/labels/train_4"
    "ProspectiveData/prostate_Combined/images/val_4"
    "ProspectiveData/prostate_Combined/labels/val_4"
    "ProspectiveData/prostate_Combined_patience100/fold_4"
    "ProspectiveData/prostate_Combined/images/train_all"
    "ProspectiveData/prostate_Combined/labels/train_all"
    "ProspectiveData/prostate_Combined/images/val_all"
    "ProspectiveData/prostate_Combined/labels/val_all"
    "ProspectiveData/prostate_Combined_patience100/fold_all"
)


training_images_path=${inputs[$((5 * SGE_TASK_ID - 5))]}
training_labels_path=${inputs[$((5 * SGE_TASK_ID - 4))]}
val_images_path=${inputs[$((5 * SGE_TASK_ID - 3))]}
val_labels_path=${inputs[$((5 * SGE_TASK_ID - 2))]}
saving_path=${inputs[$((5 * SGE_TASK_ID - 1))]}

python train_fasterrcnn.py \
  -tip="./ObjectDetectionDatasets/$training_images_path" \
  -tlp="./ObjectDetectionDatasets/$training_labels_path" \
  -vip="./ObjectDetectionDatasets/$val_images_path" \
  -vlp="./ObjectDetectionDatasets/$val_labels_path" \
  -sp="/data/scratch/exx851/FasterRCNN/$saving_path" \
  -e=1000 \
  -lres=1000 \
  -nc=2 \
  -bs=32 \
  -of=8 \
  -p=100
