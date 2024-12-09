#$ -pe smp 12            # Number of cores.
#$ -l h_vmem=7.5G        # Memory per core (max 11G per GPU).
#$ -l h_rt=240:0:0        # Requested runtime.
#$ -cwd                 # Change to current directory.
#$ -j y                 # Join output and error files.
#$ -o outputs/          # Change default output directory.
#$ -l gpu=1             # Request GPU usage.
#$ -t 1-6               # Array job.
#$ -tc 6                # Concurrent jobs.
#$ -m bea               # Email beginning, end, and aborted.

module load python

# Activate virtualenv
source venv/bin/activate

inputs=(
    "ProspectiveData/prostate_Combined/train_0/images"
    "ProspectiveData/prostate_Combined/train_0/labels"
    "ProspectiveData/prostate_Combined/val_0/images"
    "ProspectiveData/prostate_Combined/val_0/labels"
    "ProspectiveData/prostate_Combined/fold_0"
    "ProspectiveData/prostate_Combined/train_1/images"
    "ProspectiveData/prostate_Combined/train_1/labels"
    "ProspectiveData/prostate_Combined/val_1/images"
    "ProspectiveData/prostate_Combined/val_1/labels"
    "ProspectiveData/prostate_Combined/fold_1"
    "ProspectiveData/prostate_Combined/train_2/images"
    "ProspectiveData/prostate_Combined/train_2/labels"
    "ProspectiveData/prostate_Combined/val_2/images"
    "ProspectiveData/prostate_Combined/val_2/labels"
    "ProspectiveData/prostate_Combined/fold_2"
    "ProspectiveData/prostate_Combined/train_3/images"
    "ProspectiveData/prostate_Combined/train_3/labels"
    "ProspectiveData/prostate_Combined/val_3/images"
    "ProspectiveData/prostate_Combined/val_3/labels"
    "ProspectiveData/prostate_Combined/fold_3"
    "ProspectiveData/prostate_Combined/train_4/images"
    "ProspectiveData/prostate_Combined/train_4/labels"
    "ProspectiveData/prostate_Combined/val_4/images"
    "ProspectiveData/prostate_Combined/val_4/labels"
    "ProspectiveData/prostate_Combined/fold_4"
    "ProspectiveData/prostate_Combined/train_all/images"
    "ProspectiveData/prostate_Combined/train_all/labels"
    "ProspectiveData/prostate_Combined/val_all/images"
    "ProspectiveData/prostate_Combined/val_all/labels"
    "ProspectiveData/prostate_Combined/fold_all"
)


training_images_path=${inputs[$((5 * SGE_TASK_ID - 5))]}
training_labels_path=${inputs[$((5 * SGE_TASK_ID - 4))]}
val_images_path=${inputs[$((5 * SGE_TASK_ID - 3))]}
val_labels_path=${inputs[$((5 * SGE_TASK_ID - 2))]}
saving_path=${inputs[$((5 * SGE_TASK_ID - 1))]}

python train_retinanet.py \
  --train_images_path="./Datasets/$training_images_path" \
  --train_labels_path="./Datasets/$training_labels_path" \
  --val_images_path="./Datasets/$val_images_path" \
  --val_labels_path="./Datasets/$val_labels_path" \
  --save_path="/data/scratch/exx851/RetinaNetResults/$saving_path" \
  --oversampling_factor=4
