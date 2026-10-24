#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=long
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=128G
#SBATCH --array=1-5
#SBATCH --job-name=MIL_train
#SBATCH --output=logs/slurm_%a.out
#SBATCH --error=logs/slurm_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alyssakantner2027@u.northwestern.edu

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env

cd "$SLURM_SUBMIT_DIR"
SEED=1805
TAG=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

# stagger starts so the 5 jobs never create the same timestamped run folder
sleep $(( SLURM_ARRAY_TASK_ID * 15 ))

START=$(date +%s)
python3 main.py \
    --labels_csv extra_benign_case_grade_match.csv \
    --load_splits no_train_val_leakage_splits/data_splits_new_${TAG}.npz \
    --seed $SEED \
    --per_slice_cap 500 \
    --max_slices_per_stain 5 \
    > logs/split_${TAG}.log 2>&1
echo $(( $(date +%s) - START )) > logs/runtime_${TAG}.txt