#!/bin/bash
# Step 4: train the baseline and the flip-augmented model on all 5 splits (10 runs).
#   array 0-4 : baseline, splits 01-05   (original embeddings only)
#   array 5-9 : flip,     splits 01-05   (each epoch: original / hflip / vflip per patch, 1/3 each)
# Identical settings otherwise, so differences come from augmentation alone.
# Submit only the baseline half with --array=0-4, only flip with --array=5-9.
# Original results: 3_results/runs/. Re-runs write to reruns/runs/.
#
#SBATCH --account=e32998
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-9
#SBATCH --job-name=mil_base_vs_flip
#SBATCH --output=train_%A_%a.out
#SBATCH --error=train_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=callieberthold2027@u.northwestern.edu

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env

BASE=/projects/e32998/STAT390_Fall2026/Team6_InstructionalTeam/Code/Presentation2_base_code
CALLIE=/projects/e32998/STAT390_Fall2026/Team2_TheConsultancy/Presentation2/Callie_Presentation2_code
DATA=/projects/e32998/STAT390_Fall2026/Team2_TheConsultancy/Presentation2/Callie_Presentation2_data
CODE=$CALLIE/2_code
export PYTHONPATH="$CODE/modified:$CODE/new:$BASE"

SPLIT_DIR=$BASE/no_train_val_leakage_splits
AUG_ROOT=$DATA/4_flipped_embeddings
RUNS_ROOT=$CALLIE/reruns/runs

T=$SLURM_ARRAY_TASK_ID
SPLIT=$(printf "%02d" $(( T % 5 + 1 )))
if [ "$T" -lt 5 ]; then
    MODEL=baseline
    AUG_ARGS=""
else
    MODEL=flip
    AUG_ARGS="--aug_emb_dirs $AUG_ROOT/hflip $AUG_ROOT/vflip"
fi

echo "Task $T: model=$MODEL split=$SPLIT"
echo "Start time: $(date)"

# Same settings as run_training_custom_split.sbatch (attention analysis left off:
# it does not affect training and is not needed for the comparison).
# Original embeddings come from main.py's default --emb_dir (/projects/e32998/patches_varsize_pooled4096).
python3 "$CODE/modified/main.py" \
    --per_slice_cap 500 \
    --max_slices_per_stain 5 \
    --seed 42 \
    --load_splits "$SPLIT_DIR/data_splits_new_${SPLIT}.npz" \
    --runs_dir "$RUNS_ROOT/$MODEL/split$SPLIT" \
    --run_name "${MODEL}_split${SPLIT}" \
    $AUG_ARGS

echo "End time: $(date)"
