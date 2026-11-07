#!/bin/bash
# Step 3: build flipped KimiaNet embeddings for every training patch.
# Job array: task 0 = horizontal flip, task 1 = vertical flip (run in parallel, ~3 h each on CPU).
# Output filenames match the originals so the training dataset can swap them in.
#
# Re-running is safe: existing .pt files are skipped. To add the new cases
# (8/20/33/35/62/102), un-comment their line in PATCH_DIRS and re-run;
# only the new patches are embedded.
#
#SBATCH --account=e32998
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=0-1
#SBATCH --job-name=flip_embed
#SBATCH --output=flip_embed_%A_%a.out
#SBATCH --error=flip_embed_%A_%a.err
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

AUGS=(hflip vflip)
AUG=${AUGS[$SLURM_ARRAY_TASK_ID]}

# Where flipped embeddings go (~1.3 GB and ~82k files per flip).
OUT_ROOT=$DATA/4_flipped_embeddings
# Patch folders to embed.
PATCH_DIRS=(
    /projects/e32998/patches_benign_split
    # "$DATA/1_new_case_patches/patches"
)

echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}: augment=$AUG"
echo "Start time: $(date)"

for PD in "${PATCH_DIRS[@]}"; do
    python3 "$CODE/modified/precompute_pooled_features.py" \
        --patches_dir "$PD" \
        --embeddings_dir "$OUT_ROOT/$AUG" \
        --kimianet_ckpt "$BASE/KimiaNet.pth" \
        --augment "$AUG" \
        --device cpu \
        --num_workers 4 \
        --log_every 50
done

echo "End time: $(date)"
