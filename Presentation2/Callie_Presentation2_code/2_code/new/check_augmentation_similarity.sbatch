#!/bin/bash
# Step 3 check: how far does each transform (flips, rotations, color / H&E jitter)
# move a patch's KimiaNet embedding? 500 random training patches.
# Original results: 3_results/step3_transform_check/. Re-runs write to reruns/.
#
#SBATCH --account=e32998
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --job-name=aug_similarity
#SBATCH --output=aug_similarity_%j.out
#SBATCH --error=aug_similarity_%j.err

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env

BASE=/projects/e32998/STAT390_Fall2026/Team6_InstructionalTeam/Code/Presentation2_base_code
CALLIE=/projects/e32998/STAT390_Fall2026/Team2_TheConsultancy/Presentation2/Callie_Presentation2_code
CODE=$CALLIE/2_code
export PYTHONPATH="$CODE/modified:$CODE/new:$BASE"

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

python3 "$CODE/new/check_augmentation_similarity.py" \
    --patches_dir /projects/e32998/patches_benign_split \
    --kimianet_ckpt "$BASE/KimiaNet.pth" \
    --out_dir "$CALLIE/reruns/step3_transform_check" \
    --n 500 \
    --device cpu

echo "End time: $(date)"
