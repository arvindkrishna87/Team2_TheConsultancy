#!/bin/bash
# Steps 1-2: KimiaNet embeddings for the new high-grade patches (cases 8, 20, 33, 35, 62, 102).
# Existing .pt files are skipped, so re-running after new case 62 slices are added
# only embeds the new patches. Delete a .pt (or add --overwrite) to redo a re-patched one.
#
#SBATCH --account=e32998
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --job-name=embed_new_cases
#SBATCH --output=precompute_pooled_features_%j.out
#SBATCH --error=precompute_pooled_features_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=callieberthold2027@u.northwestern.edu

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env

# Krish's base code (unchanged modules, KimiaNet weights) and this folder
BASE=/projects/e32998/STAT390_Fall2026/Team6_InstructionalTeam/Code/Presentation2_base_code
CALLIE=/projects/e32998/STAT390_Fall2026/Team2_TheConsultancy/Presentation2/Callie_Presentation2_code
DATA=/projects/e32998/STAT390_Fall2026/Team2_TheConsultancy/Presentation2/Callie_Presentation2_data
CODE=$CALLIE/2_code
# Modified/new files are found before the base code's versions
export PYTHONPATH="$CODE/modified:$CODE/new:$BASE"

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

python3 "$CODE/modified/precompute_pooled_features.py" \
    --patches_dir "$DATA/1_new_case_patches/patches" \
    --embeddings_dir "$DATA/1_new_case_patches/embeddings" \
    --kimianet_ckpt "$BASE/KimiaNet.pth" \
    --device cpu \
    --num_workers 4

echo "End time: $(date)"
