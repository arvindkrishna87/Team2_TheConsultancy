#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=short
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=ncv_folds
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Outer folds A-E. If the instructors provide 5 split files for the 99 cases, add
#   --from_splits_dir <their folder>
# to use their test folds instead of making new ones.
source "${SLURM_SUBMIT_DIR}/nested_settings.sh"
activate_env
python3 -u make_folds.py --labels_csv "${LABELS_CSV}" --patches_dir "${PATCHES_DIR}" --out "${FOLDS}"
