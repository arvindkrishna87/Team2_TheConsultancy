#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=short
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --job-name=ncv_cache
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Loads every UNI2 embedding once (~89k files) into one file (~0.3 GB). Run once.
source "${SLURM_SUBMIT_DIR}/nested_settings.sh"
activate_env
python3 -u nested_cv.py --build_cache --labels_csv "${LABELS_CSV}" --patches_dir "${PATCHES_DIR}" \
    --emb_dir "${UNI2_EMB_DIR}" --cache "${CACHE}"
