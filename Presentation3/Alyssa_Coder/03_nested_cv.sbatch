#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=0-4
#SBATCH --job-name=ncv_outer
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# One array task per outer test fold (0=A ... 4=E). Safe to resubmit: finished trials are kept
# in runs/outer_X/optuna.db and an interrupted trial is retried without using up the budget.
# Rerun only some folds with e.g.  sbatch --array=1,3 03_nested_cv.sbatch
source "${SLURM_SUBMIT_DIR}/nested_settings.sh"
activate_env
python3 -u nested_cv.py --folds "${FOLDS}" --cache "${CACHE}" --out_dir "${RUNS_DIR}" \
    --n_trials ${N_TRIALS} --n_startup 16 --per_slice_cap 500 --max_slices_per_stain 5
