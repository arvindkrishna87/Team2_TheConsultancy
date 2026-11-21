#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=short
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=ncv_report
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
source "${SLURM_SUBMIT_DIR}/nested_settings.sh"
activate_env
python3 -u report_nested_cv.py --runs_dir "${RUNS_DIR}" --out_dir "${REPORT_DIR}"
