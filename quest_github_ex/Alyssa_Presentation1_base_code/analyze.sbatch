#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=short
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --output=analyze_output.out
module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env
cd "$SLURM_SUBMIT_DIR"
which python3
python3 analyze_runs.py --logs_dir logs --out_dir analysis --seed 1805
