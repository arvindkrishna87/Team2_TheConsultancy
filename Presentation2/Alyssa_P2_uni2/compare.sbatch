#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=short
#SBATCH --time=00:20:00
#SBATCH --ntasks=2
#SBATCH --mem=8G
#SBATCH --job-name=compare
#SBATCH --output=logs/compare.out
#SBATCH --error=logs/compare.err

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env

cd /home/npg2887/Alyssa_P2_uni2
python3 compare_encoders.py --runs_file /home/npg2887/run_folders.txt --out_dir /home/npg2887/uni2_vs_kimianet
