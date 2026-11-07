#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --array=0-3
#SBATCH --job-name=uni2_embed
#SBATCH --output=logs/uni2_embed_%a.out
#SBATCH --error=logs/uni2_embed_%a.err

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env
python3 -m pip install -q "timm>=1.0"

export NUM_SHARDS=4

cd /home/npg2887/Alyssa_Presentation2_base_code
python3 precompute_uni2.py --num_workers 6