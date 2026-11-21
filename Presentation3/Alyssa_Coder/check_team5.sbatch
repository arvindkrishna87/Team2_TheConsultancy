#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=short
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --job-name=check_team5
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env
python3 -m pip install -q "timm>=1.0" huggingface_hub
cd /home/npg2887/Alyssa_Presentation3_base_code
python3 -u check_team5_embeddings.py \
    --emb_dir /projects/e32998/STAT390_Fall2026/Team5_TheWildcards/patch_embeddings_UNI2 \
    --patches_dir /projects/e32998/patch_images --n 30 --seed 1
