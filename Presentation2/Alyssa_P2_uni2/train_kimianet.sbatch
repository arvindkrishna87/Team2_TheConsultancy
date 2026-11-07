#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=64G
#SBATCH --array=1-5
#SBATCH --job-name=kimia_train
#SBATCH --output=logs/kimia_split%a.out
#SBATCH --error=logs/kimia_split%a.err

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env
python3 -m pip install -q -r requirements.txt

# stagger the starts so two splits never get the same timestamped run folder
sleep $(( SLURM_ARRAY_TASK_ID * 20 ))

SPLIT=$(printf "%02d" ${SLURM_ARRAY_TASK_ID})
export RUN_TAG="_kimianet_split${SPLIT}" 
SPLITS_DIR=/projects/e32998/STAT390_Fall2026/Team6_InstructionalTeam/Code/Presentation1_base_code/no_train_val_leakage_splits

cd /home/npg2887/Alyssa_Presentation2_base_code
python3 main.py \
    --load_splits ${SPLITS_DIR}/data_splits_new_${SPLIT}.npz \
    --per_slice_cap 500 \
    --max_slices_per_stain 5 \
    --analyze_attention \
    --attention_top_n 8