#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=16G
#SBATCH --job-name=patch_features_v2
#SBATCH --output=patch_features_v2_%j.out
#SBATCH --error=patch_features_v2_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=izabellakulczycka2027@u.northwestern.edu

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env

# ---- EDIT THESE ---------------------------------------------------------------
CODE_DIR=/home/pnx4472/Presentation3_base_code          # build_patch_features_v2.py, find_missing_embeddings.py, validate_metadata.py
SPLITS_DIR=/home/pnx4472/Presentation3_base_code/no_train_val_leakage_splits    # data_splits_new_01..05.npz
LABELS_CSV=/home/pnx4472/Presentation3_base_code/extra_benign_case_grade_match.csv
PATCHES_DIR=/projects/e32998/patch_images
EMB_DIR=/projects/e32998/STAT390_Fall2026/Team2_TheConsultancy/Presentation2/patch_embeddings_UNI2
OUT_DIR=$CODE_DIR/patch_features_v2
# --------------------------------------------------------------------------------

cd $CODE_DIR

# patches_v2.parquet + case_features_v2.parquet / .csv
python3 build_patch_features_v2_FINAL.py \
    --patches_dir $PATCHES_DIR --emb_dir $EMB_DIR --labels_csv $LABELS_CSV \
    --splits_dir $SPLITS_DIR --out_dir $OUT_DIR --overwrite
