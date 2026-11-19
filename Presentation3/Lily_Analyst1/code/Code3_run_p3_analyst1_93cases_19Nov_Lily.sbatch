#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=long
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=128G
#SBATCH --job-name=P3_A1
#SBATCH --output=logs/%x_%j.out          # %j = job ID, so parallel jobs don't overwrite each other
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lilywu2027@u.northwestern.edu

# Presentation 3, Analyst choice 1: one experiment per job.
# Usage (run from ~/Presentation3_base_code):
#   sbatch --job-name=baseline run_p3_analyst1.sbatch baseline ""
#   sbatch --job-name=emb256   run_p3_analyst1.sbatch emb256   "--embed_dim 256"
#   sbatch --job-name=emb128   run_p3_analyst1.sbatch emb128   "--embed_dim 128"
#   sbatch --job-name=emb64    run_p3_analyst1.sbatch emb64    "--embed_dim 64"
#   sbatch --job-name=stain03  run_p3_analyst1.sbatch stain03  "--stain_dropout 0.3"
#   sbatch --job-name=slice03  run_p3_analyst1.sbatch slice03  "--slice_dropout 0.3"

EXP_NAME="$1"          # folder name for this experiment
EXTRA_ARGS="$2"        # the one setting that changes

SPLIT=01               # change to 02..05 for the all-splits reruns
CODE_DIR=/home/oky0687/Presentation3_93cases
RUNS_ROOT=/projects/e32998/STAT390_Fall2026/Team2_TheConsultancy/P3_analyst1/runs_93cases

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env

python3 -m pip install -r /projects/e32998/STAT390_Fall2026/Team6_InstructionalTeam/Code/Presentation3_base_code/requirements.txt
python3 -m pip install --upgrade certifi
export SSL_CERT_FILE="$(python -m certifi)"

cd "$CODE_DIR"
mkdir -p logs "$RUNS_ROOT/split$SPLIT/$EXP_NAME"

echo "Job ID: $SLURM_JOB_ID | Experiment: $EXP_NAME | Split: $SPLIT | Args: $EXTRA_ARGS"
echo "Start time: $(date)"

# Same settings as the base sbatch for every experiment; only EXTRA_ARGS differs.
python3 main.py \
    --per_slice_cap 500 \
    --max_slices_per_stain 5 \
    --seed 42 \
    --patches_dir /projects/e32998/patches_benign_split \
    --emb_dir /projects/e32998/STAT390_Fall2026/Team5_TheWildcards/patch_embeddings_UNI2 \
    --load_splits "$CODE_DIR/splits_93cases/data_splits_new_$SPLIT.npz" \
    --runs_dir "$RUNS_ROOT/split$SPLIT/$EXP_NAME" \
    $EXTRA_ARGS

echo "End time: $(date)"
