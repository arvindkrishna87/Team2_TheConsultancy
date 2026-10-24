#!/bin/bash
#SBATCH --account=e32998                  # Use your Quest project account (check with 'groups')
#SBATCH --partition=long                  # 'short', 'normal', or 'long' partition
#SBATCH --time=00:10:00                    # max run time
#SBATCH --nodes=1                         # 1 node
#SBATCH --ntasks=16                        # n nores
#SBATCH --mem=128G                          # n GB RAM
#SBATCH --job-name=MIL_train              # Job name (no spaces)
#SBATCH --output=splits_%j.out
#SBATCH --error=splits_%j.err
#SBATCH --mail-type=ALL                   # Send email on begin, end, and fail
#SBATCH --mail-user=laurenvegas2027@u.northwestern.edu  # Your NU email

# EDIT (Lauren): seed is passed in at submit time: sbatch --export=ALL,SEED=<seed> split_script.sbatch
: "${SEED:?SEED not set. Submit with: sbatch --export=ALL,SEED=<seed> split_script.sbatch}"

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env   # you might want to copy the env and put it in your new project directory, if different from e32998

# Print system info
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Seed: $SEED"

# EDIT (Lauren): removed pip install lines so the job doesn't modify the shared env
export SSL_CERT_FILE="$(python -m certifi)"

cd /projects/e32998/STAT390_Fall2026/Team2_TheConsultancy/Lauren_Presentation1

# without loading checkpoints, trying 5 diff train / test splits

python3 make_splits.py \
    --labels_csv /projects/e32998/STAT390_Fall2026/Team2_TheConsultancy/Lauren_Presentation1/extra_benign_case_grade_match.csv \
    --patches_dir /projects/e32998/patches_benign_split \
    --seed "$SEED" \
    --save_dir /projects/e32998/STAT390_Fall2026/Team2_TheConsultancy/Lauren_Presentation1/splits_seed${SEED}
