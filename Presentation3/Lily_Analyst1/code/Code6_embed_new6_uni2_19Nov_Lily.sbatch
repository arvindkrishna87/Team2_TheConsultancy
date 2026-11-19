#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --job-name=embed_new6
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lilywu2027@u.northwestern.edu

# UNI2 native-size features for the 6 new cases, same settings as Team 5's features.
module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env
python3 -m pip install -r /projects/e32998/STAT390_Fall2026/Team6_InstructionalTeam/Code/Presentation3_base_code/requirements.txt
python3 -m pip install timm huggingface_hub

# Model access: EITHER a token in ~/.hf_token, OR a cached model folder on the HF_HOME line
[ -f ~/.hf_token ] && export HF_TOKEN=$(cat ~/.hf_token)
python3 -m pip install -q --upgrade certifi
export SSL_CERT_FILE="$(python -m certifi)"
# export HF_HOME=/path/to/cached/huggingface ; export HF_HUB_OFFLINE=1

BASE=/projects/e32998/STAT390_Fall2026/Team2_TheConsultancy/P3_analyst1
NEW6=$BASE/patches_new6
FEAT=$BASE/uni2_features_99

# Shortcuts to just the 6 new cases' patches, so only those get encoded
mkdir -p "$NEW6"
for c in 8 20 33 35 62 102; do
  find /projects/e32998/patch_images -maxdepth 1 -regextype posix-extended \
       -regex ".*/case_0*${c}_.*\.png" -exec ln -sf -t "$NEW6" {} +
done
echo "patches to encode: $(ls $NEW6 | wc -l)"   # expect 5372

cd /home/oky0687/Presentation3_base_code
python3 precompute_pooled_features.py \
    --encoder uni2 \
    --patches_dir "$NEW6" \
    --embeddings_dir "$FEAT" \
    --batch_size 64 --num_workers 6 --log_every 20

for c in 8 20 33 35 62 102; do echo "case $c features: $(ls $FEAT | grep -cE "^case_0*${c}_")"; done
echo "total features: $(ls $FEAT | wc -l)"
