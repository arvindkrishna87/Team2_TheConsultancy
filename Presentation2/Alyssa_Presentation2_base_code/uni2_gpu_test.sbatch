#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --job-name=uni2_gpu_test
#SBATCH --output=uni2_gpu_test.out
#SBATCH --error=uni2_gpu_test.err

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env

python3 -m pip install -q "timm>=1.0" huggingface_hub
python3 -m pip install -q --upgrade certifi
export SSL_CERT_FILE="$(python -m certifi)"

nvidia-smi
python3 -c "import torch; print('cuda available:', torch.cuda.is_available())"

cd /home/npg2887/Alyssa_Presentation2_base_code
python3 precompute_uni2.py --num_workers 2 --overwrite

# check one output file
python3 -c "import torch,glob; f=sorted(glob.glob('/home/npg2887/patch_embeddings_UNI2/*.pt'))[0]; v=torch.load(f); print(f, v.shape, v.dtype)"