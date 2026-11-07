#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --job-name=uni2_test
#SBATCH --output=uni2_test.out
#SBATCH --error=uni2_test.err

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env

python3 -m pip install -q "timm>=1.0" huggingface_hub
python3 -m pip install -q --upgrade certifi
export SSL_CERT_FILE="$(python -m certifi)"

# download UNI2 weights once (skipped if already downloaded)
if [ ! -f /home/npg2887/uni2_weights/pytorch_model.bin ]; then
  python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('MahmoodLab/UNI2-h', 'pytorch_model.bin', local_dir='/home/npg2887/uni2_weights')"
fi

cd /home/npg2887/Alyssa_Presentation2_base_code
python3 precompute_uni2.py --num_workers 2

# check one output file
python3 -c "import torch,glob; f=sorted(glob.glob('/home/npg2887/patch_embeddings_UNI2/*.pt'))[0]; v=torch.load(f); print(f, v.shape, v.dtype)"