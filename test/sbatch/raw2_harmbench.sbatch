#!/bin/bash

#SBATCH -p nvidia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=71:59:59
#SBATCH --mem=10GB

### Define the parameter r (4,8,16,32,64,128)
r=128
#SBATCH --job-name=r$r_raw

module purge

# Load the Conda module
source ~/.bashrc

# Activate your Conda environment
conda activate recurrent

# Set the transformers cache path
export TRANSFORMERS_CACHE='/scratch/bc3194/huggingface_cache'
export HF_HOME='/scratch/bc3194/huggingface_cache'

# Call python script
cd ..
python -u hb_raw2_gen.py --r=$r
