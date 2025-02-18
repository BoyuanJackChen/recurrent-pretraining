#!/bin/bash

#SBATCH -p nvidia
#SBATCH -C 80g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3:59:59
#SBATCH --mem=10GB

### Define the parameter r (4,8,16,32,64,128)
r=128
#SBATCH --job-name=r$r_wordgame

module purge

# Load the Conda module
source ~/.bashrc

# Activate your Conda environment
conda activate recurrent

# Set the transformers cache path
export TRANSFORMERS_CACHE='/home/bc3194/Desktop/huggingface_cache'
export HF_HOME='/home/bc3194/Desktop/huggingface_cache'

# Call python script
cd ..
python -u hb_wordgame_gen.py --r=$r
