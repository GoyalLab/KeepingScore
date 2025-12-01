#!/bin/bash
#SBATCH --account=e31265
#SBATCH --partition=gengpu-long
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=16
#SBATCH --time=100:00:00
#SBATCH --mem=32G 
#SBATCH --job-name="KS256"  
#SBATCH --output=logs_path256/slurm.out
#SBATCH --error=logs_path256/slurm.err

module purge all
module load mamba

export PYTHONNOUSERSITE=1
echo 'export PYTHONNOUSERSITE=1' >> ~/.bashrc


# Call Python directly from your env
/projects/b1042/GoyalLab/jaekj/python/DL_py3.10/bin/python -u KeepingSCORE_path256_long.py \
