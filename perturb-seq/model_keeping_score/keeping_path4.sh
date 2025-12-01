#!/bin/bash
#SBATCH --account=e31265
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=16
#SBATCH --time=10:00:00
#SBATCH --mem=32G 
#SBATCH --job-name="KS4"  
#SBATCH --output=logs_path4/slurm.out
#SBATCH --error=logs_path4/slurm.err

module purge all
module load mamba

export PYTHONNOUSERSITE=1
echo 'export PYTHONNOUSERSITE=1' >> ~/.bashrc


# Call Python directly from your env
/projects/b1042/GoyalLab/jaekj/python/DL_py3.10/bin/python -u KeepingSCORE_path4.py \
