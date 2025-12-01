#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=32
#SBATCH --time=48:00:00
#SBATCH --mem=64G 
#SBATCH --job-name="mlp"  
#SBATCH --output=logs/mlp_slurm.out
#SBATCH --error=logs/mlp_slurm.err

module purge all
module load mamba

export PYTHONNOUSERSITE=1
echo 'export PYTHONNOUSERSITE=1' >> ~/.bashrc

# Call Python directly from your env
/projects/b1042/GoyalLab/jaekj/python/DL_py3.10/bin/python -u model_mlp.py \
