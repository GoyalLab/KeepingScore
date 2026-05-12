#!/bin/bash
#SBATCH --account=e31265
#SBATCH --partition=gengpu 
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1 
#SBATCH --job-name=diffusion
#SBATCH --output=logs/model_updated/diffusion_%j.out
#SBATCH --error=logs/model_updated/diffusion_%j.err
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# Load modules
module load mamba

# Make log directory
mkdir -p logs

# Run Python script
/projects/b1042/GoyalLab/jaekj/python/KS_perturb/bin/python model_updated.py 