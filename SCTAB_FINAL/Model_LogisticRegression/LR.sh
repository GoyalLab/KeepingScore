#!/bin/bash
#SBATCH --account=p32655
#SBATCH --partition=gengpu  
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1 
#SBATCH --time=48:00:00 
#SBATCH --mem=64G 
#SBATCH --job-name="LogisticR-gpu" 
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module purge all
module load mamba

# Call Python directly from your env
/projects/b1042/GoyalLab/jaekj/python/scTAB/bin/python -u LR.py
