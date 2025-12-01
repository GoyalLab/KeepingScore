#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=diffusion
#SBATCH --output=logs/diffusion_ver3_%j.out
#SBATCH --error=logs/diffusion_ver3_%j.err
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# Load modules
module load mamba

# Make log directory
mkdir -p logs

# Run Python script (PURE PYTORCH version)
/projects/b1042/GoyalLab/jaekj/python/scTAB/bin/python diffusion_model3.py \
    --emb_pred_path /projects/b1042/GoyalLab/jaekj/SCTAB_FINAL/Embedding \
    --epochs 500 \
    --batch 2048 \
    --lr 0.0003 \
    --T 1000
