#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics-gpu 
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1 
#SBATCH --job-name=vanilla_diffusion
#SBATCH --output=logs/diffusion_%j.out
#SBATCH --error=logs/diffusion_%j.err
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# Load modules
module load mamba

# Make log directory
mkdir -p logs

# Run Python script
/projects/b1042/GoyalLab/jaekj/python/scTAB/bin/python diffusion_model1.py \
    --emb_pred_path /projects/b1042/GoyalLab/jaekj/SCTAB_FINAL/Embedding \
    --epochs 500 \
    --batch_size 2048 \
    --n_devices 1 \
    --n_workers 1 \
    --state_dir /projects/b1042/GoyalLab/jaekj/SCTAB_FINAL/diffusion_model/state \
    --umap_save_dir /projects/b1042/GoyalLab/jaekj/SCTAB_FINAL/diffusion_model/umap_plots
