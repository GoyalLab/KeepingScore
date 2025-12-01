#!/bin/bash
#SBATCH --account=e31265
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=5:00:00
#SBATCH --mem=64G
#SBATCH --job-name=T500
#SBATCH --output=logs/T=500/slurm-%j.out
#SBATCH --error=logs/T=500/slurm-%j.err

module purge all
module load mamba

export TF_ENABLE_ONEDNN_OPTS=0

# Run script
/projects/b1042/GoyalLab/jaekj/python/scTAB/bin/python -u uncertainty_aware_mean.py \
    --data_path "/projects/b1042/GoyalLab/jaekj/SCTAB_FINAL/Embedding" \
    --checkpoint "/projects/b1042/GoyalLab/jaekj/SCTAB_FINAL/diffusion_model/tb_logs/Vanilla Diffusion Model/version_0/checkpoints/epoch=285-step=2128412.ckpt" \
    --T_train 1000 \
    --T_eval 500 \
    --n_paths 256 \
    --save_dir "./T=500" \

