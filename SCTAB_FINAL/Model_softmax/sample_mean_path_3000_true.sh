#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=3:00:00
#SBATCH --mem=64G
#SBATCH --job-name=sample_mean_path_3000_true
#SBATCH --output=logs/sample_mean_path_3000_true/slurm-%j.out
#SBATCH --error=logs/sample_mean_path_3000_true/slurm-%j.err

module purge all
module load mamba

export TF_ENABLE_ONEDNN_OPTS=0
module load cuda/12.1.0-gcc-11.2.0

# Run script
/projects/b1042/GoyalLab/jaekj/python/scTAB_new/bin/python -u sample_mean_path_100_true_version3.py \
    --data_path "/projects/b1042/GoyalLab/jaekj/KeepingScore/SCTAB_FINAL/Embedding/" \
    --checkpoint "/projects/b1042/GoyalLab/jaekj/KeepingScore/SCTAB_FINAL/diffusion_model/tb_logs/Vanilla Diffusion Model/version_0/checkpoints/epoch=285-step=2128412.ckpt" \
    --mean_path "/projects/b1042/GoyalLab/jaekj/KeepingScore/SCTAB_FINAL/datapoint_extraction/sample_mean/mean.npz" \
    --T 1000 \
    --n_paths 3000 \
    --save_dir "./sample_mean_path_3000_true" \
