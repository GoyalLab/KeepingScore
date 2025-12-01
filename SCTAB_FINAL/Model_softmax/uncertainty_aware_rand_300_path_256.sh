#!/bin/bash
#SBATCH --account=e31265
#SBATCH --partition=gengpu-long
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=168:00:00
#SBATCH --mem=64G
#SBATCH --job-name=uncertainty300
#SBATCH --output=logs/uncertainty_rand_300_T_1000_Path_256/slurm-%j.out
#SBATCH --error=logs/uncertainty_rand_300_T_1000_Path_256/slurm-%j.err

module purge all
module load mamba

export TF_ENABLE_ONEDNN_OPTS=0

# Run script
/projects/b1042/GoyalLab/jaekj/python/scTAB/bin/python -u uncertainty_aware_rand_300_path_256.py \
    --data_path "/projects/b1042/GoyalLab/jaekj/SCTAB_FINAL/Embedding" \
    --checkpoint "/projects/b1042/GoyalLab/jaekj/SCTAB_FINAL/diffusion_model/tb_logs/Vanilla Diffusion Model/version_0/checkpoints/epoch=285-step=2128412.ckpt" \
    --T 1000 \
    --n_paths 256 \
    --sample_size 300 \
    --save_dir "./Uncertainty_Rand_300_T_1000_Path_256" \
