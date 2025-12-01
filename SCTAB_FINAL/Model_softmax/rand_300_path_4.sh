#!/bin/bash
#SBATCH --account=e31265
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --job-name=random300
#SBATCH --output=logs/rand_300_T_1000_Path_2/slurm-%j.out
#SBATCH --error=logs/rand_300_T_1000_Path_2/slurm-%j.err

module purge all
module load mamba

export TF_ENABLE_ONEDNN_OPTS=0

# Run script
/projects/b1042/GoyalLab/jaekj/python/scTAB/bin/python -u rand_300_path_4.py \
    --data_path "/projects/b1042/GoyalLab/jaekj/SCTAB_FINAL/Embedding" \
    --checkpoint "/projects/b1042/GoyalLab/jaekj/SCTAB_FINAL/diffusion_model/tb_logs/Vanilla Diffusion Model/version_0/checkpoints/epoch=285-step=2128412.ckpt" \
    --T 1000 \
    --n_paths 4 \
    --sample_size 300 \
    --save_dir "./Rand_300_T_1000_Path_4" \
