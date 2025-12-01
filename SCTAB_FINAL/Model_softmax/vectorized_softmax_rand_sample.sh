#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics-gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --job-name=softmax_T=1000
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module purge all
module load mamba

export TF_ENABLE_ONEDNN_OPTS=0

# Run script
/projects/b1042/GoyalLab/jaekj/python/scTAB/bin/python -u vectorized_softmax_rand_sample_2.py \
    --data_path "/projects/b1042/GoyalLab/jaekj/SCTAB_FINAL/Embedding" \
    --checkpoint "/projects/b1042/GoyalLab/jaekj/SCTAB_FINAL/diffusion_model/tb_logs/Vanilla Diffusion Model/version_0/checkpoints/epoch=285-step=2128412.ckpt" \
    --n_paths 2 \
    --T 1000 \
    --sample_size 300 \
    --save_dir "./Sample_300_T_1000_Path_2" \

