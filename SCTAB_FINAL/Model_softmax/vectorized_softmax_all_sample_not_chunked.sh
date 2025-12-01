#!/bin/bash
#SBATCH --account=e31265
#SBATCH --partition=gengpu-long
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=120:00:00
#SBATCH --mem=64G
#SBATCH --job-name=Softmax_all_sample
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module purge all
module load mamba

export TF_ENABLE_ONEDNN_OPTS=0

# Run script
/projects/b1042/GoyalLab/jaekj/python/scTAB/bin/python -u vectorized_softmax_all_sample_not_chunked.py \
    --data_path "/projects/b1042/GoyalLab/jaekj/SCTAB_FINAL/Embedding" \
    --checkpoint "/projects/b1042/GoyalLab/jaekj/SCTAB_FINAL/diffusion_model/tb_logs/Vanilla Diffusion Model/version_0/checkpoints/epoch=285-step=2128412.ckpt" \
    --n_paths 2 \
    --T 1000 \
    --batch_size 64 \
    --save_dir "./Sample_all_T_1000_Path_2_not_chunked" \
