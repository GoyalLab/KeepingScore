#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics-himem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=196:00:00
#SBATCH --mem=1000G
#SBATCH --job-name="TrainEmbeddings"
#SBATCH --output=slurm-train-%j.out
#SBATCH --error=slurm-train-%j.err

module purge all
module load mamba

# Run script
/projects/b1042/GoyalLab/jaekj/python/scTAB/bin/python -u train_emb.py
