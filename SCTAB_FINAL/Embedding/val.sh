#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics-himem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=48:00:00
#SBATCH --mem=1000G
#SBATCH --job-name="ValEmbeddings"
#SBATCH --output=slurm-val-%j.out
#SBATCH --error=slurm-val-%j.err

module purge all
module load mamba

# Run script
/projects/b1042/GoyalLab/jaekj/python/scTAB/bin/python -u val_emb.py
