#!/bin/bash
#SBATCH --account=e31265
#SBATCH --partition=long  
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16 
#SBATCH --time=168:00:00 
#SBATCH --mem=256G 
#SBATCH --job-name="embedding_tsne" 
#SBATCH --output=tSNE/slurm-%j.out
#SBATCH --error=tSNE/slurm-%j.err

module purge all
module load mamba
mamba activate scTAB

# Call Python directly from your env
/projects/b1042/GoyalLab/jaekj/python/scTAB/bin/python -u embedding_visualization.py \


