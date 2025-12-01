#!/bin/bash
#SBATCH --account=e31265
#SBATCH --partition=genhimem  
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16 
#SBATCH --time=48:00:00 
#SBATCH --mem=512G 
#SBATCH --job-name="embedding_tsne" 
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module purge all
module load mamba
mamba activate scTAB

# Call Python directly from your env
/projects/b1042/GoyalLab/jaekj/python/scTAB/bin/python -u embedding_visualization_subsampled.py \


