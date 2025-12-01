#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics-himem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=48:00:00
#SBATCH --mem=1000G
#SBATCH --job-name="TestEmbeddings"
#SBATCH --output=slurm-test-%j.out
#SBATCH --error=slurm-test-%j.err

module purge all
module load mamba

# Run script
/projects/b1042/GoyalLab/jaekj/python/scTAB/bin/python -u test_emb.py