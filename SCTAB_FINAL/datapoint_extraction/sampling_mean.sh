#!/bin/bash
#SBATCH --account=e31265
#SBATCH --partition=normal 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --job-name=samlping_mean
#SBATCH --output=logs/sampling_mean_%j.out
#SBATCH --error=logs/sampling_mean_%j.err
#SBATCH --mem=32G
#SBATCH --time=10:00:00

# Load modules
module load mamba

# Make log directory
mkdir -p logs

# Run Python script
/projects/b1042/GoyalLab/jaekj/python/scTAB_new/bin/python sampling_mean.py \
    --data_path /projects/b1042/GoyalLab/jaekj/KeepingScore/SCTAB_FINAL/Embedding \
    --save_path /projects/b1042/GoyalLab/jaekj/KeepingScore/SCTAB_FINAL/datapoint_extraction/sample_mean/mean.npz \
    --mapping_path /projects/b1042/GoyalLab/jaekj/KeepingScore/merlin_cxg_2023_05_15_sf-log1p/categorical_lookup/cell_type.parquet