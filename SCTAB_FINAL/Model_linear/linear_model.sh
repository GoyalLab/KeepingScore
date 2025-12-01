#!/bin/bash
#SBATCH --account=p32655
#SBATCH --partition=gengpu  ### PARTITION (buyin, short, normal, etc)
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1 ## how many cpus or processors do you need on each computer
#SBATCH --time=48:00:00 ## how long does this need to run (remember different partitions have restrictions on this parameter)
#SBATCH --mem=64G ## how much RAM do you need per node (this effects your FairShare score so be careful to not ask for more than you need))
#SBATCH --job-name="linear_model"  ## When you run squeue -u NETID this is how you can identify the job
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module purge all
module load mamba

# Call Python directly from your env
/projects/b1042/GoyalLab/jaekj/python/scTAB/bin/python -u linear_model.py



