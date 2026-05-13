#!/bin/bash
#SBATCH --account=e31265
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=3:00:00
#SBATCH --mem=32G 
#SBATCH --job-name="KS10000"  
#SBATCH --output=logs_path10000/slurm.out
#SBATCH --error=logs_path10000/slurm.err

module purge all
module load mamba
module load cuda/12.1.0-gcc-11.2.0

export PYTHONNOUSERSITE=1

SCRIPT_DIR="/projects/b1042/GoyalLab/jaekj/KeepingScore/perturb-seq/model_keeping_score"
cd "${SCRIPT_DIR}" || exit 1

echo "[INFO] Host: $(hostname)"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[INFO] which python: /projects/b1042/GoyalLab/jaekj/python/KS_perturb/bin/python"
nvidia-smi || { echo "[ERROR] nvidia-smi failed; no GPU visible in job environment."; exit 1; }
/projects/b1042/GoyalLab/jaekj/python/KS_perturb/bin/python -c 'import torch; print("[INFO] torch", torch.__version__); print("[INFO] cuda_built", torch.version.cuda); print("[INFO] cuda_available", torch.cuda.is_available()); print("[INFO] device_count", torch.cuda.device_count()); assert torch.cuda.is_available(), "CUDA is not available"; print("[INFO] device_name", torch.cuda.get_device_name(0))' || exit 1

# Call Python directly from your env
/projects/b1042/GoyalLab/jaekj/python/KS_perturb/bin/python -u KeepingSCORE_path100.py \
                                                    --checkpoint_path "/projects/b1042/GoyalLab/jaekj/KeepingScore/perturb-seq/train_diffusion_model/Two_stage_FiLM_Diffusion_checkpoint_Sigmoid_orig.pth" \
                                                    --save_dir "${SCRIPT_DIR}/results_path10000" \
                                                    --n_paths 10000 \
                                                    --mean_path "/projects/b1042/GoyalLab/jaekj/KeepingScore/perturb-seq/datapoint_extraction/sample_mean/mean.npz" \
                                                    --z_test_path "/projects/b1042/GoyalLab/jaekj/KeepingScore/perturb-seq/ExPert/anndata/splits/z_test.npy" \
                                                    --y_test_path "/projects/b1042/GoyalLab/jaekj/KeepingScore/perturb-seq/ExPert/anndata/splits/y_test.npy"
