# Perturb-seq Analysis
## Overview 
This directory contains the full pipeline for **Perturb-seq analysis**, including diffusion-based representation learning, Keeping SCORE–based classification, and benchmark classifiers (MLP, XGBoost, and logistic regression).

## Requirements
All GPU-dependent components were executed on **NVIDIA A100 GPUs**.

## Environment Setup Guideline
Please run the following commands:
[bash] mamba env create -f DL_py3.10_repro.yml --prefix $target_directory/KS_perturb$
[bash] mamba activate $target_directory/KS_perturb$
[bash] mamba install ipykernel -y
[bash] python -m ipykernel install --prefix=$HOME/.local --name KS_perturb --display-name "KS_perturb"

## Data structure
- `model_MLP`: This folder contains the training details of the multi-layer perceptron for benchmark.
- `model_XGB`: This folder contains the training details of the XGBoost model for benchmark.
- `model_keeping_score`: This folder contains our Keeping SCORE model training and prediction details
- `model_logistic_regression`: This folder contains the training details about the logistic regression model. 
- `0_original_data_inspection.ipynb`: the file to inspect the original dataset - `latent.h5ad` file. 
- `1_train_val_test_split.ipynb`: the file to obtain the training, validation, and test split. This file produces `_train.npy`, `_test.npy`, `_val.npy` files. 
- `2_model.ipynb`: the file to train a diffusion model. 
- `3_Classification.ipynb`: the file to perform keepingSCORE-based classification. However, it is recommended to run KeepingSCORE on HPC setting with `Uncertainty_path_4.py` in `model_keeping_score` folder.
- `model.py`: the python file of the model.
- `latent.h5ad`: The original latent space embedding used for this analysis.

* Please run the 0-3 in order.
* Running the diffusion model (`2_model.ipynb`) should precede Keeping SCORE analysis.
