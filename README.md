# KeepingScore
[![Paper](https://img.shields.io/badge/Paper-bioRxiv-brightgreen)](https://www.biorxiv.org/content/10.1101/2025.11.26.690838v1)

Code for 'Interpretable Thermodynamic Score-based Classification of Relaxation Excursions'

Scripts needed to run the examples in the 'Interpretable Thermodynamic Score-based Classification of Relaxation Excursions' are as follows. Note that 1., 2. and 5. can be run on a PC, though 3. and 4. likely require an HPC. 

1. MNIST: VAE_MNIST.ipynb is a jupyter notebook to create VAE embeddings from MNIST data. MNIST_final.ipynb is a notebook with diffusion model training and classification as well as all other MNIST analysis (see Fig. 3).

2. CIFAR10: CIFAR10.ipynb is a notebook with CIFAR10 dataloading, joint VAE+diffusion modeltraining, classification and analysis for the CIFAR10 example (see Fig. 3).

3. Cell type (SCTAB_FINAL folder): this folder lists the python scripts used for the celltype analysis. linear_model: linear model, LR: logistic regression model, MLP: multi-layer Perceptron, rand_300_path_4.py: Keeping SCORE model,            TabNet.py, tabnet.sh: tabnet model, XGBoost: XGBoost model.

4. Genetic perturbation (perturb-seq folder): 
                        0_original_data_inspection: this file inspects the original latent space – latent.h5ad
                        1_train_val_test_split: train/validation/test split
                        2_model: our diffusion model used for the perturb-seq anlaysis
                        3_classification: our KeepingSCORE classifier in jupyter version
                        KeepingSCORE_path4.py: keeping score analysis file used for the benchmark with path=4
                        Lr_model: logistic regresswion
                        Model_XGB: XGBoost model
                        Model_mlp: MLP model

5. Protein stability: protein.ipynb is a notebook with protein stability data loading, joint VAE+diffusion regression model training and all other analysis regarding Fig. 5.

## Requirements for the cell type & Perturb-seq analysis
- **Cell Type Analysis:** Python 3.8  
- **Perturb-seq Analysis:** Python 3.10+  
- **Packages:** See environment `.yml` files and instructions in each analysis folder  
- **Hardware:** GPU recommended for both Cell Type and Perturb-seq analyses


## Citation

If you use this code, please cite:

> **Combinatorial prediction of therapeutic perturbations using causally-inspired neural networks**  
> bioRxiv, 2025  
> https://www.biorxiv.org/content/10.1101/2025.11.26.690838v1

Please contact Ben Kuznets-Speck (biophysben@gmail.com) with any questions. 
