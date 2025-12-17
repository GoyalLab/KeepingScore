# Keeping SCORE: scRNA-seq Cell Type Analysis
## Overview
<p align="center">
  <img src="celltype_figure.png" alt="celltype" width="300">
</p>

## Requirements 
tSNE plot for the visualization of the test embedding was executed on the high-memory RAM (1,000 GB). 
All GPU-dependent components were executed on NVIDIA A100 GPUs.

## Environment Setup
To reproduce the Perturb-seq analysis environment, follow the steps below.

### 1. Create the Conda/Mamba environment
```bash
mamba env create --prefix $TARGET_DIR/KS_celltype --file scTAB_environment_fixed.yml
```
```bash
echo $CONDA_PREFIX
```
### 2. Activate Mamba environment 
```bash
mamba activate $TARGET_DIR/KS_celltype
```
### 3. Install Jupyter kernel support
```bash
mamba install ipykernel -y
```
```bash
python -m ipykernel install --user \
    --name KS_celltype \
    --display-name "KS_celltype"
```
## Data structure
-
-
- 
- 
