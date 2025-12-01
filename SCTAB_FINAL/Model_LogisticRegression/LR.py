import os
from os.path import join

import torch
from torch.utils.data import TensorDataset, DataLoader
import scanpy as sc
import numpy as np
import pandas as pd
import anndata
import dask.dataframe as dd
import dask.array as da
import pickle

from scipy.sparse import csr_matrix

print("[Info] Dataset loading...")
DATA_PATH = '/projects/b1042/GoyalLab/jaekj/SCTAB_FINAL/Embedding'

train_emb = torch.load(join(DATA_PATH, "train_embedding.pt"), weights_only=True)
val_emb = torch.load(join(DATA_PATH, "val_embedding.pt"), weights_only=True)


print("[Info] NumPy conversion...")
X_train = train_emb["X"].cpu().numpy()  
y_train = train_emb["y_true"].squeeze().cpu().numpy().astype(int)

X_val = val_emb["X"].numpy()
y_val = val_emb["y_true"].squeeze().cpu().numpy().astype(int)

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

clf = SGDClassifier(
    loss="log",          # logistic regression
    penalty="l2",             # same as CellTypist (ridge regularization)
    alpha=1e-4,               # regularization strength (tune if needed)
    max_iter=1000,            # max number of passes over data
    tol=1e-3,                 # stop when convergence is good enough
    random_state=1
)

print("[Info] Training logistic regression classifier...")
clf.fit(X_train, y_train)

save_dir = '/projects/b1042/GoyalLab/jaekj/SCTAB_FINAL/emb_tb_logs/LogisticRegression/run1/'
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, 'logistic_regression_clf.pkl') 

with open(save_path, 'wb') as f:
    pickle.dump(clf, f)
