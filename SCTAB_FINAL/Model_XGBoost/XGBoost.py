import os
from os.path import join

import torch
import xgboost as xgb
import numpy as np
import dask.dataframe as dd

device = "cuda" if torch.cuda.is_available() else "cpu"

print("[Info] Dataset loading...")
DATA_PATH = '/projects/b1042/GoyalLab/jaekj/SCTAB_FINAL/Embedding'

train_emb = torch.load(join(DATA_PATH, "train_embedding.pt"), weights_only=True)
val_emb = torch.load(join(DATA_PATH, "val_embedding.pt"), weights_only=True)
test_emb = torch.load(join(DATA_PATH, "test_embedding.pt"), weights_only=True)

# Convert to numpy
print("[Info] NumPy conversion...")
X_train = train_emb["X"].cpu().numpy()  
y_train = train_emb["y_true"].squeeze().cpu().numpy().astype(int)

X_val = val_emb["X"].numpy()
y_val = val_emb["y_true"].squeeze().cpu().numpy().astype(int)

X_test = test_emb["X"].numpy()
y_test = test_emb["y_true"].squeeze().cpu().numpy().astype(int)

unique_test_labels = torch.unique(torch.tensor(y_test))

label_map = {v.item(): i for i, v in enumerate(unique_test_labels)}

y_test_idx = torch.tensor([label_map[int(y)] for y in y_test], device=device)

num_classes = len(unique_test_labels)
label_ids = torch.arange(num_classes, device=device)

print("[Info] Class weights loading...")
DATA_PATH = '/projects/b1042/GoyalLab/jaekj/merlin_cxg_2023_05_15_sf-log1p'
class_weights = np.load(join(DATA_PATH, 'class_weights.npy'))
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
weights = np.array([class_weights_dict[label] for label in y_train])

clf = xgb.XGBClassifier(
    tree_method='gpu_hist',
    gpu_id=0,
    n_estimators=1000,
    eta=0.075,
    subsample=0.75,
    max_depth=10,
    n_jobs=20,
    early_stopping_rounds=10
)

# --- Train ---
print("[Info] Start training")
clf.fit(
    X_train, y_train,
    sample_weight=weights,
    eval_set=[(X_val, y_val)]
)

SAVE_PATH = "/projects/b1042/GoyalLab/jaekj/SCTAB_FINAL/Model_XGBoost/run1"
os.makedirs(SAVE_PATH, exist_ok=True)

# --- Save model ---
clf.save_model(join(SAVE_PATH, 'xgb_embedding_model.json'))
print("[Info] Saved trained model")

# --- Optional: predict on test set ---
y_pred_test = clf.predict(X_test)
print("[Info] Prediction complete")


