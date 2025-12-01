import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import os

# =====================================================
# === Load latent data and labels ======================
# =====================================================
# === Load data ===
data_dir = "/projects/b1042/GoyalLab/jaekj/perturb-seq"
z_train = np.load(os.path.join(data_dir, "z_train.npy"))
z_val = np.load(os.path.join(data_dir, "z_val.npy"))
z_test = np.load(os.path.join(data_dir, "z_test.npy")) 

y_train = np.load(os.path.join(data_dir, "y_train.npy")).astype(int)
y_val = np.load(os.path.join(data_dir, "y_val.npy")).astype(int)
y_test = np.load(os.path.join(data_dir, "y_test.npy")).astype(int)

# =====================================================
# === Order test data by class (for consistency) ======
# =====================================================
unique_classes = np.sort(np.unique(y_test))
z_test_ordered, y_test_ordered = [], []

for cls in unique_classes:
    idx = np.where(y_test == cls)[0]
    print(f"Class {cls}: {len(idx)} samples")
    z_test_ordered.append(z_test[idx])
    y_test_ordered.append(y_test[idx])

z_test_ordered = np.vstack(z_test_ordered)
y_test_ordered = np.concatenate(y_test_ordered)

# =====================================================
# === Convert to torch tensors =========================
# =====================================================
X_train = torch.FloatTensor(z_train)
y_train_torch = torch.LongTensor(y_train)

X_val = torch.FloatTensor(z_val)
y_val_torch = torch.LongTensor(y_val)

X_test = torch.FloatTensor(z_test_ordered)
y_test_torch = torch.LongTensor(y_test_ordered)

# =====================================================
# === DataLoaders =====================================
# =====================================================
batch_size = 128
train_loader = DataLoader(TensorDataset(X_train, y_train_torch), batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val,   y_val_torch),   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test,  y_test_torch),  batch_size=batch_size, shuffle=False)

# =====================================================
# === MLP model =======================================
# =====================================================
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# =====================================================
# === Setup ===========================================
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

input_dim = z_train.shape[1]
num_classes = len(np.unique(y_train))
hidden_dims = [512, 256, 128]

model = SimpleMLP(input_dim, hidden_dims, num_classes, dropout=0.3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                 factor=0.5, patience=5, verbose=True)

# =====================================================
# === Training ========================================
# =====================================================
print("\n[Info] Training MLP classifier...")
epochs = 100
best_val_loss = float("inf")
patience = 15
patience_counter = 0

for epoch in range(epochs):
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0

    for Xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()

        outputs = model(Xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, pred = outputs.max(1)
        train_total += yb.size(0)
        train_correct += pred.eq(yb).sum().item()

    avg_train_loss = train_loss / len(train_loader)
    train_acc = 100 * train_correct / train_total

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0

    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)

            outputs = model(Xb)
            loss = criterion(outputs, yb)

            val_loss += loss.item()
            _, pred = outputs.max(1)
            val_total += yb.size(0)
            val_correct += pred.eq(yb).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total

    print(f"Epoch {epoch+1}: "
          f"Train Loss {avg_train_loss:.4f}, Train Acc {train_acc:.2f}% | "
          f"Val Loss {avg_val_loss:.4f}, Val Acc {val_acc:.2f}%")

    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "mlp_best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping!")
            break

# =====================================================
# === Load best model & evaluate ======================
# =====================================================
model.load_state_dict(torch.load("mlp_best_model.pt"))
model.eval()

all_preds = []
all_posteriors = []

with torch.no_grad():
    for Xb, _ in test_loader:
        Xb = Xb.to(device)
        logits = model(Xb)
        probs = torch.softmax(logits, dim=1)

        all_posteriors.append(probs.cpu().numpy())
        all_preds.append(logits.argmax(1).cpu().numpy())

posteriors_mlp = np.vstack(all_posteriors)
y_pred_mlp = np.concatenate(all_preds)

print("\n=== MLP Test Performance ===")
print("Accuracy:", accuracy_score(y_test_ordered, y_pred_mlp))
print("Macro-F1:", f1_score(y_test_ordered, y_pred_mlp, average='macro'))

# =====================================================
# === Save results ====================================
# =====================================================
save_dir = "./"
os.makedirs(save_dir, exist_ok=True)

torch.save(model.state_dict(), f"{save_dir}/mlp_classifier.pt")
np.save(f"{save_dir}/y_pred_mlp.npy", y_pred_mlp)
np.save(f"{save_dir}/posteriors_mlp.npy", posteriors_mlp)

with open(f"{save_dir}/model_config.pkl", "wb") as f:
    pickle.dump({
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "num_classes": num_classes,
        "dropout": 0.3
    }, f)

print(f"\n[Info] Saved MLP predictions + posteriors to {save_dir}")
