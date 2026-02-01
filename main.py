import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# TODO
# Add interface trough python library for easy use
# Use the exported model with the python library
# Also speed up the GPU training since it's eating my cpu atm.....
# FIX BUG NOT USING CUDA SO NOT USING GPU

# Use nvidia gpu if found, otherwise use cpu (please use a gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# Load dataset (replace with your own)
data = fetch_california_housing()
X = data.data
y = data.target.reshape(-1, 1)

print("Dataset shape:", X.shape, y.shape)

# Scale input features
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Scale target for better training
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors and send to device (GPU is faster)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# Save scalers
with open("scaler_X.pkl", "wb") as f:
    pickle.dump(scaler_X, f)
with open("scaler_y.pkl", "wb") as f:
    pickle.dump(scaler_y, f)

# Auto-select batch size based on dataset size
N = len(X_train)
BATCH_SIZE = 2048  # try 4096 if VRAM allows
print("Training batch size:", BATCH_SIZE)


# Model definition (slightly wider = more GPU math per step)
class NOVA(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)


# Loss
criterion = nn.MSELoss()

# Training config
USE_ENSEMBLE = True   # True = multiple runs + averaging | False = single run
NUM_RUNS = 5          # Number of independent runs if ensemble is enabled

patience = 20
max_epochs = 1000

all_preds = []
all_state_dicts = []   # <-- this will store model weights

# Train one model
def train_single_model():
    model = NOVA().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    best_loss = float("inf")
    counter = 0

    for epoch in range(max_epochs):
        perm = torch.randperm(N)
        epoch_loss = 0.0

        for i in range(0, N, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            X_batch = X_train_tensor[idx]
            y_batch = y_train_tensor[idx]

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(idx)

        epoch_loss /= N

        # early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 50 == 0:
            print(f"Epoch {epoch} | Loss: {epoch_loss:.4f}")

    return model

# Run training
if USE_ENSEMBLE:
    print(f"\nRunning ensemble training with {NUM_RUNS} models...\n")

    for run in range(NUM_RUNS):
        print(f"Model {run + 1}/{NUM_RUNS}")
        model = train_single_model()

        # save weights in memory
        all_state_dicts.append(model.state_dict())

        model.eval()
        with torch.no_grad():
            preds = model(X_test_tensor).cpu().numpy()
            all_preds.append(preds)

    avg_preds_scaled = np.mean(all_preds, axis=0)

else:
    print("\nRunning single model training...\n")

    model = train_single_model()
    all_state_dicts.append(model.state_dict())

    model.eval()
    with torch.no_grad():
        avg_preds_scaled = model(X_test_tensor).cpu().numpy()

# Evaluation
y_pred = scaler_y.inverse_transform(avg_preds_scaled)
y_true = scaler_y.inverse_transform(y_test_tensor.cpu().numpy())

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)

print(f"\nFinal Test Metrics:")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")

print("\nSample | Predicted | Actual | % Error")
print("-" * 40)
for i in range(5):
    p = y_pred[i][0]
    a = y_true[i][0]
    err = 100 * abs(p - a) / a if a != 0 else 0.0
    print(f"{i+1:>6} | {p:9.2f} | {a:6.2f} | {err:7.1f}%")

# Export all networks to a single file

export_bundle = {
    "model_class": "NOVA",
    "num_models": len(all_state_dicts),
    "state_dicts": all_state_dicts,
    "scaler_X": scaler_X,
    "scaler_y": scaler_y
}

# Save everything into a single .pt file
torch.save(export_bundle, "nova_ensemble.pt")

print(f"\nEnsemble exported successfully!")
print(f"Number of models: {len(all_state_dicts)}")
print("Saved to: nova_ensemble.pt")