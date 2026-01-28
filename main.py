import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Use nvidia gpu if found, otherwise use cpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
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

# Convert to PyTorch tensors and send to device
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# Auto-select batch size based on dataset size
N = len(X_train)
BATCH_SIZE = min(128, max(32, N // 200))
print("Training batch size:", BATCH_SIZE)


# Initialisation

class NOVA(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


model = NOVA().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

# Training config

USE_ENSEMBLE = True   # True = multiple runs + averaging | False = single run
NUM_RUNS = 5          # Number of independent runs if ensemble is enabled

N = len(X_train)
patience = 20
max_epochs = 1000

all_preds = []


# Run the training with multiple models

def train_single_model():
    model = NOVA().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    best_loss = float('inf')
    counter = 0

    for epoch in range(max_epochs):
        perm = torch.randperm(N)
        epoch_loss = 0

        for i in range(0, N, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            X_batch = X_train_tensor[idx]
            y_batch = y_train_tensor[idx]

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(idx)

        epoch_loss /= N

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


# Run the training with multiple models

if USE_ENSEMBLE:
    print(f"\nRunning ensemble training with {NUM_RUNS} models...\n")

    for run in range(NUM_RUNS):
        print(f"Model {run+1}/{NUM_RUNS}")
        model = train_single_model()

        model.eval()
        with torch.no_grad():
            preds = model(X_test_tensor).cpu().numpy()
            all_preds.append(preds)

    avg_preds_scaled = np.mean(all_preds, axis=0)

else:
    print("\nRunning single model training...\n")

    model = train_single_model()

    model.eval()
    with torch.no_grad():
        avg_preds_scaled = model(X_test_tensor).cpu().numpy()


# Testing

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
    perc_err = 100 * abs(p - a) / a if a != 0 else 0.0
    print(f"{i+1:>6} | {p:9.2f} | {a:6.2f} | {perc_err:7.1f}%")