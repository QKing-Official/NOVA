import torch
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

BUNDLE_PATH = "nova_ensemble.pt"
USE_TEST_SET = True   # True = use original test set, False = use CSV
NEW_DATA_CSV = "new_samples.csv"  # Only used if USE_TEST_SET=False

# Check for GPU or CPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load the model cluster

bundle = torch.load(BUNDLE_PATH, map_location=device)
all_state_dicts = bundle["state_dicts"]
scaler_X = bundle["scaler_X"]
scaler_y = bundle["scaler_y"]
num_models = bundle["num_models"]

print(f"Loaded ensemble with {num_models} models.")

# Initialise parameters

import torch.nn as nn

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

# Testing

if USE_TEST_SET:
    data = fetch_california_housing()
    X = data.data
    y = data.target.reshape(-1, 1)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    import pandas as pd
    df = pd.read_csv(NEW_DATA_CSV)
    X_test = df.values
    y_test = None

X_test_scaled = scaler_X.transform(X_test)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

ensemble = []
for sd in all_state_dicts:
    model = NOVA().to(device)
    model.load_state_dict(sd)
    model.eval()
    ensemble.append(model)

all_preds = []
for model in ensemble:
    with torch.no_grad():
        preds = model(X_test_tensor).cpu().numpy()
        all_preds.append(preds)

avg_preds_scaled = np.mean(all_preds, axis=0)
y_pred = scaler_y.inverse_transform(avg_preds_scaled)

if y_test is not None:
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\nTest Metrics: MSE = {mse:.4f}, MAE = {mae:.4f}")

print("\nSample | Predicted | Actual | % Error")
print("-"*40)
num_samples = min(5, len(y_pred))
for i in range(num_samples):
    p = y_pred[i][0]
    a = y_test[i][0] if y_test is not None else float('nan')
    perc_err = 100 * abs(p - a) / a if (y_test is not None and a != 0) else 0.0
    print(f"{i+1:>6} | {p:9.2f} | {a:6.2f} | {perc_err:7.1f}%")
