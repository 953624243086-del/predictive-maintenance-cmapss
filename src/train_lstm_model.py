"""
C-MAPSS FD001 LSTM Model — Week 3
===================================
Trains an LSTM regressor on the raw windowed sequences (not flattened,
unlike Week 2's baseline) to predict Remaining Useful Life (RUL).

LSTMs are built to learn temporal patterns directly from sequences, so
this script uses X_train/X_test exactly as Week 1 produced them:
(n_samples, window_size, n_features) — no flattening, no engineered
mean/std/trend features.

Evaluates with the same two metrics as Week 2, and automatically compares
against models/baseline_results.json (from Week 2) so you can see at a
glance whether the LSTM improved on XGBoost/Random Forest.

Usage:
    python train_lstm_model.py --data_dir ./processed --baseline_path ./models/baseline_results.json

Requires: torch (pip install torch)
"""

import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------
class RULDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class LSTMRegressor(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, window_size, n_features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]          # (batch, hidden_size) — final layer's last timestep
        return self.head(last_hidden).squeeze(-1)


# ---------------------------------------------------------------------------
# C-MAPSS scoring function (same as Week 2, kept identical for fair comparison)
# ---------------------------------------------------------------------------
def cmapss_score(y_true, y_pred):
    diff = y_pred - y_true
    score = np.where(
        diff < 0,
        np.exp(-diff / 13.0) - 1,
        np.exp(diff / 10.0) - 1,
    )
    return np.sum(score)


def evaluate(y_true, y_pred, label):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    score = cmapss_score(y_true, y_pred)
    print(f"[{label}] RMSE: {rmse:.2f}   C-MAPSS Score: {score:.2f}")
    return rmse, score


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_preds = []
    for batch in loader:
        X_batch = batch[0] if isinstance(batch, (list, tuple)) else batch
        X_batch = X_batch.to(device)
        preds = model(X_batch)
        all_preds.append(preds.cpu().numpy())
    return np.concatenate(all_preds)


def main():
    parser = argparse.ArgumentParser(description="Week 3 LSTM model training")
    parser.add_argument("--data_dir", type=str, default="./processed",
                         help="Directory with Week 1 processed .npy files")
    parser.add_argument("--output_dir", type=str, default="./models",
                         help="Directory to save the trained LSTM model")
    parser.add_argument("--baseline_path", type=str, default="./models/baseline_results.json",
                         help="Path to Week 2 baseline results for comparison")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.15)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] using {device}")

    # 1. Load Week 1 outputs (raw sequences, NOT flattened)
    X_train = np.load(os.path.join(args.data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(args.data_dir, "y_train.npy"))
    X_test = np.load(os.path.join(args.data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(args.data_dir, "y_test.npy"))
    print(f"[load] X_train: {X_train.shape}, X_test: {X_test.shape}")

    n_features = X_train.shape[2]

    # 2. Train/validation split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=args.val_split, random_state=42
    )

    train_loader = DataLoader(RULDataset(X_tr, y_tr), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(RULDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(RULDataset(X_test), batch_size=args.batch_size, shuffle=False)

    # 3. Build model
    model = LSTMRegressor(
        n_features=n_features,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # 4. Train
    print("\n--- Training LSTM ---")
    best_val_rmse = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_preds = predict(model, val_loader, device)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs} | train loss: {train_loss:.2f} | val RMSE: {val_rmse:.2f}")

    # 5. Load best checkpoint (lowest validation RMSE across training)
    model.load_state_dict(best_state)
    print(f"\n[best] validation RMSE: {best_val_rmse:.2f}")

    # 6. Final evaluation on validation and test sets
    val_preds = predict(model, val_loader, device)
    evaluate(y_val, val_preds, "LSTM - validation")

    test_preds = predict(model, test_loader, device)
    test_rmse, test_score = evaluate(y_test, test_preds, "LSTM - TEST")

    # 7. Save model
    model_path = os.path.join(args.output_dir, "lstm_model.pt")
    torch.save(model.state_dict(), model_path)

    lstm_results = {
        "lstm": {
            "rmse": float(test_rmse),
            "score": float(test_score),
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "epochs": args.epochs,
        }
    }

    # 8. Compare against Week 2 baseline, if available
    print("\n--- Comparison against Week 2 baseline ---")
    if os.path.exists(args.baseline_path):
        with open(args.baseline_path) as f:
            baseline = json.load(f)
        combined = {**baseline, **lstm_results}
        for name, metrics in combined.items():
            print(f"{name:15s}  RMSE: {metrics['rmse']:.2f}   Score: {metrics['score']:.2f}")

        best_name = min(combined, key=lambda k: combined[k]["rmse"])
        print(f"\nBest model by RMSE: {best_name}")
    else:
        print(f"[warn] baseline file not found at {args.baseline_path} — skipping comparison")
        combined = lstm_results

    # 9. Save combined results
    results_path = os.path.join(args.output_dir, "all_results.json")
    with open(results_path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\n[done] LSTM model saved to: {model_path}")
    print(f"[done] Combined results saved to: {results_path}")


if __name__ == "__main__":
    main()
