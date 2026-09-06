"""
C-MAPSS FD001 Baseline Model — Week 2
======================================
Trains Random Forest and XGBoost regressors to predict Remaining Useful
Life (RUL) from the preprocessed C-MAPSS sequences produced in Week 1.

Since tree-based models take flat feature vectors (not 3D sequences), this
script flattens each (window_size, n_features) window and adds a few
engineered features (mean / std / last-value / trend) per sensor to help
recover some of the time-dependency that flattening would otherwise lose.

Evaluates with:
  - RMSE (standard regression metric)
  - The official C-MAPSS scoring function (asymmetric: penalizes LATE
    predictions much more heavily than early ones, since predicting an
    engine will fail later than it actually does is the dangerous case)

Usage:
    python train_baseline_model.py --data_dir ./processed --model both

Expects these files in --data_dir (produced by Week 1's preprocess_cmapss.py):
    X_train.npy, y_train.npy, X_test.npy, y_test.npy, config.json
"""

import os
import json
import pickle
import argparse

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def load_processed_data(data_dir):
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    with open(os.path.join(data_dir, "config.json")) as f:
        config = json.load(f)
    return X_train, y_train, X_test, y_test, config


def engineer_features(X):
    """
    Turn each (n_samples, window_size, n_features) array into a flat
    (n_samples, n_features * 4) array using, per sensor:
      - mean over the window
      - std over the window
      - last value in the window (most recent reading)
      - simple trend (last value - first value)
    This keeps some time-series signal even though the model itself
    (Random Forest / XGBoost) only sees flat feature vectors.
    """
    mean_feats = X.mean(axis=1)
    std_feats = X.std(axis=1)
    last_feats = X[:, -1, :]
    trend_feats = X[:, -1, :] - X[:, 0, :]
    return np.concatenate([mean_feats, std_feats, last_feats, trend_feats], axis=1)


def cmapss_score(y_true, y_pred):
    """
    Official C-MAPSS asymmetric scoring function.
    Late predictions (predicted RUL > actual RUL, i.e. model was overly
    optimistic and engine fails sooner than predicted) are penalized far
    more heavily than early predictions, reflecting real safety cost.
    """
    diff = y_pred - y_true
    score = np.where(
        diff < 0,
        np.exp(-diff / 13.0) - 1,   # early prediction, mild penalty
        np.exp(diff / 10.0) - 1,    # late prediction, steep penalty
    )
    return np.sum(score)


def evaluate(y_true, y_pred, label):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    score = cmapss_score(y_true, y_pred)
    print(f"[{label}] RMSE: {rmse:.2f}   C-MAPSS Score: {score:.2f}")
    return rmse, score


def train_random_forest(X_train, y_train, X_val, y_val):
    print("\n--- Training Random Forest ---")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    evaluate(y_val, val_pred, "RandomForest - validation")
    return model


def train_xgboost(X_train, y_train, X_val, y_val):
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("\n[skip] xgboost not installed. Run: pip install xgboost")
        return None

    print("\n--- Training XGBoost ---")
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    evaluate(y_val, val_pred, "XGBoost - validation")
    return model


def main():
    parser = argparse.ArgumentParser(description="Week 2 baseline model training")
    parser.add_argument("--data_dir", type=str, default="./processed",
                         help="Directory with Week 1 processed .npy files and config.json")
    parser.add_argument("--output_dir", type=str, default="./models",
                         help="Directory to save trained models")
    parser.add_argument("--model", type=str, default="both",
                         choices=["rf", "xgb", "both"],
                         help="Which model(s) to train")
    parser.add_argument("--val_split", type=float, default=0.15,
                         help="Fraction of training data held out for validation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Week 1 outputs
    X_train_raw, y_train, X_test_raw, y_test, config = load_processed_data(args.data_dir)
    print(f"[load] X_train: {X_train_raw.shape}, X_test: {X_test_raw.shape}")

    # 2. Flatten + engineer features
    X_train_flat = engineer_features(X_train_raw)
    X_test_flat = engineer_features(X_test_raw)
    print(f"[features] flattened X_train: {X_train_flat.shape}, X_test: {X_test_flat.shape}")

    # 3. Train/validation split (keep test set untouched until the very end)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_flat, y_train, test_size=args.val_split, random_state=42
    )

    results = {}

    # 4. Train models
    if args.model in ("rf", "both"):
        rf_model = train_random_forest(X_tr, y_tr, X_val, y_val)
        test_pred = rf_model.predict(X_test_flat)
        rmse, score = evaluate(y_test, test_pred, "RandomForest - TEST")
        results["random_forest"] = {"rmse": float(rmse), "score": float(score)}
        with open(os.path.join(args.output_dir, "random_forest.pkl"), "wb") as f:
            pickle.dump(rf_model, f)

    if args.model in ("xgb", "both"):
        xgb_model = train_xgboost(X_tr, y_tr, X_val, y_val)
        if xgb_model is not None:
            test_pred = xgb_model.predict(X_test_flat)
            rmse, score = evaluate(y_test, test_pred, "XGBoost - TEST")
            results["xgboost"] = {"rmse": float(rmse), "score": float(score)}
            with open(os.path.join(args.output_dir, "xgboost.pkl"), "wb") as f:
                pickle.dump(xgb_model, f)

    # 5. Save results summary
    with open(os.path.join(args.output_dir, "baseline_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[done] Models and results saved to: {args.output_dir}")
    print("Compare RMSE and C-MAPSS Score across models — lower is better for both.")
    print("These numbers are your baseline to beat with the LSTM in Week 3.")


if __name__ == "__main__":
    main()
