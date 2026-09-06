"""
C-MAPSS FD001 Preprocessing Pipeline
=====================================
Week 1 deliverable for the Predictive Maintenance project.

Takes raw NASA C-MAPSS FD001 text files and produces:
  - Cleaned, RUL-labeled training data
  - Scaled features (scaler saved for reuse in the API later)
  - Sliding-window sequences ready for model training
  - A saved scaler + config so downstream code (Week 2+) can reuse
    the exact same preprocessing logic on new/live data

Usage:
    python preprocess_cmapss.py --data_dir ./data --window_size 30 --clip_rul 125

Expected input files in --data_dir:
    train_FD001.txt
    test_FD001.txt
    RUL_FD001.txt

(Download these from NASA's Prognostics Data Repository / Kaggle mirror
and place them in the data directory before running this script.)
"""

import os
import json
import pickle
import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------------------------
# Column names — C-MAPSS files have no header row
# ---------------------------------------------------------------------------
BASE_COLUMNS = ["unit_number", "time_cycle",
                "op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLUMNS = BASE_COLUMNS + SENSOR_COLUMNS


def load_raw_data(data_dir, dataset="FD001"):
    """Load train, test, and RUL files into pandas DataFrames."""
    train_path = os.path.join(data_dir, f"train_{dataset}.txt")
    test_path = os.path.join(data_dir, f"test_{dataset}.txt")
    rul_path = os.path.join(data_dir, f"RUL_{dataset}.txt")

    train = pd.read_csv(train_path, sep=r"\s+", header=None, names=ALL_COLUMNS)
    test = pd.read_csv(test_path, sep=r"\s+", header=None, names=ALL_COLUMNS)
    rul = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["RUL"])

    print(f"[load] train shape: {train.shape}, "
          f"test shape: {test.shape}, rul shape: {rul.shape}")
    print(f"[load] train engines: {train['unit_number'].nunique()}, "
          f"test engines: {test['unit_number'].nunique()}")
    return train, test, rul


def find_low_variance_columns(df, columns, std_threshold=1e-5):
    """Identify sensor/setting columns with near-zero variance (no signal)."""
    stds = df[columns].std()
    dropped = stds[stds < std_threshold].index.tolist()
    print(f"[variance] dropping low-variance columns: {dropped}")
    return dropped


def add_rul_labels(train_df, clip_rul):
    """Compute RUL per row for training data and clip early-life values."""
    max_cycles = train_df.groupby("unit_number")["time_cycle"].transform("max")
    train_df = train_df.copy()
    train_df["RUL"] = max_cycles - train_df["time_cycle"]
    if clip_rul is not None:
        train_df["RUL"] = train_df["RUL"].clip(upper=clip_rul)
    return train_df


def scale_features(train_df, test_df, feature_cols):
    """Fit a MinMaxScaler on train only, apply to both train and test."""
    scaler = MinMaxScaler()
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    return train_df, test_df, scaler


def create_sequences(df, feature_cols, window_size, label_col="RUL"):
    """
    Build sliding-window sequences per engine unit.
    Engines shorter than window_size are padded by repeating the first row.
    Returns: X (n_samples, window_size, n_features), y (n_samples,)
    """
    sequences, labels = [], []

    for unit in df["unit_number"].unique():
        unit_df = df[df["unit_number"] == unit].reset_index(drop=True)
        n_rows = len(unit_df)

        if n_rows < window_size:
            pad_needed = window_size - n_rows
            first_row = unit_df.iloc[[0]]
            padding = pd.concat([first_row] * pad_needed, ignore_index=True)
            unit_df = pd.concat([padding, unit_df], ignore_index=True)
            n_rows = len(unit_df)

        for i in range(n_rows - window_size + 1):
            window = unit_df[feature_cols].iloc[i:i + window_size].values
            sequences.append(window)
            if label_col in unit_df.columns:
                labels.append(unit_df[label_col].iloc[i + window_size - 1])

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.float32) if labels else None
    return X, y


def create_test_sequences(df, feature_cols, window_size):
    """
    For test data, take only the LAST window per engine
    (that's the point at which we need to predict RUL).
    """
    sequences = []
    for unit in df["unit_number"].unique():
        unit_df = df[df["unit_number"] == unit].reset_index(drop=True)
        n_rows = len(unit_df)

        if n_rows < window_size:
            pad_needed = window_size - n_rows
            first_row = unit_df.iloc[[0]]
            padding = pd.concat([first_row] * pad_needed, ignore_index=True)
            unit_df = pd.concat([padding, unit_df], ignore_index=True)

        window = unit_df[feature_cols].iloc[-window_size:].values
        sequences.append(window)

    return np.array(sequences, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="C-MAPSS Week 1 preprocessing pipeline")
    parser.add_argument("--data_dir", type=str, default="./data",
                         help="Directory containing train/test/RUL txt files")
    parser.add_argument("--output_dir", type=str, default="./processed",
                         help="Directory to save processed arrays and artifacts")
    parser.add_argument("--dataset", type=str, default="FD001",
                         help="Which C-MAPSS subset to process (FD001-FD004)")
    parser.add_argument("--window_size", type=int, default=30,
                         help="Sliding window size (number of cycles)")
    parser.add_argument("--clip_rul", type=int, default=125,
                         help="Max RUL value to clip training labels to")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load
    train, test, rul = load_raw_data(args.data_dir, args.dataset)

    # 2. Drop low-variance columns (check both settings and sensors)
    candidate_cols = [c for c in ALL_COLUMNS if c not in ["unit_number", "time_cycle"]]
    drop_cols = find_low_variance_columns(train, candidate_cols)
    feature_cols = [c for c in candidate_cols if c not in drop_cols]
    print(f"[features] keeping {len(feature_cols)} feature columns: {feature_cols}")

    # 3. Label training data with RUL
    train = add_rul_labels(train, args.clip_rul)

    # 4. Scale features (fit on train only)
    train, test, scaler = scale_features(train, test, feature_cols)

    # 5. Build sliding-window sequences
    X_train, y_train = create_sequences(train, feature_cols, args.window_size)
    X_test = create_test_sequences(test, feature_cols, args.window_size)
    y_test = rul["RUL"].values.astype(np.float32)

    print(f"[sequences] X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"[sequences] X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    # 6. Save everything needed downstream (Week 2+ and later the API)
    np.save(os.path.join(args.output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(args.output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(args.output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(args.output_dir, "y_test.npy"), y_test)

    with open(os.path.join(args.output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    config = {
        "dataset": args.dataset,
        "window_size": args.window_size,
        "clip_rul": args.clip_rul,
        "feature_cols": feature_cols,
        "dropped_cols": drop_cols,
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n[done] All artifacts saved to: {args.output_dir}")
    print("  - X_train.npy, y_train.npy, X_test.npy, y_test.npy")
    print("  - scaler.pkl (reuse this in your API to transform live sensor input)")
    print("  - config.json (feature columns, window size, clip value — reuse everywhere)")


if __name__ == "__main__":
    main()
