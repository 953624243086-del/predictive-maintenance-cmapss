# Aircraft Engine Predictive Maintenance System (NASA C-MAPSS)

A full-stack predictive maintenance system with an integrated ML pipeline,
built on NASA's C-MAPSS (Commercial Modular Aero-Propulsion System
Simulation) turbofan engine degradation dataset. Predicts Remaining Useful
Life (RUL) from multivariate sensor time-series data.

**Status:** 🚧 In progress — Week 2 of 8 (see [Roadmap](#roadmap))

---

## Overview

C-MAPSS simulates multiple turbofan engines running under normal operation
until failure, recording 21 sensors and 3 operational settings per cycle.
The goal is to predict how many operating cycles remain before an engine
fails (RUL), enabling proactive maintenance instead of reactive repair.

This project builds:
1. A reproducible data preprocessing pipeline
2. ML models (baseline + deep learning) for RUL prediction
3. A backend API serving predictions
4. A frontend dashboard for fleet monitoring
5. A simulated real-time streaming layer

## Dataset

Using the **FD001** subset (single operating condition, single fault mode)
from NASA's C-MAPSS dataset.

Download from NASA's Prognostics Center of Excellence Data Repository, or
via [this Kaggle mirror](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps),
then place these files in `data/`:
- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

## Setup

```bash
git clone https://github.com/953624243086-del/predictive-maintenance-cmapss.git
cd predictive-maintenance-cmapss
pip install -r requirements.txt
```

### 1. Preprocess the data
```bash
python src/preprocess_cmapss.py --data_dir ./data --output_dir ./processed --window_size 30 --clip_rul 125
```
Produces `X_train.npy`, `y_train.npy`, `X_test.npy`, `y_test.npy`,
`scaler.pkl`, and `config.json` in `processed/`.

### 2. Train baseline models
```bash
python src/train_baseline_model.py --data_dir ./processed --output_dir ./models --model both
```
Trains Random Forest and XGBoost regressors, evaluates with RMSE and the
official C-MAPSS asymmetric scoring function, saves results to
`models/baseline_results.json`.

## Results

### Baseline models (Week 2)

| Model | Test RMSE | Test Score |
|---|---|---|
| Random Forest | 18.10 | 607.18 |
| **XGBoost** | **17.32** | **559.93** |

*Lower is better for both metrics. The C-MAPSS score is asymmetric — it
penalizes late predictions (predicting an engine will last longer than it
actually does) far more heavily than early ones, reflecting real
maintenance safety costs.*

## Preprocessing notes

- Feature columns with near-zero variance were dropped automatically
  based on measured standard deviation (not a hardcoded list):
  `op_setting_3, sensor_1, sensor_5, sensor_10, sensor_16, sensor_18, sensor_19`
- Training RUL labels are clipped at 125 cycles, since early-life
  degradation is negligible and unclipped labels hurt model training.
- Features are scaled with `MinMaxScaler`, fit only on training data to
  avoid leakage; the fitted scaler is saved for reuse on new/live data.
- Sequences use a 30-cycle sliding window. Engines with fewer than 30
  cycles are padded by repeating their first row.

## Roadmap

- [x] **Week 1** — Data preprocessing pipeline
- [x] **Week 2** — Baseline models (Random Forest, XGBoost)
- [ ] **Week 3** — LSTM/GRU deep learning model
- [ ] **Week 4** — Backend API (FastAPI) + database
- [ ] **Week 5** — Frontend dashboard (React)
- [ ] **Week 6** — Simulated real-time streaming layer
- [ ] **Week 7** — Auth, error handling, polish
- [ ] **Week 8** — Dockerize, deploy, document

## Project structure

```
predictive-maintenance-cmapss/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                     # raw C-MAPSS files (not committed)
├── src/
│   ├── preprocess_cmapss.py
│   └── train_baseline_model.py
├── processed/
│   └── config.json           # feature cols, window size, clip value
└── models/
    └── baseline_results.json # RMSE / score per model
```

## Tech stack

- **Data/ML:** Python, pandas, scikit-learn, XGBoost (PyTorch for LSTM, coming Week 3)
- **Backend:** FastAPI (coming Week 4)
- **Database:** PostgreSQL (coming Week 4)
- **Frontend:** React (coming Week 5)
- **Deployment:** Docker, Render/Vercel (coming Week 8)

## Acknowledgments

Dataset: A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation
Simulation Data Set", NASA Ames Prognostics Data Repository, NASA Ames
Research Center, Moffett Field, CA.
