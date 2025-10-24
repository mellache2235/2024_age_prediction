#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HCP-Dev OOF Evaluation (r² and p)
---------------------------------
- Per-fold: load TRAIN/VALID data, transform to FCz
- Load pre-trained fold model
- Fit bias on TRAIN (ŷ = a*y + b), apply to VALID predictions
- Concatenate OOF predictions across folds
- Report r² (p) per model in a dataset-centric wide table
- Save CSV: hcpdev_oof_r2_p_wide.csv
"""

import os
import math
import pickle
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from scipy import stats
from sklearn import svm, tree, neighbors, linear_model, ensemble

# -----------------------------
# Config — adjust if necessary
# -----------------------------
K = 5
MODEL_SUFFIX = "dev_age"

# Where the per-fold .bin files live (must have fold_0.bin ... fold_4.bin)
FOLD_BIN_TEMPLATE = "/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/hcp_dev_age_five_fold/fold_{k}.bin"

# Where the pre-trained .sav models live
MODEL_DIR = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/models/dev/ml_models"

# Regressor names that correspond to the saved model filenames
MODEL_NAMES = ["linSVR", "KNN", "DT", "LR", "RC", "LASSO", "RF"]

# -----------------------------
# Minimal data I/O
# -----------------------------
def load_finetune_dataset(path: str):
    """Load a numpy-pickled dict with keys: X_train, X_test, Y_train, Y_test."""
    with open(path, "rb") as fp:
        data = pickle.load(fp)
    return data["X_train"], data["X_test"], data["Y_train"], data["Y_test"]

def load_hcpdev_fold_timeseries(k: int):
    """Return (X_train_ts, X_val_ts, y_train, y_val) for fold k."""
    path = FOLD_BIN_TEMPLATE.format(k=k)
    X_tr, X_va, y_tr, y_va = load_finetune_dataset(path)
    return X_tr, X_va, y_tr, y_va

# -----------------------------
# Feature transform: TS -> FCz
# -----------------------------
def timeseries_to_fcz(X_ts: np.ndarray) -> np.ndarray:
    """
    Convert per-subject time-series [N, T, R] to static FCz features:
      - Pearson correlation (R x R)
      - upper triangle (k=1)
      - Fisher z (arctanh) with safe clipping
    Returns: X_fcz [N, R*(R-1)/2]
    """
    if X_ts.ndim != 3:
        raise ValueError(f"Expected [N,T,R], got {X_ts.shape}")
    n, T, R = X_ts.shape
    iu = np.triu_indices(R, k=1)
    X_fcz = np.empty((n, len(iu[0])), dtype=np.float32)
    for i in range(n):
        fc = pd.DataFrame(X_ts[i]).corr(method="pearson").to_numpy()
        np.clip(fc, -0.999999, 0.999999, out=fc)
        X_fcz[i, :] = np.arctanh(fc[iu])
    return X_fcz

# -----------------------------
# Bias correction + stats
# -----------------------------
def fit_bias(y_true_td: np.ndarray, y_pred_td: np.ndarray) -> Tuple[float, float]:
    """
    Fit y_pred = a*y_true + b via least squares. Returns (a, b).
    """
    y_true_td = np.asarray(y_true_td).reshape(-1)
    y_pred_td = np.asarray(y_pred_td).reshape(-1)
    if y_true_td.size < 2 or np.std(y_true_td) == 0:
        return 1.0, 0.0
    A = np.vstack([y_true_td, np.ones_like(y_true_td)]).T
    sol, *_ = np.linalg.lstsq(A, y_pred_td, rcond=None)
    a, b = float(sol[0]), float(sol[1])
    if not np.isfinite(a) or abs(a) < 1e-6: a = 1.0
    if not np.isfinite(b): b = 0.0
    return a, b

def apply_bias(y_pred: np.ndarray, a: float, b: float) -> np.ndarray:
    """y_corr = (y_pred - b) / a (safe divide)."""
    y_pred = np.asarray(y_pred).reshape(-1)
    if not np.isfinite(a) or abs(a) < 1e-6:
        return y_pred
    return (y_pred - b) / a

def pearson_r2_p(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Return (r², p)."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan"), float("nan")
    r, p = stats.pearsonr(y_true, y_pred)
    return float(r**2), float(p)

# -----------------------------
# Model utilities
# -----------------------------
def load_fold_model(model_name: str, k: int):
    """
    Load a single fold model: model_{model_name}_dev_age_{k}.sav
    """
    fname = os.path.join(MODEL_DIR, f"model_{model_name}_{MODEL_SUFFIX}_{k}.sav")
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Missing model file: {fname}")
    try:
        return joblib.load(fname)
    except Exception:
        with open(fname, "rb") as fp:
            return pickle.load(fp)

# -----------------------------
# OOF evaluation for one model
# -----------------------------
def eval_hcpdev_oof_for_model(model_name: str) -> Tuple[float, float]:
    """
    For a single model type:
      - Loop k=0..K-1
      - Load fold-k model
      - Load fold-k TRAIN/VALID time-series, transform to FCz
      - Fit bias on TRAIN, apply correction to VALID predictions
      - Accumulate VALID ŷ_corr and y
    Return: (r², p) on concatenated OOF VALID predictions.
    """
    print(f"\n--- Evaluating {model_name} ---")
    y_pred_all, y_true_all = [], []

    for k in range(K):
        print(f"  Fold {k}: load data")
        X_tr_ts, X_va_ts, y_tr, y_va = load_hcpdev_fold_timeseries(k)

        print(f"  Fold {k}: transform TRAIN/VAL to FCz")
        X_tr = timeseries_to_fcz(X_tr_ts)
        X_va = timeseries_to_fcz(X_va_ts)

        print(f"  Fold {k}: load model file")
        model_k = load_fold_model(model_name, k)

        print(f"  Fold {k}: predict TRAIN (for bias fit) & VALID (for scoring)")
        yhat_tr = model_k.predict(X_tr).reshape(-1)
        yhat_va = model_k.predict(X_va).reshape(-1)

        print(f"  Fold {k}: fit bias on TRAIN, apply to VALID")
        a, b = fit_bias(y_tr, yhat_tr)
        yhat_va_corr = apply_bias(yhat_va, a, b)

        y_pred_all.append(yhat_va_corr)
        y_true_all.append(y_va)

    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)

    r2, p = pearson_r2_p(y_true, y_pred)
    print(f"  -> OOF r²: {r2:.3f}, p: {p:.2e}")
    return r2, p

# -----------------------------
# Pretty printer + CSV
# -----------------------------
def print_dataset_centric_table(results: Dict[str, Tuple[float, float]]):
    """
    One row (HCP-Dev Val), columns are models; each cell 'r² (p)'.
    """
    dataset_name = "HCP-Dev Val"
    header = ["Dataset"] + list(results.keys())
    def cell(v): 
        r2, p = v
        return f"{r2:.3f} ({p:.2e})" if np.isfinite(r2) and np.isfinite(p) else "nan"

    # widths
    col_widths = [len(h) for h in header]
    col_widths[0] = max(col_widths[0], len(dataset_name))
    for i, m in enumerate(results.keys(), start=1):
        col_widths[i] = max(col_widths[i], len(cell(results[m])))

    def mk_row(cells): 
        return "| " + " | ".join(s.ljust(w) for s, w in zip(cells, col_widths)) + " |"

    print("\n=== Brain Age (HCP-Dev OOF): r² (p) by Model ===")
    print(mk_row(header))
    print("| " + " | ".join("-"*w for w in col_widths) + " |")
    row = [dataset_name] + [cell(results[m]) for m in results.keys()]
    print(mk_row(row))

def save_wide_csv(results: Dict[str, Tuple[float, float]], csv_path: str = "hcpdev_oof_r2_p_wide.csv"):
    """
    Save a one-row wide CSV: Dataset, <model1>, <model2>, ...
    Cell values are 'r2 (p)' strings to match the printed table.
    """
    dataset_name = "HCP-Dev Val"
    header = ["Dataset"] + list(results.keys())
    def cell(v):
        r2, p = v
        return f"{r2:.3f} ({p:.2e})" if np.isfinite(r2) and np.isfinite(p) else "nan"

    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow([dataset_name] + [cell(results[m]) for m in results.keys()])
    print(f"\nSaved CSV: {csv_path}")

# -----------------------------
# Main
# -----------------------------
def main():
    print("\n=== HCP-Dev OOF evaluation (bias from TRAIN, score on VALID) ===")
    # Ensure consistent model column order like your prior tables
    results_ordered: Dict[str, Tuple[float, float]] = {}
    for model_name in MODEL_NAMES:
        try:
            r2, p = eval_hcpdev_oof_for_model(model_name)
        except FileNotFoundError as e:
            print(f"[warn] {model_name}: {e}")
            r2, p = float("nan"), float("nan")
        except Exception as e:
            print(f"[warn] {model_name} error: {e}")
            r2, p = float("nan"), float("nan")
        results_ordered[model_name] = (r2, p)

    # Pretty print + save
    print_dataset_centric_table(results_ordered)
    save_wide_csv(results_ordered)

if __name__ == "__main__":
    main()

