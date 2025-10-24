
# run_cv5_trainval_fixed.py
# ------------------------------------------------------------
# 5-fold run using only TRAIN and VALIDATION from each .bin.
# - No extra splitting; uses the provided TRAIN for fitting and VAL for early stopping/eval.
# - 100 epochs, batch size 32.
# - Fits bias-correction on TRAIN predictions ONLY; applies to both TRAIN and VAL.
# - Saves predictions + metrics per fold and a summary CSV.
# - Assumes arrays saved as [N, T, C]; set TRANSPOSE_NTC=False if already [N, C, T].
# ------------------------------------------------------------

import os, pickle, json
import numpy as np
import pandas as pd

from normative_brain_age import (
    NormativeAgeNet, TrainConfig, NormativeTrainer,
    make_loader, fit_bias_correction, apply_bias_correction, compute_bag,
    save_predictions_csv
)

# ------------------ Fixed settings ------------------
FOLD_FILES = [f"/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/hcp_dev_age_five_fold/fold_{i}.bin" for i in range(5)]
OUTDIR = "cv5_runs_trainval_fixed"
EPOCHS = 100
BATCH_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 12
SEED = 42
NUM_WORKERS = 0
DROPOUT = 0.30
TRANSPOSE_NTC = True           # Original shape is [N, T, C]; set False if already [N, C, T]
APPLY_TD_MASK_IF_PRESENT = False  # Set True if your .bin includes 'td_mask_train' and you want TD-only training

# ------------------ Utils ------------------
ALIASES = {
    'X_train': ['X_train','x_train','trainX'],
    'y_train': ['y_train','Y_train','trainY'],
    'X_val':   ['X_val','x_val','X_valid','validX','X_validation'],
    'y_val':   ['y_val','Y_val','validY','y_valid','y_validation'],
    'ids_train': ['ids_train','train_ids','subjects_train','sid_train'],
    'ids_val':   ['ids_val','valid_ids','subjects_val','sid_val'],
}

def _get(d, key):
    for k in ALIASES.get(key, [key]):
        if k in d: return d[k]
    return None

def load_trainval_bin(path):
    with open(path, "rb") as fp:
        d = pickle.load(fp)
    Xtr = _get(d,'X_train'); ytr = _get(d,'y_train')
    Xva = _get(d,'X_val');   yva = _get(d,'y_val')
    if Xtr is None or ytr is None or Xva is None or yva is None:
        raise ValueError(f"{path}: expected TRAIN/VAL keys (e.g., X_train, y_train, X_val, y_val)")
    idtr = _get(d,'ids_train'); idva = _get(d,'ids_val')
    extra = {k:v for k,v in d.items() if k not in set(sum(ALIASES.values(), []))}
    return Xtr, ytr, Xva, yva, idtr, idva, extra

def reshape_ntc_to_nct(x: np.ndarray) -> np.ndarray:
    return np.transpose(x, (0,2,1))

def ensure_float32(*arrs):
    out=[]; 
    for a in arrs:
        out.append(a.astype(np.float32) if isinstance(a, np.ndarray) and a.dtype!=np.float32 else a)
    return out

# ------------------ Per-fold run ------------------
def run_fold(fold_path: str):
    fold_name = os.path.splitext(os.path.basename(fold_path))[0]
    fold_dir = os.path.join(OUTDIR, fold_name); os.makedirs(fold_dir, exist_ok=True)

    X_train, y_train, X_val, y_val, ids_train, ids_val, extra = load_trainval_bin(fold_path)

    # Optional TD-only filter
    if APPLY_TD_MASK_IF_PRESENT and 'td_mask_train' in extra:
        m = np.asarray(extra['td_mask_train']).astype(bool)
        X_train, y_train = X_train[m], y_train[m]
        if ids_train is not None:
            ids_train = np.asarray(ids_train)[m]

    # Transpose if original is [N,T,C]
    if TRANSPOSE_NTC:
        X_train = reshape_ntc_to_nct(X_train)
        X_val   = reshape_ntc_to_nct(X_val)

    # IDs
    if ids_train is None: ids_train = np.array([f"{fold_name}_tr_{i}" for i in range(len(y_train))], dtype=object)
    else: ids_train = np.asarray(ids_train, dtype=object)
    if ids_val   is None: ids_val   = np.array([f"{fold_name}_va_{i}" for i in range(len(y_val))], dtype=object)
    else: ids_val = np.asarray(ids_val, dtype=object)

    # Dtypes
    X_train, X_val = ensure_float32(X_train, X_val)
    y_train = np.asarray(y_train).astype(np.float32).reshape(-1)
    y_val   = np.asarray(y_val).astype(np.float32).reshape(-1)

    # Build + train
    cfg = TrainConfig(lr=LR, weight_decay=WEIGHT_DECAY, epochs=EPOCHS, batch_size=BATCH_SIZE,
                      patience=PATIENCE, seed=SEED)
    #model = NormativeAgeNet(in_ch=X_train.shape[1], dropout=DROPOUT)
    model = NormativeAgeNet(in_ch=246, stem_ch=64, blocks=(64,32), embed_dim=32, dropout=0.3)
    trainer = NormativeTrainer(model, cfg)

    dl_tr = make_loader(X_train, y_train, ids=ids_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    dl_va = make_loader(X_val,   y_val,   ids=ids_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    trainer.fit(dl_tr, dl_va)

    # Predict TRAIN and VAL
    mu_tr, logv_tr, y_tr, id_tr = trainer.predict(dl_tr)
    mu_va, logv_va, y_va, id_va = trainer.predict(dl_va)

    # Bias on TRAIN-only
    params = fit_bias_correction(y_true=y_tr, y_pred=mu_tr)

    mu_tr_bc = apply_bias_correction(y_true=y_tr, y_pred=mu_tr, params=params)
    mu_va_bc = apply_bias_correction(y_true=y_va, y_pred=mu_va, params=params)

    bag_tr = compute_bag(y_true=y_tr, y_pred_corr=mu_tr_bc)
    bag_va = compute_bag(y_true=y_va, y_pred_corr=mu_va_bc)

    # Save
    save_predictions_csv(os.path.join(fold_dir, "train_predictions.csv"), id_tr, y_tr, mu_tr, mu_tr_bc, bag_tr,
                         extra={"pred_var": np.exp(logv_tr)})
    save_predictions_csv(os.path.join(fold_dir, "val_predictions.csv"), id_va, y_va, mu_va, mu_va_bc, bag_va,
                         extra={"pred_var": np.exp(logv_va)})

    metrics = {
        "fold": fold_name,
        "n_train": int(len(y_tr)), "n_val": int(len(y_va)),
        "val_mae_raw": float(np.mean(np.abs(mu_va - y_va))),
        "val_mae_bc": float(np.mean(np.abs(mu_va_bc - y_va))),
        "train_mae_raw": float(np.mean(np.abs(mu_tr - y_tr))),
        "train_mae_bc": float(np.mean(np.abs(mu_tr_bc - y_tr))),
        "bias_alpha": float(params.alpha), "bias_beta": float(params.beta),
    }
    with open(os.path.join(fold_dir, "metrics.json"), "w") as f: json.dump(metrics, f, indent=2)
    with open(os.path.join(fold_dir, "bias_params.json"), "w") as f: json.dump({"alpha": float(params.alpha), "beta": float(params.beta)}, f, indent=2)
    return metrics

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    all_metrics = []
    for fp in FOLD_FILES:
        print(f"=== Running fold: {fp} ===")
        m = run_fold(fp); all_metrics.append(m)
    df = pd.DataFrame(all_metrics).sort_values("fold")
    df.to_csv(os.path.join(OUTDIR, "cv_summary.csv"), index=False)
    print("\n=== CV Summary (VAL) ===")
    print(df[["fold","val_mae_raw","val_mae_bc"]])
    print(f"\nVal MAE raw: {df['val_mae_raw'].mean():.4f} | Val MAE bias-corr: {df['val_mae_bc'].mean():.4f}")

if __name__ == "__main__":
    main()
