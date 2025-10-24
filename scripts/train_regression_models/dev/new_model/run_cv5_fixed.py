
# run_cv5_fixed.py
# No-CLI 5-fold run with your exact paths + settings.
# - 100 epochs, batch 32
# - Train/Val split = 80/20 from TRAIN in each fold
# - Uses your .bin format: X_train, X_test, Y_train, Y_test
# - Applies N,T,C -> N,C,T transpose (set TRANSPOSE_NTC accordingly)
# - Fits bias on TRAIN only; outputs BAG for VAL/TEST
import os, pickle, json
import numpy as np
import pandas as pd
from datetime import datetime

from normative_brain_age import (NormativeAgeNet, TrainConfig, NormativeTrainer,
    make_loader, fit_bias_correction, apply_bias_correction, compute_bag,
    save_predictions_csv)

# ------------------ User-fixed settings ------------------
FOLD_FILES = [f"/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/hcp_dev_age_five_fold/fold_{i}.bin" for i in range(5)]
OUTDIR = "cv5_runs_fixed"
EPOCHS = 100
BATCH_SIZE = 32
VAL_SPLIT = 0.20   # 80/20 split
LR = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 12
SEED = 42
NUM_WORKERS = 0
DROPOUT = 0.30
TRANSPOSE_NTC = True   # Set False if your arrays are already [N, C, T]

# Optional: if a boolean mask 'td_mask_train' is present in the .bin, use it to keep training TD-only
APPLY_TD_MASK_IF_PRESENT = True

# ---------------------------------------------------------
def load_finetune_dataset(path: str):
    with open(path, "rb") as fp:
        d = pickle.load(fp)
    X_train, X_test, Y_train, Y_test = d["X_train"], d["X_test"], d["Y_train"], d["Y_test"]
    extra = {k:v for k,v in d.items() if k not in {"X_train","X_test","Y_train","Y_test"}}
    return X_train, X_test, Y_train, Y_test, extra

def reshape_ntc_to_nct(x: np.ndarray) -> np.ndarray:
    return np.transpose(x, (0,2,1))

def ensure_float32(*arrs):
    out=[]; 
    for a in arrs:
        out.append(a.astype(np.float32) if isinstance(a, np.ndarray) and a.dtype!=np.float32 else a)
    return out

def train_val_split(X, y, ids, val_split, seed):
    n=len(y); rng=np.random.default_rng(seed); idx=np.arange(n); rng.shuffle(idx)
    n_val=int(round(val_split*n)); val_idx=idx[:n_val]; tr_idx=idx[n_val:]
    return (X[tr_idx], y[tr_idx], ids[tr_idx]), (X[val_idx], y[val_idx], ids[val_idx])

def run_fold(fold_path: str):
    fold_name = os.path.splitext(os.path.basename(fold_path))[0]
    fold_dir = os.path.join(OUTDIR, fold_name); os.makedirs(fold_dir, exist_ok=True)

    X_train, X_test, Y_train, Y_test, extra = load_finetune_dataset(fold_path)

    if APPLY_TD_MASK_IF_PRESENT and "td_mask_train" in extra:
        m = np.asarray(extra["td_mask_train"]).astype(bool)
        X_train, Y_train = X_train[m], Y_train[m]

    if TRANSPOSE_NTC:
        X_train = reshape_ntc_to_nct(X_train)
        X_test  = reshape_ntc_to_nct(X_test)

    ids_train = np.array([f"{fold_name}_tr_{i}" for i in range(len(Y_train))], dtype=object)
    ids_test  = np.array([f"{fold_name}_te_{i}" for i in range(len(Y_test))], dtype=object)

    X_train, X_test = ensure_float32(X_train, X_test)
    Y_train = Y_train.astype(np.float32).reshape(-1)
    Y_test  = Y_test.astype(np.float32).reshape(-1)

    (Xtr,ytr,idtr), (Xva,yva,idva) = train_val_split(X_train, Y_train, ids_train, VAL_SPLIT, SEED)

    cfg = TrainConfig(lr=LR, weight_decay=WEIGHT_DECAY, epochs=EPOCHS, batch_size=BATCH_SIZE,
                      patience=PATIENCE, seed=SEED)
    model = NormativeAgeNet(in_ch=Xtr.shape[1], dropout=DROPOUT)
    trainer = NormativeTrainer(model, cfg)

    dl_tr = make_loader(Xtr, ytr, ids=idtr, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    dl_va = make_loader(Xva, yva, ids=idva, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    dl_te = make_loader(X_test, Y_test, ids=ids_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    trainer.fit(dl_tr, dl_va)

    mu_tr, logv_tr, y_tr, ids_tr = trainer.predict(dl_tr)
    mu_va, logv_va, y_va, ids_va = trainer.predict(dl_va)
    mu_te, logv_te, y_te, ids_te = trainer.predict(dl_te)

    params = fit_bias_correction(y_true=y_tr, y_pred=mu_tr)

    mu_tr_bc = apply_bias_correction(y_true=y_tr, y_pred=mu_tr, params=params)
    mu_va_bc = apply_bias_correction(y_true=y_va, y_pred=mu_va, params=params)
    mu_te_bc = apply_bias_correction(y_true=y_te, y_pred=mu_te, params=params)

    bag_tr = compute_bag(y_true=y_tr, y_pred_corr=mu_tr_bc)
    bag_va = compute_bag(y_true=y_va, y_pred_corr=mu_va_bc)
    bag_te = compute_bag(y_true=y_te, y_pred_corr=mu_te_bc)

    save_predictions_csv(os.path.join(fold_dir, "train_predictions.csv"), ids_tr, y_tr, mu_tr, mu_tr_bc, bag_tr,
                         extra={"pred_var": np.exp(logv_tr)})
    save_predictions_csv(os.path.join(fold_dir, "val_predictions.csv"), ids_va, y_va, mu_va, mu_va_bc, bag_va,
                         extra={"pred_var": np.exp(logv_va)})
    save_predictions_csv(os.path.join(fold_dir, "test_predictions.csv"), ids_te, y_te, mu_te, mu_te_bc, bag_te,
                         extra={"pred_var": np.exp(logv_te)})

    metrics = {
        "fold": fold_name,
        "n_train": int(len(y_tr)), "n_val": int(len(y_va)), "n_test": int(len(y_te)),
        "val_mae_raw": float(np.mean(np.abs(mu_va - y_va))),
        "val_mae_bc": float(np.mean(np.abs(mu_va_bc - y_va))),
        "test_mae_raw": float(np.mean(np.abs(mu_te - y_te))),
        "test_mae_bc": float(np.mean(np.abs(mu_te_bc - y_te))),
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
    print("\n=== CV Summary ===")
    print(df[["fold","val_mae_raw","val_mae_bc","test_mae_raw","test_mae_bc"]])
    print(f"\nVal MAE raw: {df['val_mae_raw'].mean():.4f} | Val MAE bias-corr: {df['val_mae_bc'].mean():.4f}")
    print(f"Test MAE raw: {df['test_mae_raw'].mean():.4f} | Test MAE bias-corr: {df['test_mae_bc'].mean():.4f}")

if __name__ == "__main__":
    main()
