import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing, linear_model, model_selection, svm, tree, ensemble, neighbors
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.preprocessing import LabelEncoder
import pdb
import scipy
from scipy import stats
import joblib
from typing import Dict, List, Tuple
import os
import math

def timeseries_to_fcz(X_ts: np.ndarray) -> np.ndarray:
    """
    Convert per-subject time-series [N, T, R] to static FC features:
      - Pearson correlation (R x R)
      - take upper triangle (k=1)
      - Fisher z transform (arctanh) with safe clipping
    Returns: X_fcz [N, R*(R-1)/2]
    """
    if X_ts.ndim != 3:
        raise ValueError(f"Expected 3D timeseries array [N, T, R], got shape {X_ts.shape}")

    n_subj, n_t, n_roi = X_ts.shape
    n_feat = n_roi * (n_roi - 1) // 2
    X_fcz = np.empty((n_subj, n_feat), dtype=np.float32)

    # index for upper triangle (once)
    iu = np.triu_indices(n_roi, k=1)

    for i in range(n_subj):
        # shape [T, R] → DataFrame for convenient Pearson corr
        df = pd.DataFrame(X_ts[i])
        fc = df.corr(method="pearson").to_numpy()

        # numeric safety: bound r ∈ (-1, 1) before arctanh
        np.clip(fc, -0.999999, 0.999999, out=fc)

        X_fcz[i, :] = np.arctanh(fc[iu])

    return X_fcz

def load_finetune_dataset(path):
    """A function to load numpy dictionary for finetune data given path

    Args:
        path (string): A string path to load the numpy dictionary

    Returns:
        Numpy matrices (train_input, test_input, train_target, test_target)
    """
    fp = open(path, "rb")
    data_dict = pickle.load(fp)
    fp.close()
    return data_dict["X_train"], data_dict["X_test"], data_dict["Y_train"], data_dict["Y_test"]

def _drop_nan_subjects(X, y):
    """Remove subjects where any feature is NaN."""
    indices_to_keep = [i for i in range(len(X)) if not np.isnan(X[i]).any()]
    X_new = np.asarray([X[i] for i in indices_to_keep])
    y_new = np.asarray([y[i] for i in indices_to_keep])
    return X_new, y_new

def data_cleaning_nkirs_age(path_to_dataset):
    saved_data_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/nki_age_cog_dev_wIDs/fold_0.bin'
    X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(saved_data_path)

    X_total = np.concatenate((X_train,X_valid))
    Y_total = np.concatenate((Y_train,Y_valid))

    return X_total, Y_total

def data_cleaning_hcp_dev_age(path_to_dataset):
    saved_data_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/hcp_dev_age_five_fold/fold_0.bin'
    X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(saved_data_path)

    X_total = np.concatenate((X_train,X_valid))
    Y_total = np.concatenate((Y_train,Y_valid))

    return X_total, Y_total


def data_cleaning_adhd200_td_age(path_to_dataset):
    saved_data_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/adhd200_regression_age_TD_wIDs/fold_0.bin'
    X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(saved_data_path)

    X_total = np.concatenate((X_train,X_valid))
    Y_total = np.concatenate((Y_train,Y_valid))

    return X_total, Y_total

def data_cleaning_cmi_td_age(path_to_dataset):
    saved_data_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/cmihbn_age_TD/fold_0.bin'
    X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(saved_data_path)

    X_total = np.concatenate((X_train,X_valid))
    Y_total = np.concatenate((Y_train,Y_valid))

    return X_total, Y_total

def data_cleaning_abide_td_age(path_to_dataset):
    saved_data_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/abide_asd_td_dev_age/fold_0.bin'
    X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(saved_data_path)

    X_total = np.concatenate((X_train,X_valid))
    Y_total = np.concatenate((Y_train,Y_valid))
   
    return X_total, Y_total

def data_cleaning_stanford_td_age(path_to_dataset):
    saved_data_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/stanford_autism_age_TD_wIDS/fold_0.bin'
    X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(saved_data_path)

    X_total = np.concatenate((X_train,X_valid))
    Y_total = np.concatenate((Y_train,Y_valid))
 
    return X_total, Y_total

def data_cleaning_abide_asd_age(path_to_dataset):
    saved_data_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/abide_asd_asd_dev_age_wIDs/fold_0.bin'
    X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(saved_data_path)
    
    X_total = np.concatenate((X_train,X_valid))
    Y_total = np.concatenate((Y_train,Y_valid))

    return X_total, Y_total


def data_cleaning_stanford_autism_age(path_to_dataset):
    saved_data_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/stanford_autism_age_wIDS/fold_0.bin'
    X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(saved_data_path)

    X_total = np.concatenate((X_train,X_valid))
    Y_total = np.concatenate((Y_train,Y_valid))
 
    return X_total, Y_total


def data_cleaning_adhd200_age(path_to_dataset):
    saved_data_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/adhd200_regression_age_ADHD_wIDs/fold_0.bin'

    X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(saved_data_path)

    X_total = np.concatenate((X_train,X_valid))
    Y_total = np.concatenate((Y_train,Y_valid))

    return X_total, Y_total

def data_cleaning_cmi_age(path_to_dataset):
    saved_data_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/adhd_cmihbn_age_ADHD_wIDS/fold_0.bin'

    X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(saved_data_path)

    X_total = np.concatenate((X_train,X_valid))
    Y_total = np.concatenate((Y_train,Y_valid))

    return X_total, Y_total

# ------------------ Dataset wrappers (now convert to FC) ------------------

def _load_nkirs_TD():
    X_ts, y = data_cleaning_nkirs_age(None)
    X_fcz = timeseries_to_fcz(X_ts)
    return X_fcz, y

def _load_cmihbn_TD():
    X_ts, y = data_cleaning_cmi_td_age(None)
    X_fcz = timeseries_to_fcz(X_ts)
    return X_fcz, y

def _load_cmihbn_ADHD():
    X_ts, y = data_cleaning_cmi_age(None)
    X_fcz = timeseries_to_fcz(X_ts)
    return X_fcz, y

def _load_adhd200_TD():
    X_ts, y = data_cleaning_adhd200_td_age(None)
    X_fcz = timeseries_to_fcz(X_ts)
    return X_fcz, y

def _load_adhd200_ADHD():
    X_ts, y = data_cleaning_adhd200_age(None)
    X_fcz = timeseries_to_fcz(X_ts)
    return X_fcz, y

def _load_stanford_TD():
    X_ts, y = data_cleaning_stanford_td_age(None)
    X_fcz = timeseries_to_fcz(X_ts)
    X_fcz, y = _drop_nan_subjects(X_fcz, y)
    return X_fcz, y

def _load_stanford_ASD():
    X_ts, y = data_cleaning_stanford_autism_age(None)
    X_fcz = timeseries_to_fcz(X_ts)
    X_fcz, y = _drop_nan_subjects(X_fcz, y)
    return X_fcz, y

def _load_abide_TD():
    X_ts, y = data_cleaning_abide_td_age(None)
    X_fcz = timeseries_to_fcz(X_ts)
    return X_fcz, y

def _load_abide_ASD():
    X_ts, y = data_cleaning_abide_asd_age(None)
    X_fcz = timeseries_to_fcz(X_ts)
    return X_fcz, y

def _to_1d(a):
    a = np.asarray(a)
    return a.reshape(-1)

def _pearsonr(y_true, y_pred) -> float:
    """Pearson r with guards against degenerate cases."""
    y_true = _to_1d(y_true)
    y_pred = _to_1d(y_pred)
    if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])

def _pearsonr_stats(y_true, y_pred):
    """
    Return (r, r2, p) between y_true and y_pred.
    Safely handles degenerate cases.
    """
    y_true = _to_1d(y_true)
    y_pred = _to_1d(y_pred)
    if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan"), float("nan"), float("nan")
    r, p = stats.pearsonr(y_true, y_pred)
    return float(r**2), float(p)

def _fit_bias(y_true_td, y_pred_td) -> Tuple[float, float]:
    """
    Fit y_pred = a*y_true + b (least squares) on TD.
    Returns (a, b). Falls back to (1, 0) if degenerate.
    """
    y_true_td = _to_1d(y_true_td)
    y_pred_td = _to_1d(y_pred_td)
    if y_true_td.size < 2 or np.std(y_true_td) == 0:
        return 1.0, 0.0
    A = np.vstack([y_true_td, np.ones_like(y_true_td)]).T
    sol, *_ = np.linalg.lstsq(A, y_pred_td, rcond=None)
    a, b = float(sol[0]), float(sol[1])
    if not np.isfinite(a) or abs(a) < 1e-6: a = 1.0
    if not np.isfinite(b): b = 0.0
    return a, b

def _apply_bias(y_pred, a, b):
    """Apply correction: y_corr = (y_pred - b) / a."""
    y_pred = _to_1d(y_pred)
    if not np.isfinite(a) or abs(a) < 1e-6:
        return y_pred
    return (y_pred - b) / a


if __name__ == '__main__':
    
    MODEL_DIR = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/models/dev/ml_models/'
    
    #####################################
    FOLDS = range(5)
    FILENAME_PATTERN = "model_{model}_dev_age_{fold}.sav"
    # Perform classification and compute classification performance metrics
    MODEL_KEYMAP = {
    "linSVR": "linSVR",
    "KNN":    "KNN",
    "DT":     "DT",
    "LR":     "LR",
    "RC":     "RC",
    "LASSO":  "LASSO",
    "RF":     "RF",
    }
    regressors = []
    # linear SVM classifier → linear SVR
    regressors.append(('linSVR', svm.SVR(kernel='linear')))
    # KNN classifier → KNN regressor
    regressors.append(('KNN', neighbors.KNeighborsRegressor()))
    # Decision tree classifier → decision tree regressor
    regressors.append(('DT', tree.DecisionTreeRegressor()))
    # logistic regression (classification) → ordinary least squares regression
    regressors.append(('LR', linear_model.LinearRegression()))
    # ridge classifier → ridge regression
    regressors.append(('RC', linear_model.Ridge(alpha=0.5)))
    # L1-penalized logistic → LASSO regression
    regressors.append(('LASSO', linear_model.Lasso(alpha=0.1)))
    # regressors.append(('ELNet', linear_model.ElasticNet(alpha=1.0, l1_ratio=0.5)))
    # random forest classifier → random forest regressor
    regressors.append(('RF', ensemble.RandomForestRegressor()))
    # get features and labels (0: male, 1: female)
    _model_cache: Dict[str, List[object]] = {}
    def _load_fold_models(model_key: str) -> List[object]:
        """
        Load the 5 CV models for a given model_key (e.g., 'DT', 'RF', ...).
        Cached after first load.
        """
        if model_key in _model_cache:
            return _model_cache[model_key]

        models = []
        for f in FOLDS:
            fname = os.path.join(MODEL_DIR, FILENAME_PATTERN.format(model=model_key, fold=f))
            if not os.path.exists(fname):
                raise FileNotFoundError(f"Missing model file: {fname}")
            try:
                m = joblib.load(fname)
            except Exception:
                with open(fname, "rb") as fp:
                    m = pickle.load(fp)
            models.append(m)

        _model_cache[model_key] = models
        return models

    def _predict_5fold_mean(model_key: str, X: np.ndarray) -> np.ndarray:
        """Predict with each of the 5 fold models and average per-sample."""
        models = _load_fold_models(model_key)
        preds = []
        for m in models:
            preds.append(_to_1d(m.predict(X)))
        return np.mean(np.stack(preds, axis=0), axis=0)   
    
    def _eval_td_self(model_key: str, td_loader) -> float:
        """
        External TD performance: fit bias on the dataset's own TD and apply to itself.
        Returns Pearson r(actual, corrected_pred).
        """
        X_td, y_td = td_loader()
        y_hat = _predict_5fold_mean(model_key, X_td)
        a, b = _fit_bias(y_td, y_hat)
        y_hat_corr = _apply_bias(y_hat, a, b)
        return _pearsonr_stats(y_td, y_hat_corr)

    def _eval_with_td_correction(model_key: str, td_loader, target_loader) -> float:
        """
        Fit bias on dataset TD, apply to a target split (ADHD/ASD).
        Returns Pearson r(actual, corrected_pred) on target.
        """
        X_td, y_td = td_loader()
        y_td_hat = _predict_5fold_mean(model_key, X_td)
        a, b = _fit_bias(y_td, y_td_hat)

        X_tgt, y_tgt = target_loader()
        y_tgt_hat = _predict_5fold_mean(model_key, X_tgt)
        y_tgt_hat_corr = _apply_bias(y_tgt_hat, a, b)
        return _pearsonr_stats(y_tgt, y_tgt_hat_corr)

   
    def run_external_eval_saved_models():
        print("\n=== Starting external evaluation with saved 5-CV HCP-Dev models ===")
        # Normalize your `regressors` into a list of names that map to saved files
        if not isinstance(regressors, list) or len(regressors) == 0:
            raise RuntimeError("`regressors` must be a non-empty list.")

        model_names = []
        for item in regressors:
            # your `regressors` are (name, estimator) tuples
            name = item[0] if (isinstance(item, tuple) and len(item) == 2) else getattr(item, "__class__", type(item)).__name__
            if name not in MODEL_KEYMAP:
                raise KeyError(f"Unknown model name '{name}' in `regressors`. "
                               f"Add it to MODEL_KEYMAP or rename to one of {list(MODEL_KEYMAP)}.")
            model_names.append(name)

            # Datasets we’ll report (row order)
            dataset_rows = [
                ("NKI-RS TD",      lambda key: _eval_td_self(key, _load_nkirs_TD)),
                ("CMI-HBN TD",     lambda key: _eval_td_self(key, _load_cmihbn_TD)),
                ("CMI-HBN ADHD",   lambda key: _eval_with_td_correction(key, _load_cmihbn_TD,  _load_cmihbn_ADHD)),
                ("ADHD-200 TD",    lambda key: _eval_td_self(key, _load_adhd200_TD)),
                ("ADHD-200 ADHD",  lambda key: _eval_with_td_correction(key, _load_adhd200_TD, _load_adhd200_ADHD)),
                ("Stanford ASD",   lambda key: _eval_with_td_correction(key, _load_stanford_TD, _load_stanford_ASD)),
                ("ABIDE ASD",      lambda key: _eval_with_td_correction(key, _load_abide_TD,    _load_abide_ASD)),
            ]

            # results_by_dataset[dataset][model_name] = (r2, p)
            results_by_dataset = {row[0]: {} for row in dataset_rows}

            print(f"\n=== Running evaluations for {len(model_names)} regressors across {len(dataset_rows)} datasets ===")
            for name in model_names:
                key = MODEL_KEYMAP[name]
                print(f"\n--- Evaluating regressor: {name} ---")
                for ds_name, fn in dataset_rows:
                    try:
                        r2, p = fn(key)
                    except FileNotFoundError as e:
                        print(f"[warn] {name} @ {ds_name}: {e}"); r2 = p = float("nan")
                    except Exception as e:
                        print(f"[warn] {name} @ {ds_name}: {e}"); r2 = p = float("nan")
                    results_by_dataset[ds_name][name] = (r2, p)

                # compact one-liner per model
                def _get(ds): 
                    r2, p = results_by_dataset[ds][name]
                    return f"{r2:.3f} (p={p:.2e})" if np.isfinite(r2) and np.isfinite(p) else "nan"
                print(
                    "[summary] "
                    f"{name:7s} | "
                    f"NKI-RS TD: {_get('NKI-RS TD')} | "
                    f"CMI-HBN TD: {_get('CMI-HBN TD')}, ADHD: {_get('CMI-HBN ADHD')} | "
                    f"ADHD-200 TD: {_get('ADHD-200 TD')}, ADHD: {_get('ADHD-200 ADHD')} | "
                    f"Stanford ASD: {_get('Stanford ASD')} | "
                    f"ABIDE ASD: {_get('ABIDE ASD')}"
                )

            # ---------- Pretty dataset-centric Markdown table ----------
            # Columns: Dataset | <Model1 r² (p)> | <Model2 r² (p)> | ...
            def _cell(r2, p):
                return f"{r2:.3f} ({p:.2e})" if np.isfinite(r2) and np.isfinite(p) else "nan"

            header = ["Dataset"] + model_names
            # compute column widths
            col_widths = [len(h) for h in header]
            for ds_name in results_by_dataset:
                # longest cell across models
                for i, m in enumerate(model_names, start=1):
                    r2, p = results_by_dataset[ds_name].get(m, (float("nan"), float("nan")))
                    col_widths[i] = max(col_widths[i], len(_cell(r2, p)))
                col_widths[0] = max(col_widths[0], len(ds_name))

            def _mk_row(cells):
                return "| " + " | ".join(s.ljust(w) for s, w in zip(cells, col_widths)) + " |"

            print("\n=== Brain Age: r² (p) by Dataset × Model ===")
            print(_mk_row(header))
            print("| " + " | ".join("-"*w for w in col_widths) + " |")
            for ds_name, _ in dataset_rows:
                row = [ds_name]
                for m in model_names:
                    r2, p = results_by_dataset[ds_name].get(m, (float("nan"), float("nan")))
                    row.append(_cell(r2, p))
                print(_mk_row(row))

            # ---------- Save a CSV (wide format, matches the printed table) ----------
            # Each cell is the string "r2 (p)"
            import csv
            csv_path = "external_eval_r2_p_wide.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for ds_name, _ in dataset_rows:
                    row = [ds_name]
                    for m in model_names:
                        r2, p = results_by_dataset[ds_name].get(m, (float("nan"), float("nan")))
                        row.append(_cell(r2, p))
                    writer.writerow(row)
            print(f"\nSaved CSV: {csv_path}")
        '''
        columns = [
            "Model",
            "NKI-RS TD",
            "CMI-HBN TD", "CMI-HBN ADHD",
            "ADHD-200 TD", "ADHD-200 ADHD",
            "Stanford ASD",
            "ABIDE ASD",
        ]

        rows = []
        for name in model_names:
            key = MODEL_KEYMAP[name]
            print(f"\n--- Evaluating regressor: {name} ---")
            try:
                print("  [1] Loading models and computing predictions...")
                # External TD (fit+apply on own TD)
                r2_nki_td, p_nki       = _eval_td_self(key, _load_nkirs_TD)
                r2_cmihbn_td, p_cmihbn_td    = _eval_td_self(key, _load_cmihbn_TD)
                r2_adhd200_td, p_adhd200_td   = _eval_td_self(key, _load_adhd200_TD)
                print("  [2] Evaluating ADHD cohorts (bias from TD → apply to ADHD)...")
                # ADHD (fit on TD, apply to ADHD)
                r2_cmihbn_adhd, p_cmihbn_adhd  = _eval_with_td_correction(key, _load_cmihbn_TD,  _load_cmihbn_ADHD)
                r2_adhd200_adhd, p_adhd200_adhd = _eval_with_td_correction(key, _load_adhd200_TD, _load_adhd200_ADHD)
                print("  [3] Evaluating ASD cohorts (bias from TD → apply to ASD)...")
                # ASD (fit on TD, apply to ASD)
                r2_stanford_asd, p_stanford_asd = _eval_with_td_correction(key, _load_stanford_TD, _load_stanford_ASD)
                r2_abide_asd, p_abide_asd    = _eval_with_td_correction(key, _load_abide_TD,    _load_abide_ASD)

            except FileNotFoundError as e:
                print(f"[warn] {name}: {e}")
                r2_nki_td = r2_cmihbn_td = r2_adhd200_td = float("nan")
                r2_cmihbn_adhd = r2_adhd200_adhd = float("nan")
                r2_stanford_asd = r2_abide_asd = float("nan")
            except Exception as e:
                print(f"[warn] {name} evaluation error: {e}")
                r2_nki_td = r2_cmihbn_td = r2_adhd200_td = float("nan")
                r2_cmihbn_adhd = r2_adhd200_adhd = float("nan")
                r2_stanford_asd = r2_abide_asd = float("nan")

            rows.append({
                "Model": name,
                "NKI-RS TD": r2_nki_td,
                "CMI-HBN TD": r2_cmihbn_td, "CMI-HBN ADHD": r_cmihbn_adhd,
                "ADHD-200 TD": r2_adhd200_td, "ADHD-200 ADHD": r_adhd200_adhd,
                "Stanford ASD": r2_stanford_asd,
                "ABIDE ASD": r2_abide_asd,
            })

        print("\n=== Finished evaluation. Printing summary table... ===")
        # Pretty print Markdown + save CSV
        def _fmt(v):
            if isinstance(v, float):
                if math.isnan(v): return "nan"
                return f"{v:.3f}"
            return str(v)

        col_widths = [len(c) for c in columns]
        for r in rows:
            for i, c in enumerate(columns):
                col_widths[i] = max(col_widths[i], len(_fmt(r.get(c, ""))))
    
        def _mk_row(cells):
            return "| " + " | ".join(s.ljust(w) for s, w in zip(cells, col_widths)) + " |"

        print("\n=== Brain Age: External Evaluation with Saved 5-CV Models (r^2) ===")
        print(_mk_row(columns))
        print("| " + " | ".join("-"*w for w in col_widths) + " |")
        for r in rows:
            print(_mk_row([_fmt(r.get(c, "")) for c in columns]))

        csv_path = "external_eval_summary.csv"
        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for r in rows:
                writer.writerow({c: r.get(c, "") for c in columns})
        print(f"\nSaved CSV: {csv_path}")
        '''
    run_external_eval_saved_models() 
