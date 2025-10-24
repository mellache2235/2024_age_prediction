# ============================================================
# benchmark_ml_brain_age_Yscaled.py
# ------------------------------------------------------------
# Train classical ML regressors on HCP-Development:
#  • SCALE y (age) with StandardScaler fit on HCP-Dev only
#  • DO NOT scale X (FC edges); only impute NaNs
#  • Predict standardized age, inverse-transform to years
#  • TD-derived bias correction per cohort (apply to TD & clinical)
#  • Save tidy CSV of R^2 (and optional MAE) per Model × Cohort × Group
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd

# sklearn
from sklearn import svm, neighbors, tree, linear_model, ensemble
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import pickle
# ============================================================
# 0) Utilities
# ============================================================

def sanitize_X(X: np.ndarray) -> np.ndarray:
    """Replace ±Inf with 0; keep NaNs (imputer will handle)."""
    X = np.asarray(X, float)
    X[np.isposinf(X)] = 0.0
    X[np.isneginf(X)] = 0.0
    return X

def _finite_mask(*arrs):
    m = np.ones(len(arrs[0]), dtype=bool)
    for a in arrs:
        a = np.asarray(a, float)
        m &= np.isfinite(a)
    return m

def fit_td_bias(age, pred_years):
    """Fit TD bias in YEARS: BAG = (pred - age) = β*age + α."""
    age = np.asarray(age, float); pred = np.asarray(pred_years, float)
    bag = pred - age
    m = _finite_mask(age, bag)
    if m.sum() < 3:
        return 0.0, 0.0, int(m.sum())
    beta, alpha = np.polyfit(age[m], bag[m], 1)
    return float(beta), float(alpha), int(m.sum())

def apply_td_bias(age, pred_years, beta, alpha):
    """Apply TD bias in YEARS to predictions → (pred_corr_years, bag_corr_years)."""
    age = np.asarray(age, float); pred = np.asarray(pred_years, float)
    pred_corr = pred - (beta*age + alpha)
    bag_corr  = pred_corr - age
    return pred_corr, bag_corr

# ============================================================
# 1) YOUR MODELS (no X scaling; impute only)
# ============================================================

def mkpipe(estimator):
    # Only imputer; NO StandardScaler on X
    return Pipeline([('impute', SimpleImputer(strategy='median')),
                     ('est', estimator)])

regressors = [
    ('linSVR', svm.SVR(kernel='linear', C=10.0, epsilon=0.1)),
    ('KNN',    neighbors.KNeighborsRegressor(n_neighbors=10)),
    ('DT',     tree.DecisionTreeRegressor(max_depth=12, min_samples_leaf=25, random_state=27)),
    ('LR',     linear_model.LinearRegression()),
    ('RC',     linear_model.Ridge(alpha=0.5)),
    ('LASSO',  linear_model.Lasso(alpha=0.1, max_iter=10000)),
    ('RF',     ensemble.RandomForestRegressor(n_estimators=600, random_state=27, n_jobs=-1)),
]
regressors = [(name, mkpipe(est)) for name, est in regressors]

# ============================================================
# 2) DATA SECTION (fill these with your arrays)
#    X_*: (N, P) feature matrices (FC edges)
#    y_*: (N,) ages in YEARS
# ============================================================
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


def data_cleaning_nkirs_age():
    saved_data_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/nki_age_cog_dev_wIDs/fold_0.bin'
    X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(saved_data_path)

    X_total = np.concatenate((X_train,X_valid))
    Y_total = np.concatenate((Y_train,Y_valid))

    no_subjs, no_ts, no_rois = X_total.shape
    data_fcz = np.empty((no_subjs, int(no_rois * (no_rois - 1) / 2)))
    print('data_fcz dimension {}'.format(data_fcz.shape))

    for subj in range(no_subjs):
        # print(subj)
        x_subj = X_total[subj, :, :]
        df_subj = pd.DataFrame(x_subj)
        fc_subj = df_subj.corr('pearson')  # get correlation matrix
        fc_subj = fc_subj.to_numpy()
        # get upper tri elements of the FC matrix and apply fisher z transformation
        data_fcz[subj, :] = np.arctanh(fc_subj[np.triu_indices(fc_subj.shape[0], k=1)])

    return data_fcz, Y_total

def data_cleaning_hcp_dev_age():
    saved_data_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/hcp_dev_age_five_fold/fold_0.bin'
    X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(saved_data_path)

    X_total = np.concatenate((X_train,X_valid))
    Y_total = np.concatenate((Y_train,Y_valid))

    no_subjs, no_ts, no_rois = X_total.shape
    data_fcz = np.empty((no_subjs, int(no_rois * (no_rois - 1) / 2)))
    print('data_fcz dimension {}'.format(data_fcz.shape))

    for subj in range(no_subjs):
        # print(subj)
        x_subj = X_total[subj, :, :]
        df_subj = pd.DataFrame(x_subj)
        fc_subj = df_subj.corr('pearson')  # get correlation matrix
        fc_subj = fc_subj.to_numpy()
        # get upper tri elements of the FC matrix and apply fisher z transformation
        data_fcz[subj, :] = np.arctanh(fc_subj[np.triu_indices(fc_subj.shape[0], k=1)])

    return data_fcz, Y_total


def data_cleaning_adhd200_td_age():
    saved_data_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/adhd200_regression_age_TD_wIDs/fold_0.bin'
    X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(saved_data_path)

    X_total = np.concatenate((X_train,X_valid))
    Y_total = np.concatenate((Y_train,Y_valid))

    no_subjs, no_ts, no_rois = X_total.shape
    data_fcz = np.empty((no_subjs, int(no_rois * (no_rois - 1) / 2)))
    print('data_fcz dimension {}'.format(data_fcz.shape))

    for subj in range(no_subjs):
        # print(subj)
        x_subj = X_total[subj, :, :]
        df_subj = pd.DataFrame(x_subj)
        fc_subj = df_subj.corr('pearson')  # get correlation matrix
        fc_subj = fc_subj.to_numpy()
        # get upper tri elements of the FC matrix and apply fisher z transformation
        data_fcz[subj, :] = np.arctanh(fc_subj[np.triu_indices(fc_subj.shape[0], k=1)])

    return data_fcz, Y_total

def data_cleaning_cmi_td_age():
    saved_data_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/cmihbn_age_TD/fold_0.bin'
    X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(saved_data_path)

    X_total = np.concatenate((X_train,X_valid))
    Y_total = np.concatenate((Y_train,Y_valid))

    no_subjs, no_ts, no_rois = X_total.shape
    data_fcz = np.empty((no_subjs, int(no_rois * (no_rois - 1) / 2)))
    print('data_fcz dimension {}'.format(data_fcz.shape))

    for subj in range(no_subjs):
        # print(subj)
        x_subj = X_total[subj, :, :]
        df_subj = pd.DataFrame(x_subj)
        fc_subj = df_subj.corr('pearson')  # get correlation matrix
        fc_subj = fc_subj.to_numpy()
        # get upper tri elements of the FC matrix and apply fisher z transformation
        data_fcz[subj, :] = np.arctanh(fc_subj[np.triu_indices(fc_subj.shape[0], k=1)])

    return data_fcz, Y_total

def data_cleaning_abide_asd_age():
    saved_data_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/abide_asd_asd_dev_age_wIDs/fold_0.bin'
    X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(saved_data_path)

    X_total = np.concatenate((X_train,X_valid))
    Y_total = np.concatenate((Y_train,Y_valid))

    no_subjs, no_ts, no_rois = X_total.shape
    data_fcz = np.empty((no_subjs, int(no_rois * (no_rois - 1) / 2)))
    print('data_fcz dimension {}'.format(data_fcz.shape))

    for subj in range(no_subjs):
        # print(subj)
        x_subj = X_total[subj, :, :]
        df_subj = pd.DataFrame(x_subj)
        fc_subj = df_subj.corr('pearson')  # get correlation matrix
        fc_subj = fc_subj.to_numpy()
        # get upper tri elements of the FC matrix and apply fisher z transformation
        data_fcz[subj, :] = np.arctanh(fc_subj[np.triu_indices(fc_subj.shape[0], k=1)])

    return data_fcz, Y_total

def data_cleaning_abide_td_age():
    saved_data_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/abide_asd_td_dev_age/fold_0.bin'
    X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(saved_data_path)

    X_total = np.concatenate((X_train,X_valid))
    Y_total = np.concatenate((Y_train,Y_valid))

    no_subjs, no_ts, no_rois = X_total.shape
    data_fcz = np.empty((no_subjs, int(no_rois * (no_rois - 1) / 2)))
    print('data_fcz dimension {}'.format(data_fcz.shape))

    for subj in range(no_subjs):
        # print(subj)
        x_subj = X_total[subj, :, :]
        df_subj = pd.DataFrame(x_subj)
        fc_subj = df_subj.corr('pearson')  # get correlation matrix
        fc_subj = fc_subj.to_numpy()
        # get upper tri elements of the FC matrix and apply fisher z transformation
        data_fcz[subj, :] = np.arctanh(fc_subj[np.triu_indices(fc_subj.shape[0], k=1)])

    return data_fcz, Y_total

def data_cleaning_stanford_autism_age():
    saved_data_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/stanford_autism_age_wIDS/fold_0.bin'
    X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(saved_data_path)

    X_total = np.concatenate((X_train,X_valid))
    Y_total = np.concatenate((Y_train,Y_valid))

    no_subjs, no_ts, no_rois = X_total.shape
    data_fcz = np.empty((no_subjs, int(no_rois * (no_rois - 1) / 2)))
    print('data_fcz dimension {}'.format(data_fcz.shape))

    for subj in range(no_subjs):
        # print(subj)
        x_subj = X_total[subj, :, :]
        df_subj = pd.DataFrame(x_subj)
        fc_subj = df_subj.corr('pearson')  # get correlation matrix
        fc_subj = fc_subj.to_numpy()
        # get upper tri elements of the FC matrix and apply fisher z transformation
        data_fcz[subj, :] = np.arctanh(fc_subj[np.triu_indices(fc_subj.shape[0], k=1)])

    return data_fcz, Y_total

def data_cleaning_stanford_td_age():
    saved_data_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/stanford_autism_age_TD_wIDS/fold_0.bin'
    X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(saved_data_path)

    X_total = np.concatenate((X_train,X_valid))
    Y_total = np.concatenate((Y_train,Y_valid))

    no_subjs, no_ts, no_rois = X_total.shape
    data_fcz = np.empty((no_subjs, int(no_rois * (no_rois - 1) / 2)))
    print('data_fcz dimension {}'.format(data_fcz.shape))

    for subj in range(no_subjs):
        # print(subj)
        x_subj = X_total[subj, :, :]
        df_subj = pd.DataFrame(x_subj)
        fc_subj = df_subj.corr('pearson')  # get correlation matrix
        fc_subj = fc_subj.to_numpy()
        # get upper tri elements of the FC matrix and apply fisher z transformation
        data_fcz[subj, :] = np.arctanh(fc_subj[np.triu_indices(fc_subj.shape[0], k=1)])

    return data_fcz, Y_total

def data_cleaning_adhd200_age():
    saved_data_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/adhd200_regression_age_ADHD_wIDs/fold_0.bin'

    X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(saved_data_path)

    X_total = np.concatenate((X_train,X_valid))
    Y_total = np.concatenate((Y_train,Y_valid))

    no_subjs, no_ts, no_rois = X_total.shape
    data_fcz = np.empty((no_subjs, int(no_rois * (no_rois - 1) / 2)))
    print('data_fcz dimension {}'.format(data_fcz.shape))

    for subj in range(no_subjs):
        # print(subj)
        x_subj = X_total[subj, :, :]
        df_subj = pd.DataFrame(x_subj)
        fc_subj = df_subj.corr('pearson')  # get correlation matrix
        fc_subj = fc_subj.to_numpy()
        # get upper tri elements of the FC matrix and apply fisher z transformation
        data_fcz[subj, :] = np.arctanh(fc_subj[np.triu_indices(fc_subj.shape[0], k=1)])

    return data_fcz, Y_total

def data_cleaning_cmi_age():
    saved_data_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/adhd_cmihbn_age_ADHD_wIDS/fold_0.bin'

    X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(saved_data_path)

    X_total = np.concatenate((X_train,X_valid))
    Y_total = np.concatenate((Y_train,Y_valid))

    no_subjs, no_ts, no_rois = X_total.shape
    data_fcz = np.empty((no_subjs, int(no_rois * (no_rois - 1) / 2)))
    print('data_fcz dimension {}'.format(data_fcz.shape))

    for subj in range(no_subjs):
        # print(subj)
        x_subj = X_total[subj, :, :]
        df_subj = pd.DataFrame(x_subj)
        fc_subj = df_subj.corr('pearson')  # get correlation matrix
        fc_subj = fc_subj.to_numpy()
        # get upper tri elements of the FC matrix and apply fisher z transformation
        data_fcz[subj, :] = np.arctanh(fc_subj[np.triu_indices(fc_subj.shape[0], k=1)])

    return data_fcz, Y_total

# HCP-Development (training)
X_dev,y_dev = data_cleaning_hcp_dev_age()

# External TD datasets
X_nki_td,      y_nki      = data_cleaning_nkirs_age()
X_hbn_td,      y_hbn_td   = data_cleaning_cmi_td_age()
X_adhd200_td,  y_adhd200_td = data_cleaning_adhd200_td_age()
# (optional ABIDE/Stanford TD if you have them)
X_abide_td,    y_abide_td = data_cleaning_abide_td_age()
X_stan_td,     y_stan_td  = data_cleaning_stanford_td_age()

# Clinical datasets
# ADHD
X_hbn_adhd,      y_hbn_adhd      = data_cleaning_cmi_age()
X_adhd200_adhd,  y_adhd200_adhd  = data_cleaning_adhd200_age()
# ASD
X_abide_asd,     y_abide_asd     = data_cleaning_abide_asd_age()
X_stan_asd,      y_stan_asd      = data_cleaning_stanford_autism_age()

# Sanitize feature matrices (keep NaNs for imputer)
X_dev        = sanitize_X(X_dev)
X_nki_td     = sanitize_X(X_nki_td)
X_hbn_td     = sanitize_X(X_hbn_td)
X_adhd200_td = sanitize_X(X_adhd200_td)
X_hbn_adhd   = sanitize_X(X_hbn_adhd)
X_adhd200_adhd = sanitize_X(X_adhd200_adhd)
X_abide_asd  = sanitize_X(X_abide_asd)
X_stan_asd   = sanitize_X(X_stan_asd)
X_abide_td = sanitize_X(X_abide_td)
X_stan_td = sanitize_X(X_stan_td)

# ============================================================
# 3) Results logger (R^2 / MAE in YEARS)
# ============================================================

class BenchmarkLogger:
    def __init__(self, model_name: str, save_mae: bool = False):
        self.model = model_name
        self.save_mae = save_mae
        self.td_bias = {}   # cohort -> (beta, alpha, n_fit)
        self.rows   = []    # compact results
        self.debug  = []    # raw vs corr + bias info

    def _metrics(self, y_true_years, y_pred_years):
        m = _finite_mask(y_true_years, y_pred_years)
        if m.sum() < 2:
            return np.nan, np.nan, m.sum(), np.nan, np.nan
        r2  = r2_score(y_true_years[m], y_pred_years[m])
        mae = mean_absolute_error(y_true_years[m], y_pred_years[m]) if self.save_mae else np.nan
        amin, amax = float(np.min(y_true_years[m])), float(np.max(y_true_years[m]))
        return r2, mae, m.sum(), amin, amax

    def add_td(self, cohort: str, ages_years: np.ndarray, preds_years: np.ndarray):
        r2_raw, mae_raw, n_raw, amin, amax = self._metrics(ages_years, preds_years)
        beta, alpha, n_fit = fit_td_bias(ages_years, preds_years)
        preds_corr, bag_corr = apply_td_bias(ages_years, preds_years, beta, alpha)
        r2_cor, mae_cor, n_cor, _, _ = self._metrics(ages_years, preds_corr)

        self.td_bias[cohort] = (beta, alpha, n_fit)
        row = {"Model": self.model, "Cohort": cohort, "Group": "TD",
               "N": n_cor, "AgeMin": amin, "AgeMax": amax, "R2": r2_cor}
        if self.save_mae: row["MAE"] = mae_cor
        self.rows.append(row)

        self.debug.append({"Model": self.model, "Cohort": cohort, "Group": "TD",
                           "N": n_cor, "AgeMin": amin, "AgeMax": amax,
                           "R2_raw": r2_raw, "R2_corr": r2_cor,
                           **({"MAE_raw": mae_raw, "MAE_corr": mae_cor} if self.save_mae else {}),
                           "beta": beta, "alpha": alpha, "N_fit_bias": n_fit})

    def add_clinical(self, cohort: str, group: str,
                     ages_years: np.ndarray, preds_years: np.ndarray,
                     td_for_bias: tuple[str, np.ndarray, np.ndarray] | None = None):
        if cohort not in self.td_bias:
            if td_for_bias is None:
                raise ValueError(f"TD bias for '{cohort}' unknown. Add TD first or pass td_for_bias=(cohort, ages_td, preds_td).")
            c_name, ages_td, preds_td = td_for_bias
            assert c_name == cohort
            beta, alpha, n_fit = fit_td_bias(ages_td, preds_td)
            self.td_bias[cohort] = (beta, alpha, n_fit)
        else:
            beta, alpha, n_fit = self.td_bias[cohort]

        r2_raw, mae_raw, n_raw, amin, amax = self._metrics(ages_years, preds_years)
        preds_corr, bag_corr = apply_td_bias(ages_years, preds_years, beta, alpha)
        r2_cor, mae_cor, n_cor, _, _ = self._metrics(ages_years, preds_corr)

        row = {"Model": self.model, "Cohort": cohort, "Group": group,
               "N": n_cor, "AgeMin": amin, "AgeMax": amax, "R2": r2_cor}
        if self.save_mae: row["MAE"] = mae_cor
        self.rows.append(row)

        self.debug.append({"Model": self.model, "Cohort": cohort, "Group": group,
                           "N": n_cor, "AgeMin": amin, "AgeMax": amax,
                           "R2_raw": r2_raw, "R2_corr": r2_cor,
                           **({"MAE_raw": mae_raw, "MAE_corr": mae_cor} if self.save_mae else {}),
                           "beta(TD)": beta, "alpha(TD)": alpha, "N_fit_bias(TD)": n_fit})

    def to_dataframe(self) -> pd.DataFrame:
        cols = ["Model","Cohort","Group","N","AgeMin","AgeMax","R2"]
        if self.save_mae: cols.append("MAE")
        return pd.DataFrame(self.rows)[cols].sort_values(["Model","Cohort","Group"]).reset_index(drop=True)

    def debug_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.debug).sort_values(["Model","Cohort","Group"]).reset_index(drop=True)

# ============================================================
# 4) RUN: y-scaling on HCP-Dev ONLY; no X scaling
# ============================================================

def run_benchmark(save_csv="./ml_benchmark_r2_by_dataset.csv", save_mae=False):
    # ---- y-scaler fit on HCP-Dev ONLY ----
    y_scaler = StandardScaler(with_mean=True, with_std=True)
    y_dev_std = y_scaler.fit_transform(np.asarray(y_dev, float).reshape(-1,1)).ravel()

    # ---- sanitize X matrices (replace ±inf; NaNs left for imputer) ----
    def sX(X): return sanitize_X(X)

    # ---- assemble datasets dictionary (unscaled X, raw ages in YEARS) ----
    sets = {}
    def add_set(key, X, y): sets[key] = (sX(X), np.asarray(y, float))

    add_set("NKI-RS_TD",        X_nki_td,       y_nki)
    add_set("CMI-HBN_TD",       X_hbn_td,       y_hbn_td)
    add_set("ADHD-200_TD",      X_adhd200_td,   y_adhd200_td)
    try: add_set("ABIDE_TD",    X_abide_td,     y_abide_td)
    except NameError: pass
    try: add_set("Stanford_TD", X_stan_td,      y_stan_td)
    except NameError: pass

    add_set("CMI-HBN_ADHD",     X_hbn_adhd,     y_hbn_adhd)
    add_set("ADHD-200_ADHD",    X_adhd200_adhd, y_adhd200_adhd)
    add_set("ABIDE_ASD",        X_abide_asd,    y_abide_asd)
    add_set("Stanford_ASD",     X_stan_asd,     y_stan_asd)

    # ---- train & evaluate models ----
    all_rows, all_dbg = [], []
    for name, est in regressors:
        print(f"\n=== MODEL: {name} ===")

        # train on HCP-Dev with standardized y
        est.fit(sX(X_dev), y_dev_std)

        log = BenchmarkLogger(model_name=name, save_mae=save_mae)

        # TD first per cohort (fit bias in YEARS)
        for cohort_key in ["NKI-RS_TD","CMI-HBN_TD","ADHD-200_TD","ABIDE_TD","Stanford_TD"]:
            if cohort_key not in sets: continue
            X_td, y_td = sets[cohort_key]
            # predict standardized, inverse-transform to YEARS
            yhat_td_std = est.predict(X_td)
            yhat_td_yrs = y_scaler.inverse_transform(yhat_td_std.reshape(-1,1)).ravel()
            cohort = cohort_key.replace("_TD","").replace("_","-")
            log.add_td(cohort=cohort, ages_years=y_td, preds_years=yhat_td_yrs)

        # Clinical ADHD (use same cohort TD bias)
        for cohort_key in ["CMI-HBN_ADHD","ADHD-200_ADHD"]:
            if cohort_key not in sets: continue
            X_c, y_c = sets[cohort_key]
            yhat_std = est.predict(X_c)
            yhat_yrs = y_scaler.inverse_transform(yhat_std.reshape(-1,1)).ravel()
            cohort = cohort_key.replace("_ADHD","")
            log.add_clinical(cohort=cohort, group="ADHD", ages_years=y_c, preds_years=yhat_yrs)

        # Clinical ASD (ABIDE/Stanford) — supply TD bias if not already added
        for cohort_key in ["ABIDE_ASD","Stanford_ASD"]:
            if cohort_key not in sets: continue
            X_c, y_c = sets[cohort_key]
            yhat_std = est.predict(X_c)
            yhat_yrs = y_scaler.inverse_transform(yhat_std.reshape(-1,1)).ravel()
            cohort = cohort_key.replace("_ASD","")
            td_key = cohort + "_TD"
            if cohort in log.td_bias:
                log.add_clinical(cohort=cohort, group="ASD", ages_years=y_c, preds_years=yhat_yrs)
            else:
                if td_key not in sets:
                    raise ValueError(f"TD set for {cohort} not provided; needed for ASD bias.")
                X_td, y_td = sets[td_key]
                yhat_td_std = est.predict(X_td)
                yhat_td_yrs = y_scaler.inverse_transform(yhat_td_std.reshape(-1,1)).ravel()
                log.add_clinical(cohort=cohort, group="ASD", ages_years=y_c, preds_years=yhat_yrs,
                                 td_for_bias=(cohort, y_td, yhat_td_yrs))

        df_model = log.to_dataframe()
        df_dbg   = log.debug_dataframe()
        all_rows.append(df_model); all_dbg.append(df_dbg)

    df_all = pd.concat(all_rows, ignore_index=True)
    df_dbg = pd.concat(all_dbg,  ignore_index=True)

    Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(save_csv, index=False)
    df_dbg.to_csv(save_csv.replace(".csv","_debug.csv"), index=False)
    print(f"\n[saved] {save_csv}")
    print(f"[saved] {save_csv.replace('.csv','_debug.csv')}")

    # quick view
    print("\nR^2 by Cohort × Group × Model:")
    print(df_all.pivot_table(index=["Cohort","Group"], columns="Model", values="R2", aggfunc="first").round(3))

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # Ensure DATA SECTION arrays are set above
    run_benchmark(save_csv="./ml_benchmark_r2_by_dataset.csv", save_mae=False)

