# ============================================
# benchmark_ml_brain_age.py
# --------------------------------------------
# Train classical ML models on HCP-Development,
# evaluate on external TD / ADHD / ASD datasets,
# apply age-bias correction (per cohort TD),
# and save a tidy CSV:  one row per (Model × Dataset)
# with R^2 (and optional MAE).
#
# Copy this whole file into your project and:
#  1) Fill the "DATA SECTION" with your feature matrices.
#  2) Run:  python benchmark_ml_brain_age.py
# ============================================

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pickle
from sklearn import svm, neighbors, tree, linear_model, ensemble
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# -------------------------------------------------
# ======= BIAS CORRECTION (TD-derived) =============
# -------------------------------------------------

def _finite_mask(*arrs):
    m = np.ones(len(arrs[0]), dtype=bool)
    for a in arrs:
        a = np.asarray(a, float)
        m &= np.isfinite(a)
    return m

def _fit_bias_params(age, pred):
    """
    Fit linear age bias on TD of a cohort:
    BAG = pred - age = β * age + α
    Returns (β, α). If not enough data, returns (0, 0).
    """
    age = np.asarray(age, float)
    pred = np.asarray(pred, float)
    bag = pred - age
    m = _finite_mask(age, bag)
    if m.sum() < 3:
        return 0.0, 0.0
    beta, alpha = np.polyfit(age[m], bag[m], 1)
    return float(beta), float(alpha), int(m.sum())

def _apply_bias(age, pred, beta, alpha):
    """
    Apply cohort TD bias to predictions:
    pred_corr = pred - (β * age + α)
    BAG_corr  = pred_corr - age
    """
    age = np.asarray(age, float)
    pred = np.asarray(pred, float)
    pred_corr = pred - (beta * age + alpha)
    bag_corr = pred_corr - age
    return pred_corr, bag_corr

# -------------------------------------------------
# ======= RESULTS LOGGER ==========================
# -------------------------------------------------
class BenchmarkLogger:
    def __init__(self, model_name: str, save_mae: bool = False):
        self.model    = model_name
        self.save_mae = save_mae
        self._td_bias = {}     # cohort -> (β, α, n_fit)
        self._rows    = []     # compact table
        self._debug   = []     # raw vs corr + β,α

    def _metrics(self, y_true, y_pred):
        m = _finite_mask(y_true, y_pred)
        if m.sum() < 2:
            return np.nan, np.nan, m.sum(), np.nan, np.nan
        r2  = r2_score(y_true[m], y_pred[m])
        mae = mean_absolute_error(y_true[m], y_pred[m]) if self.save_mae else np.nan
        return r2, mae, m.sum(), float(np.min(y_true[m])), float(np.max(y_true[m]))

    def add_td(self, cohort: str, ages: np.ndarray, preds: np.ndarray):
        # raw
        r2_raw, mae_raw, n_raw, amin, amax = self._metrics(ages, preds)
        beta, alpha, n_fit = _fit_bias_params(ages, preds)
        # corrected
        preds_corr, _bag_corr = _apply_bias(ages, preds, beta, alpha)
        r2_cor, mae_cor, n_cor, _, _ = self._metrics(ages, preds_corr)

        # store bias and rows
        self._td_bias[cohort] = (beta, alpha, n_fit)
        self._rows.append({"Model": self.model, "Cohort": cohort, "Group": "TD",
                           "N": n_cor, "AgeMin": amin, "AgeMax": amax, "R2": r2_cor,
                           **({"MAE": mae_cor} if self.save_mae else {})})
        self._debug.append({"Model": self.model, "Cohort": cohort, "Group": "TD",
                            "N": n_cor, "AgeMin": amin, "AgeMax": amax,
                            "R2_raw": r2_raw, "R2_corr": r2_cor,
                            **({"MAE_raw": mae_raw, "MAE_corr": mae_cor} if self.save_mae else {}),
                            "beta": beta, "alpha": alpha, "N_fit_bias": n_fit})

    def add_clinical(self, cohort: str, group: str, ages: np.ndarray, preds: np.ndarray,
                     td_for_bias: tuple[str, np.ndarray, np.ndarray] | None = None):
        if cohort not in self._td_bias:
            if td_for_bias is None:
                raise ValueError(f"TD bias for '{cohort}' unknown. Add TD first or pass td_for_bias=(cohort, ages_td, preds_td).")
            c_name, ages_td, preds_td = td_for_bias
            assert c_name == cohort
            beta, alpha, n_fit = _fit_bias_params(ages_td, preds_td)
            self._td_bias[cohort] = (beta, alpha, n_fit)
        else:
            beta, alpha, n_fit = self._td_bias[cohort]

        # raw & corrected metrics for clinical
        r2_raw, mae_raw, n_raw, amin, amax = self._metrics(ages, preds)
        preds_corr, _bag_corr = _apply_bias(ages, preds, beta, alpha)
        r2_cor, mae_cor, n_cor, _, _ = self._metrics(ages, preds_corr)

        self._rows.append({"Model": self.model, "Cohort": cohort, "Group": group,
                           "N": n_cor, "AgeMin": amin, "AgeMax": amax, "R2": r2_cor,
                           **({"MAE": mae_cor} if self.save_mae else {})})
        self._debug.append({"Model": self.model, "Cohort": cohort, "Group": group,
                            "N": n_cor, "AgeMin": amin, "AgeMax": amax,
                            "R2_raw": r2_raw, "R2_corr": r2_cor,
                            **({"MAE_raw": mae_raw, "MAE_corr": mae_cor} if self.save_mae else {}),
                            "beta(TD)": beta, "alpha(TD)": alpha, "N_fit_bias(TD)": n_fit})

    def to_dataframe(self) -> pd.DataFrame:
        cols = ["Model","Cohort","Group","N","AgeMin","AgeMax","R2"]
        if self.save_mae: cols.append("MAE")
        return pd.DataFrame(self._rows)[cols].sort_values(["Model","Cohort","Group"]).reset_index(drop=True)

    def debug_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._debug).sort_values(["Model","Cohort","Group"]).reset_index(drop=True)

# -------------------------------------------------
# ======= DATA SECTION (FILL THIS PART) ===========
# -------------------------------------------------
# Provide your matrices/vectors here. The code assumes:
# - Train on HCP-Dev: X_dev, y_dev
# - External TD:       (NKI-RS)     X_nki_td, y_nki
#                      (CMI-HBN TD) X_hbn_td, y_hbn_td
#                      (ADHD-200 TD)X_adhd200_td, y_adhd200_td
#                      (ABIDE TD, Stanford TD) optionally
# - Clinical ADHD:     (CMI-HBN ADHD)   X_hbn_adhd, y_hbn_adhd
#                      (ADHD-200 ADHD)  X_adhd200_adhd, y_adhd200_adhd
# - Clinical ASD:      (ABIDE ASD)      X_abide_asd, y_abide_asd
#                      (Stanford ASD)   X_stan_asd,  y_stan_asd
#
# >>> Replace the placeholders below with your actual arrays <<<
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

# -------------------------------------------------
# ======= MODELS SECTION (EDIT AS NEEDED) =========
# -------------------------------------------------
# Example classical ML models (scikit-learn)
def sanitize_X(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    X[np.isposinf(X)] = 0.0
    X[np.isneginf(X)] = 0.0
    return X

def mkpipe(estimator, *, scale=True, use_pca=False, pca_n=200):
    steps = [('impute', SimpleImputer(strategy='median'))]
    if scale:
        steps.append(('scale', StandardScaler(with_mean=True)))
    if use_pca:
        steps.append(('pca', PCA(n_components=pca_n, svd_solver='auto', random_state=27)))
    steps.append(('est', estimator))
    return Pipeline(steps)

USE_PCA = False
PCA_N   = 200

# ==== REPLACE your old 'regressors = []' with this ====
regressors = [
    ('linSVR', mkpipe(svm.SVR(kernel='linear'),               scale=True,  use_pca=USE_PCA, pca_n=PCA_N)),
    ('KNN',    mkpipe(neighbors.KNeighborsRegressor(),        scale=True,  use_pca=USE_PCA, pca_n=PCA_N)),
    ('DT',     mkpipe(tree.DecisionTreeRegressor(),           scale=False, use_pca=USE_PCA, pca_n=PCA_N)),
    ('LR',     mkpipe(linear_model.LinearRegression(),        scale=True,  use_pca=USE_PCA, pca_n=PCA_N)),
    ('RC',     mkpipe(linear_model.Ridge(alpha=0.5),          scale=True,  use_pca=USE_PCA, pca_n=PCA_N)),
    ('LASSO',  mkpipe(linear_model.Lasso(alpha=0.1),          scale=True,  use_pca=USE_PCA, pca_n=PCA_N)),
    ('RF',     mkpipe(ensemble.RandomForestRegressor(random_state=27), scale=False, use_pca=USE_PCA, pca_n=PCA_N)),
]
X_dev        = sanitize_X(X_dev)
X_nki_td     = sanitize_X(X_nki_td)
X_hbn_td     = sanitize_X(X_hbn_td)
X_adhd200_td = sanitize_X(X_adhd200_td)
X_hbn_adhd   = sanitize_X(X_hbn_adhd)
X_adhd200_adhd = sanitize_X(X_adhd200_adhd)
X_abide_asd  = sanitize_X(X_abide_asd)
X_stan_asd   = sanitize_X(X_stan_asd)
X_abide_td   = sanitize_X(X_abide_td)
X_stan_td = sanitize_X(X_stan_td)
 
# -------------------------------------------------
# ======= RUN BENCHMARK ===========================
# -------------------------------------------------
def predict_on(model, X, y):
    """Return predicted ages (years) from a fitted model."""
    yhat = model.predict(X)
    yhat = np.asarray(yhat, float)
    return yhat

def run_benchmark(save_csv="ml_benchmark_r2_by_dataset.csv", save_mae=False):
    #feat_scaler = StandardScaler(with_mean=True)  # fit on HCP-Dev only
    #X_dev_s = feat_scaler.fit_transform(X_dev)

    #def s(X): return feat_scaler.transform(X)

    # Prepare scaled external sets (use try/except to allow missing datasets)
    sets = {}
    def add_set(key, X, y):
        sets[key] = (X, y)

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

    # ---- 5b. Train each model on scaled HCP-Dev and log results ----
    all_rows = []
    for name, est in regressors:
        print(f"\n=== MODEL: {name} ===")
        est.fit(X_dev, y_dev)

        log = BenchmarkLogger(model_name=name, save_mae=save_mae)

        # TD first per cohort (fits TD bias)
        for cohort_td in ["NKI-RS_TD","CMI-HBN_TD","ADHD-200_TD","ABIDE_TD","Stanford_TD"]:
            if cohort_td not in sets: continue
            X_s, y = sets[cohort_td]
            yhat = est.predict(X_s)
            cohort = cohort_td.replace("_TD","").replace("_","-")
            log.add_td(cohort=cohort, ages=y, preds=yhat)

        # ADHD cohorts (use same cohort's TD bias)
        for cohort_key in ["CMI-HBN_ADHD","ADHD-200_ADHD"]:
            if cohort_key not in sets: continue
            X_s, y = sets[cohort_key]
            yhat = est.predict(X_s)
            cohort = cohort_key.replace("_ADHD","")
            log.add_clinical(cohort=cohort, group="ADHD", ages=y, preds=yhat)

        # ASD cohorts (if TD not added above, pass td_for_bias)
        for cohort_key in ["ABIDE_ASD","Stanford_ASD"]:
            if cohort_key not in sets: continue
            X_s, y = sets[cohort_key]
            yhat = est.predict(X_s)
            cohort = cohort_key.replace("_ASD","")
            # If TD bias for this cohort exists, just add. Otherwise, fit on-the-fly with provided TD features.
            if cohort+"_TD" in sets:
                log.add_clinical(cohort=cohort, group="ASD", ages=y, preds=yhat)
            else:
                # requires TD features to be defined; if not, raise with a helpful message
                td_key = cohort+"_TD"
                if td_key not in sets:
                    raise ValueError(f"TD features for {cohort} not provided; needed to derive ASD bias. "
                                     f"Provide X_{cohort.lower()}_td / y_{cohort.lower()}_td, "
                                     f"or add TD first via log.add_td.")
                X_td_s, y_td = sets[td_key]
                yhat_td = est.predict(X_td_s)
                log.add_clinical(cohort=cohort, group="ASD", ages=y, preds=yhat,
                                 td_for_bias=(cohort, y_td, yhat_td))

        df_model = log.to_dataframe()
        df_debug = log.debug_dataframe()
        all_rows.append(df_model)

    # ---- 5c. Save combined table and pretty-print pivot ----
    df_all = pd.concat(all_rows, ignore_index=True)
    Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(save_csv, index=False)
    df_debug.to_csv("./ml_benchmark_debug.csv", index=False)
    print(f"\n[saved] {save_csv}  (rows={len(df_all)})")

    # Quick pivot like your manuscript table (R^2)
    # Note: HCP-Development line is not included here because this benchmark tests external sets.
    pvt = (df_all
           .pivot_table(index=["Cohort","Group"], columns="Model", values="R2", aggfunc="first")
           .reindex(index=[
               ("NKI-RS","TD"), ("CMI-HBN","TD"), ("ADHD-200","TD"),
               ("ADHD-200","ADHD"), ("CMI-HBN","ADHD"),
               ("ABIDE","ASD"), ("Stanford","ASD")
           ])
           .round(3))
    print("\nR^2 by Cohort × Group × Model:\n", pvt)

# -------------------------------------------------
# ======= MAIN ===================================
# -------------------------------------------------
if __name__ == "__main__":
    # >>> Ensure DATA SECTION variables are filled before running <<<
    # Example call (set save_mae=True if you want MAE in the CSV as well):
    run_benchmark(save_csv="./ml_benchmark_r2_by_dataset.csv", save_mae=False)

