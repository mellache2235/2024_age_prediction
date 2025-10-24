# run_adhd200_brain_age_to_excel.py
import os, random, math, numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from scipy.interpolate import interp1d

# your utilities / model (same ones used in ABIDE scripts)
#from test_hcp_dev_all_CV_models_abide_asd import ConvNet  # Conv1d(246,...) → global avg → Linear(32→1)
import sys
sys.path.append('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/')
from utility_functions import load_finetune_dataset             # only used if scaling
import torch.nn as nn

# =================== SEEDS ===================
SEED = 27
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(246, 32, kernel_size=5, stride=1, bias=False),
            nn.BatchNorm1d(32),
            nn.PReLU(32),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(32,32, kernel_size=7, stride=1, bias=False),
            nn.BatchNorm1d(32),
            nn.PReLU(32),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.drop_out = nn.Dropout(p=0.6)
        self.regressor = nn.Linear(32, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.drop_out(out)
        out = self.layer2(out)
        out = self.drop_out(out)
        out = out.mean(axis=2)
        out = self.regressor(out)
        return out

# =================== CONFIG ===================
DATA_DIR = Path("/oak/stanford/groups/menon/deriveddata/public/adhd200/restfmri/timeseries/group_level/brainnetome/normz/")
PKLZ = DATA_DIR / "adhd200_run-rest_brainnetome_mean_regMov-6param_wmcsf_dt1_bpf008-09_normz_246ROIs.pklz"

# groups / labels
TD_LABEL  = 0
ADHD_LABEL= 1

# developmental bands (≤21y)
DEV_BINS = [
    ("child_5_8", 5.0, 8.0),
    ("late_child_8_11", 8.0, 11.0),
    ("early_ado_11_14", 11.0, 14.0),
    ("midlate_ado_14_18", 14.0, 18.0),
    ("emerging_adult_18_21", 18.0, 21.0),
]

# model paths & scaling (same as ABIDE)
MODEL_ROOT = Path("/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/train_regression_models/dev")
HCP_DEV_TPL = Path("/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/"
                   "hcp_dev_age_five_fold/fold_{k}.bin")
USE_SCALER_FROM_HCP = True      # False if your network outputs years directly

# interpolation settings
TR_REF = 0.8                    # HCP reference TR (seconds)
BASE_T = 180                    # start by padding/truncating to 180 TRs before resampling

# Excel output
OUT_DIR = Path("/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/analysis/adhd200")
OUT_DIR.mkdir(parents=True, exist_ok=True)
EXCEL_PATH = OUT_DIR / "adhd200_brain_age_summary.xlsx"

# =================== HELPERS ===================
def adjust_timesteps_for_subjects(subjects_data, target_timesteps=BASE_T, padding_value=0):
    """Pad/truncate each subject to (T=target_timesteps, C=ROIs)."""
    adjusted = []
    for subject in subjects_data:
        arr = np.asarray(subject)
        if arr.ndim != 2:
            raise ValueError(f"Timeseries must be 2D (T,C); got {arr.shape}")
        if arr.shape[0] > target_timesteps:
            adj = arr[:target_timesteps]
        else:
            pad_len = target_timesteps - arr.shape[0]
            padding = np.full((pad_len, arr.shape[1]), padding_value, dtype=arr.dtype)
            adj = np.vstack([arr, padding])
        adjusted.append(adj)
    return np.asarray(adjusted)  # (N, T, C)

def resample_time_axis_to_len(X_tc: np.ndarray, new_len: int) -> np.ndarray:
    """(N,T,C) → (N,new_len,C) via linear interpolation along T."""
    N, T, C = X_tc.shape
    if new_len == T: return X_tc
    f = interp1d(np.linspace(0,1,T), X_tc, axis=1, kind="linear", assume_sorted=True)
    return f(np.linspace(0,1,new_len))

def zscore_per_subject_region(X_tc: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per subject & ROI, z-score across time: (X - mean_t)/std_t."""
    mean_t = X_tc.mean(axis=1, keepdims=True)   # (N,1,C)
    std_t  = X_tc.std(axis=1, keepdims=True)    # (N,1,C)
    return (X_tc - mean_t) / (std_t + eps)

def to_tensor_timeseries(X_tc: np.ndarray) -> torch.Tensor:
    """(N,T,C) → (N,C,T) for Conv1d."""
    return torch.as_tensor(np.transpose(X_tc, (0,2,1)), dtype=torch.float32)

def make_scaler_from_hcp(k: int) -> StandardScaler:
    bin_path = str(HCP_DEV_TPL).format(k=k)
    Xtr, Xva, Ytr, Yva, *_ = load_finetune_dataset(bin_path)
    return StandardScaler().fit(Ytr.reshape(-1,1))

def ensemble_predict(X_tensor: torch.Tensor) -> np.ndarray:
    """Predict ages (years) with 5 folds; inverse-transform per fold if trained on z-age."""
    preds = []
    for k in range(5):
        model_path = MODEL_ROOT / f"best_outer_fold_{k}_hcp_dev_model_2_27_24.pt"
        model = ConvNet()
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        with torch.no_grad():
            yk = model(X_tensor).squeeze().cpu().numpy()
        if USE_SCALER_FROM_HCP:
            sc = make_scaler_from_hcp(k)
            yk = sc.inverse_transform(yk.reshape(-1,1)).ravel()
        preds.append(yk)
    return np.mean(np.stack(preds, axis=0), axis=0)

def corrstats(x, y):
    """Return (pearson_r, pearson_p, spearman_rho, spearman_p) with NaNs if <3 pairs."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3: return (np.nan, np.nan, np.nan, np.nan)
    pr, pp = pearsonr(x[ok], y[ok])
    sr, sp = spearmanr(x[ok], y[ok])
    return (pr, pp, sr, sp)

def corrstats_with_n(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 3:
        return (np.nan, np.nan, np.nan, np.nan, n)
    pr, pp = pearsonr(x[ok], y[ok])
    sr, sp = spearmanr(x[ok], y[ok])
    return (pr, pp, sr, sp, n)

def fit_bias_td_in_bin(df_td_pred: pd.DataFrame, lo: float, hi: float,
                       n_min=25, span_min=2.0):
    """
    Fit BAG = beta*age + alpha from TD in [lo,hi). If insufficient, widen across DEV_BINS.
    Returns (beta, alpha, mode, used_bins).
    """
    def pick(df, bins):
        m = pd.Series(False, index=df.index)
        for _, l, h in bins:
            m |= ((df['age'] >= l) & (df['age'] < h))
        sub = df.loc[m]
        if sub.empty: return None, None
        return sub['age'].to_numpy(float), sub['pred_age_raw'].to_numpy(float)

    # in-bin
    ages, preds = pick(df_td_pred, [("this", lo, hi)])
    if ages is not None:
        span = ages.max() - ages.min()
        if len(ages) >= n_min and span >= span_min:
            beta, alpha = np.polyfit(ages, preds - ages, 1)
            return float(beta), float(alpha), 'in-bin', [(lo,hi)]

    # widen
    names = [b[0] for b in DEV_BINS]
    idx = names.index(next(n for n,l,h in DEV_BINS if (l, h) == (lo, hi)))
    L=R=0; used=[DEV_BINS[idx]]
    while True:
        expanded=False
        if idx-(L+1) >= 0: L+=1; used.insert(0, DEV_BINS[idx-L]); expanded=True
        if idx+(R+1) < len(DEV_BINS): R+=1; used.append(DEV_BINS[idx+R]); expanded=True
        ages, preds = pick(df_td_pred, used)
        if ages is None: break
        span = ages.max() - ages.min()
        if len(ages) >= n_min and span >= span_min:
            beta, alpha = np.polyfit(ages, preds - ages, 1)
            return float(beta), float(alpha), 'widened', [(b[1],b[2]) for b in used]
        if not expanded: break

    # intercept-only
    if len(df_td_pred) > 0:
        alpha = float((df_td_pred['pred_age_raw'] - df_td_pred['age']).mean())
        return 0.0, alpha, 'intercept-only', ['all-td']
    return None, None, 'none', []

def _finite_mask(*arrs):
    m = np.ones(len(arrs[0]), dtype=bool)
    for a in arrs:
        m &= np.isfinite(np.asarray(a, float))
    return m

def safe_pearson(x, y):
    m = _finite_mask(x, y); n = int(m.sum())
    if n < 3: 
        return np.nan, np.nan, n
    r, p = pearsonr(np.asarray(x)[m], np.asarray(y)[m])
    return r, p, n

def safe_mae(x, y):
    m = _finite_mask(x, y); n = int(m.sum())
    return (mean_absolute_error(np.asarray(x)[m], np.asarray(y)[m]) if n > 0 else np.nan, n)

def nanmean(v):
    v = np.asarray(v, float)
    return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan

# --- EXACT-NAME behavior attach (no guessing) ---
import re

SENTINELS = {-999, -999., -9999, 999, 9999}

def _clean_colnames(df: pd.DataFrame) -> pd.DataFrame:
    # normalize column names: strip spaces, unify division slash variants to '/'
    mapping = {}
    for c in df.columns:
        cc = str(c).strip().replace("\u2215", "/")  # U+2215 → '/'
        mapping[c] = cc
    return df.rename(columns=mapping)

def _coerce_numeric(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(np.nan, index=pd.RangeIndex(0))
    vals = pd.to_numeric(series, errors="coerce")
    if np.isfinite(vals).sum() == 0:
        # fallback: extract first number token e.g., "65T" -> 65
        tok = series.astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0]
        vals = pd.to_numeric(tok, errors="coerce")
    for s in SENTINELS:
        vals = vals.mask(vals == s, np.nan)
    return vals

def _z_by_site(vals: pd.Series, site: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=vals.index, dtype=float)
    if "site" not in site.index.names and site.shape[0] == vals.shape[0]:
        s = site.astype(str)
    else:
        s = pd.Series("all", index=vals.index)
    for sitename, idx in s.groupby(s).groups.items():
        v = vals.iloc[idx]
        m, sd = np.nanmean(v), np.nanstd(v)
        if np.isfinite(m) and np.isfinite(sd) and sd > 0:
            out.iloc[idx] = (v - m) / (sd + 1e-8)
    return out

def attach_behavior_columns_exact(df: pd.DataFrame) -> pd.DataFrame:
    df = _clean_colnames(df.copy())

    need = {"Inattentive", "Hyper/Impulsive"}
    if not need.issubset(set(df.columns)):
        print("[behavior] Columns missing. Have:", list(df.columns))
        missing = need - set(df.columns)
        print("[behavior] Missing expected:", missing)
        # nothing more to do; create empty numeric columns so later code won't crash
        for c in ("inatt_num","hyper_num","inatt_z_site","hyper_z_site"):
            df[c] = np.nan
        return df

    inatt = _coerce_numeric(df["Inattentive"])
    hyper = _coerce_numeric(df["Hyper/Impulsive"])

    df["inatt_num"] = inatt
    df["hyper_num"] = hyper
    site = df["site"] if "site" in df.columns else pd.Series("all", index=df.index)
    df["inatt_z_site"] = _z_by_site(inatt, site)
    df["hyper_z_site"] = _z_by_site(hyper, site)

    print(f"[behavior] counts: Inattentive non-NaN={int(np.isfinite(inatt).sum())}, "
          f"Hyper/Impulsive non-NaN={int(np.isfinite(hyper).sum())}")
    # small peek at first few numeric values
    print("[behavior] sample:",
          df.loc[np.isfinite(inatt)|np.isfinite(hyper), ["site","Inattentive","Hyper/Impulsive"]].head(8).to_string(index=False))
    return df

# =================== LOAD ADHD200 & FILTER ===================
# --- CONFIG: turn off severity gating by default ---
STRICT_BEHAV_FILTERS = False   # True to enforce NYU/Peking score gates
NYU_T_CUTOFF = 65.0            # T-score threshold if strict mode
PEKING_RAW_CUTOFF = 20.0       # raw-score threshold if strict mode

def load_adhd200_df(pklz_path: Path) -> pd.DataFrame:
    df = np.load(pklz_path, allow_pickle=True)
    
    # base filters
    df = df[df['tr'] != 2.5]
    df = df[df['label'] != 'pending']
    df = df[df['mean_fd'] < 0.5]
    #df['Inattentive']     = pd.to_numeric(df['Inattentive'], errors='coerce')
    #df['Hyper/Impulsive'] = pd.to_numeric(df['Hyper/Impulsive'], errors='coerce')
    
    df['age']             = pd.to_numeric(df['age'], errors='coerce')
    df['label']           = pd.to_numeric(df['label'], errors='coerce')
    df = df[df['age'] <= 21].copy()

    df['Inattentive'] = df['Inattentive'].astype(float)
    df['Hyper/Impulsive'] = df['Hyper/Impulsive'].astype(float)
    valid_df = df[
    df[['Inattentive', 'Hyper/Impulsive']].notna().all(axis=1) &
    (df['Inattentive'] != -999.) &
    (df['Hyper/Impulsive'] != -999.)
    ]
    valid_df = valid_df.reset_index()
    df = valid_df
    # keep all rows now; behavior NaNs will be omitted pairwise in correlations later
    # normalize site strings a bit
    df['site'] = df['site'].astype(str)

    # label-only groups first (primary fallback)
    df_labelonly = df.copy()
    labs = df_labelonly['label'].to_numpy()
    labs[labs != 0] = 1
    df_labelonly['label'] = labs.astype(int)

    td_labelonly   = df_labelonly[df_labelonly['label'] == 0].copy()
    adhd_labelonly = df_labelonly[df_labelonly['label'] == 1].copy()

    print(f"[label-only] TD={len(td_labelonly)}, ADHD={len(adhd_labelonly)} | "
          f"sites: {df_labelonly['site'].value_counts().to_dict()}")

    if not STRICT_BEHAV_FILTERS:
        return pd.concat([td_labelonly, adhd_labelonly]).reset_index(drop=True)

    # ---------- strict severity gating (optional) ----------
    nyu_df    = df[df['site'].str.contains('NYU', case=False, na=False)].copy()
    peking_df = df[df['site'].str.contains('Peking', case=False, na=False)].copy()

    nyu_td   = nyu_df[(nyu_df['label'] == 0) &
                      ~((nyu_df['Inattentive'] > NYU_T_CUTOFF) |
                        (nyu_df['Hyper/Impulsive'] > NYU_T_CUTOFF))]
    nyu_adhd = nyu_df[(nyu_df['label'] == 1) &
                      ((nyu_df['Inattentive'] >= NYU_T_CUTOFF) |
                       (nyu_df['Hyper/Impulsive'] >= NYU_T_CUTOFF))]

    peking_td   = peking_df[(peking_df['label'] == 0) &
                            ~((peking_df['Inattentive'] > PEKING_RAW_CUTOFF) |
                              (peking_df['Hyper/Impulsive'] > PEKING_RAW_CUTOFF))]
    peking_adhd = peking_df[(peking_df['label'] == 1) &
                            ((peking_df['Inattentive'] >= PEKING_RAW_CUTOFF) |
                             (peking_df['Hyper/Impulsive'] >= PEKING_RAW_CUTOFF))]

    df_strict = pd.concat([nyu_td, nyu_adhd, peking_td, peking_adhd]).reset_index(drop=True)
    labs = df_strict['label'].to_numpy()
    labs[labs != 0] = 1
    df_strict['label'] = labs.astype(int)

    td_strict   = df_strict[df_strict['label'] == 0]
    adhd_strict = df_strict[df_strict['label'] == 1]
    print(f"[strict] TD={len(td_strict)}, ADHD={len(adhd_strict)} "
          f"(NYU cutoff {NYU_T_CUTOFF}, Peking cutoff {PEKING_RAW_CUTOFF})")

    # auto-fallback if strict gating kills ADHD
    if len(adhd_strict) == 0:
        print("[strict] No ADHD after gating → reverting to label-only grouping.")
        return pd.concat([td_labelonly, adhd_labelonly]).reset_index(drop=True)

    return df_strict.reset_index(drop=True)


# =================== PREDICT (per-TR interpolation) ===================
def predict_with_tr_groups(df_grp: pd.DataFrame) -> pd.Series:
    # Pad/truncate to BASE_T first
    X_tc0 = adjust_timesteps_for_subjects(df_grp['data'].values)    # (N, BASE_T, C)
    tr_vec = pd.to_numeric(df_grp['tr'], errors='coerce').to_numpy(float)

    # Warn and keep NaNs where TR is missing
    bad_tr = ~np.isfinite(tr_vec)
    if bad_tr.any():
        print(f"[warn] {bad_tr.sum()} rows have NaN/Inf TR → predictions will be NaN for those rows.")

    preds = np.full(len(df_grp), np.nan, dtype=float)
    uniq_tr = np.unique(np.round(tr_vec[np.isfinite(tr_vec)], 3))

    for trv in uniq_tr:
        idx = np.where(np.isfinite(tr_vec) & np.isclose(tr_vec, trv, rtol=0, atol=1e-3))[0]
        if idx.size == 0: 
            continue
        X_block = X_tc0[idx]
        T_new = int(math.floor(BASE_T * (float(trv) / TR_REF)))
        if T_new <= 16:   # guard against too-short sequences
            T_new = 16
        X_rs = resample_time_axis_to_len(X_block, T_new)
        X_rs = zscore_per_subject_region(X_rs, eps=1e-8)
        Xt    = to_tensor_timeseries(X_rs)
        preds[idx] = ensemble_predict(Xt)

    return pd.Series(preds, index=df_grp.index, dtype=float)

# =================== MAIN ===================
def main():
    # Load and filter ADHD200 (≤21y, NYU/Peking, motion, behavior rules)
    df = load_adhd200_df(PKLZ)
    df["Inattentive"]     = pd.to_numeric(df["Inattentive"], errors="coerce")
    df["Hyper/Impulsive"] = pd.to_numeric(df["Hyper/Impulsive"], errors="coerce")

    # map sentinels -> NaN (so they’re excluded pairwise later)
    for col in ["Inattentive", "Hyper/Impulsive"]:
        df.loc[df[col].isin([-999, -999.0, -9999, 999, 9999]), col] = np.nan

    # optional: within-site z for harmonization
    def _z_by_site_safe(series, site):
        return series.groupby(site).transform(
            lambda v: (v - np.nanmean(v)) / (np.nanstd(v) + 1e-8) if np.nanstd(v) > 0 else np.nan
        )

    df["Inattentive_z_site"]     = _z_by_site_safe(df["Inattentive"], df["site"].astype(str))
    df["Hyper/Impulsive_z_site"] = _z_by_site_safe(df["Hyper/Impulsive"], df["site"].astype(str))

    print("[diag] non-NaN counts:",
          "Inattentive=", df["Inattentive"].notna().sum(),
          "Hyper/Impulsive=", df["Hyper/Impulsive"].notna().sum())
    # now split groups
    df_td   = df[df["label"] == 0].copy()
    df_adhd = df[df["label"] == 1].copy()
    print(f"ADHD200 after filters: TD N={len(df_td)}, ADHD N={len(df_adhd)} | sites: {df['site'].value_counts().to_dict()}")

    # Predict raw ages with per-TR interpolation
    df_td['pred_age_raw']   = predict_with_tr_groups(df_td)
    df_adhd['pred_age_raw'] = predict_with_tr_groups(df_adhd)

    # Excel writer
    with pd.ExcelWriter(EXCEL_PATH, engine="xlsxwriter") as xlw:
        summary_rows = []

        for name, lo, hi in DEV_BINS:
            adhd_bin = df_adhd[(df_adhd['age'] >= lo) & (df_adhd['age'] < hi)].copy()
            td_bin   = df_td  [(df_td['age'] >= lo) & (df_td['age'] < hi)].copy()  # noqa: E999
            if adhd_bin.empty:
                print(f"[{name}] ADHD empty — skip"); continue

            # TD-derived bias (adaptive)
            df_td_all = df_td[['age','pred_age_raw']].dropna()
            # --- build raw/corrected arrays (may contain NaN) ---
            age   = adhd_bin['age'].to_numpy(float)
            ypred = adhd_bin['pred_age_raw'].to_numpy(float)
            bag_raw  = ypred - age

            # TD-derived bias (as you already have)
            beta, alpha, mode, used = fit_bias_td_in_bin(df_td[['age','pred_age_raw']].dropna(), lo, hi)
            if beta is None:
                # fallback: residualize within ADHD
                m_tmp = _finite_mask(age, bag_raw)
                beta, alpha = np.polyfit(age[m_tmp], bag_raw[m_tmp], 1) if m_tmp.sum() >= 2 else (0.0, 0.0)
                mode = "adhd-residualize"

            bag_corr = bag_raw - (beta * age + alpha)
            ycorr    = age + bag_corr

            # --- accuracy on pairwise-finite rows only ---
            r_raw,  p_r_raw,  n_age_raw  = safe_pearson(age, ypred)
            mae_raw,              _      = safe_mae(age, ypred)
            mbag_raw = nanmean(bag_raw)

            r_corr, p_r_corr,  n_age_cor = safe_pearson(age, ycorr)
            mae_corr,             _      = safe_mae(age, ycorr)
            mbag_corr = nanmean(bag_corr)


            # behavior (pairwise-complete)
            #inatt = adhd_bin["Inattentive"].to_numpy(float)
            #hyper = adhd_bin["Hyper/Impulsive"].to_numpy(float)
            inatt = adhd_bin["Inattentive_z_site"].to_numpy(float)
            hyper = adhd_bin["Hyper/Impulsive_z_site"].to_numpy(float)
            
            pr_inatt_raw, pp_inatt_raw, sr_inatt_raw, sp_inatt_raw, n_inatt_raw = corrstats_with_n(bag_raw,  inatt)
            pr_inatt_cor, pp_inatt_cor, sr_inatt_cor, sp_inatt_cor, n_inatt_cor = corrstats_with_n(bag_corr, inatt)

            pr_hyper_raw, pp_hyper_raw, sr_hyper_raw, sp_hyper_raw, n_hyper_raw = corrstats_with_n(bag_raw,  hyper)
            pr_hyper_cor, pp_hyper_cor, sr_hyper_cor, sp_hyper_cor, n_hyper_cor = corrstats_with_n(bag_corr, hyper)
            
            # TD diagnostics in this band
            td_n = len(td_bin)
            td_mean_bag_raw = td_mean_bag_corr = np.nan
            if td_n > 0:
                td_bag_raw  = td_bin['pred_age_raw'].to_numpy() - td_bin['age'].to_numpy()
                td_bag_corr = td_bag_raw - (beta*td_bin['age'].to_numpy() + alpha)
                td_mean_bag_raw  = float(np.mean(td_bag_raw))
                td_mean_bag_corr = float(np.mean(td_bag_corr))

            # per-subject sheet + one-row summary
            df_sheet = pd.DataFrame({
                "row_type": "subject",
                "subjid": adhd_bin.get("subject_id", pd.Series(np.arange(len(adhd_bin)))),
                "age": adhd_bin["age"],
                "tr": adhd_bin["tr"],
                "site": adhd_bin["site"],
                "pred_age_raw": adhd_bin["pred_age_raw"],
                "BAG_raw": bag_raw,
                "pred_age_corr": ycorr,
                "BAG_corr": bag_corr,
                "Inattentive": inatt,
                "Hyper/Impulsive": hyper,
            })

            summary_row = pd.DataFrame([{
                "row_type": "summary",
                "subjid": np.nan, "age": np.nan, "tr": np.nan, "site": np.nan,
                "pred_age_raw": np.nan, "BAG_raw": mbag_raw,
                "pred_age_corr": np.nan, "BAG_corr": mbag_corr,
                # bias info
                "beta": beta, "alpha": alpha, "bias_mode": mode, "td_bins_used": str(used),
                # accuracy + significance
                "r_raw": r_raw, "p_r_raw": p_r_raw, "MAE_raw": mae_raw, "mean_BAG_raw": mbag_raw,
                "r_corr": r_corr, "p_r_corr": p_r_corr, "MAE_corr": mae_corr, "mean_BAG_corr": mbag_corr,
                # BAG–Inattentive
                "pearson_BAG_inatt_raw_r":   pr_inatt_raw, "pearson_BAG_inatt_raw_p":   pp_inatt_raw,
                "spearman_BAG_inatt_raw_r":  sr_inatt_raw, "spearman_BAG_inatt_raw_p":  sp_inatt_raw,
                "pearson_BAG_inatt_corr_r":  pr_inatt_cor, "pearson_BAG_inatt_corr_p":  pp_inatt_cor,
                "spearman_BAG_inatt_corr_r": sr_inatt_cor, "spearman_BAG_inatt_corr_p": sp_inatt_cor,
                # BAG–Hyper/Impulsive
                "pearson_BAG_hyper_raw_r":   pr_hyper_raw, "pearson_BAG_hyper_raw_p":   pp_hyper_raw,
                "spearman_BAG_hyper_raw_r":  sr_hyper_raw, "spearman_BAG_hyper_raw_p":  sp_hyper_raw,
                "pearson_BAG_hyper_corr_r":  pr_hyper_cor, "pearson_BAG_hyper_corr_p":  pp_hyper_cor,
                "spearman_BAG_hyper_corr_r": sr_hyper_cor, "spearman_BAG_hyper_corr_p": sp_hyper_cor,
                # TD diag
                "N_TD_in_bin": td_n, "TD_mean_BAG_raw": td_mean_bag_raw, "TD_mean_BAG_corr": td_mean_bag_corr,
            }])
            pd.concat([df_sheet, summary_row], ignore_index=True).to_excel(xlw, sheet_name=name[:31], index=False)

            print(f"[{name}] N_ADHD={len(adhd_bin):3d} | r(raw)={r_raw:5.3f} (p={p_r_raw:.3g}) MAE(raw)={mae_raw:4.2f} "
                  f"BAG(raw)={mbag_raw:5.2f} | r(corr)={r_corr:5.3f} (p={p_r_corr:.3g}) MAE(corr)={mae_corr:4.2f} "
                  f"BAG(corr)={mbag_corr:5.2f} | bias={mode} used={used} "
                  f"| ρ_inatt(raw)={sr_inatt_raw:.3f} (p={sp_inatt_raw:.3g}) → corr {sr_inatt_cor:.3f} (p={sp_inatt_cor:.3g}) "
                  f"| ρ_hyper(raw)={sr_hyper_raw:.3f} (p={sp_hyper_raw:.3g}) → corr {sr_hyper_cor:.3f} (p={sp_hyper_cor:.3g})")

            # add to global summary
            summary_rows.append([
                name, len(adhd_bin),
                r_raw, p_r_raw, mae_raw, mbag_raw,
                r_corr, p_r_corr, mae_corr, mbag_corr,
                pr_inatt_raw, pp_inatt_raw, sr_inatt_raw, sp_inatt_raw,
                pr_inatt_cor, pp_inatt_cor, sr_inatt_cor, sp_inatt_cor,
                pr_hyper_raw, pp_hyper_raw, sr_hyper_raw, sp_hyper_raw,
                pr_hyper_cor, pp_hyper_cor, sr_hyper_cor, sp_hyper_cor,
                mode, str(used), td_n, td_mean_bag_raw, td_mean_bag_corr
            ])

        # Summary sheet
        summary_cols = [
            "bin","N_ADHD",
            "r_raw","p_r_raw","MAE_raw","mean_BAG_raw",
            "r_corr","p_r_corr","MAE_corr","mean_BAG_corr",
            # Inattentive
            "pearson_BAG_inatt_raw_r","pearson_BAG_inatt_raw_p",
            "spearman_BAG_inatt_raw_r","spearman_BAG_inatt_raw_p",
            "pearson_BAG_inatt_corr_r","pearson_BAG_inatt_corr_p",
            "spearman_BAG_inatt_corr_r","spearman_BAG_inatt_corr_p",
            # Hyper/Impulsive
            "pearson_BAG_hyper_raw_r","pearson_BAG_hyper_raw_p",
            "spearman_BAG_hyper_raw_r","spearman_BAG_hyper_raw_p",
            "pearson_BAG_hyper_corr_r","pearson_BAG_hyper_corr_p",
            "spearman_BAG_hyper_corr_r","spearman_BAG_hyper_corr_p",
            # bias + TD diag
            "bias_mode","td_bins_used","N_TD_in_bin","TD_mean_BAG_raw","TD_mean_BAG_corr"
        ]
        summary_df = pd.DataFrame(summary_rows, columns=summary_cols)
        summary_df.to_excel(xlw, sheet_name="Summary", index=False)

    print(f"\nSaved Excel → {EXCEL_PATH}")

if __name__ == "__main__":
    main()

