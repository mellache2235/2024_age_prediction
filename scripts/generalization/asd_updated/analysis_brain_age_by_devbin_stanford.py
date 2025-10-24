# run_stanford_asd_td_brain_age_to_excel.py
import os, random, numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr, spearmanr
#from test_hcp_dev_all_CV_models_abide_asd import ConvNet        # your model
from scipy.interpolate import interp1d
import math
import sys
sys.path.append('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/')
from utility_functions import load_finetune_dataset             # only used if scaling
import torch.nn as nn

# ========== SEEDS (deterministic inference) ==========
SEED = 27
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ========== CONFIG ==========
# Stanford ASD dataset (npz)
DATA_DIR = Path("/oak/stanford/groups/menon/projects/sryali/2019_DNN/scripts/daelsaid/output/group")
NPZ_FILE = DATA_DIR / "stanford_brainnnetome_6_wmcsf.npz"

# Behavior (SRS)
SRS_DIR = Path("/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/prepare_data/stanford_autism")
SRS_CSV = SRS_DIR / "SRS_data_20230925.csv"
SRS_ID_COL = "record_id"
SRS_SCORE_COL = "srs_total_score_standard"
SRS_SENTINELS = {-9999}   # any weird codes to omit

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

# Labels
ASD_LABEL = 1
TD_LABEL  = 2

# Developmental bands for <19y
DEV_BINS = [
    ("child_5_8", 5.0, 8.0),
    ("late_child_8_11", 8.0, 11.0),
    ("early_ado_11_14", 11.0, 14.0),
    ("midlate_ado_14_18", 14.0, 18.0),
]

# Model & scaler sources (same as ABIDE workflow)
MODEL_ROOT = Path("/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/train_regression_models/dev")
HCP_DEV_TPL = Path("/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/"
                   "hcp_dev_age_five_fold/fold_{k}.bin")
USE_SCALER_FROM_HCP = True  # set False if the network outputs age in years already

# Output Excel
OUT_DIR = Path("/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/analysis/stanford_asd_td")
OUT_DIR.mkdir(parents=True, exist_ok=True)
EXCEL_PATH = OUT_DIR / "stanford_asd_brain_age_summary.xlsx"

# ========== PREP HELPERS ==========

RESAMPLE_RATIO = 2.0 / 0.8   # matches: floor(T * 2 / 0.8)

def resample_time_axis(X_tc: np.ndarray, ratio: float | None = RESAMPLE_RATIO) -> np.ndarray:
    """
    X_tc: (N, T, C). Returns (N, T_new, C) with T_new = floor(T * ratio).
    Implements: f = interp1d(linspace(0,1,T), X_tc, axis=1); f(linspace(0,1,T_new)).
    """
    if ratio is None or ratio == 1.0:
        return X_tc
    N, T, C = X_tc.shape
    T_new = int(math.floor(T * float(ratio)))
    # build interpolator over the time axis for all subjects/ROIs at once
    f = interp1d(np.linspace(0, 1, T), X_tc, axis=1, kind="linear", assume_sorted=True)
    X_rs = f(np.linspace(0, 1, T_new))
    return X_rs

def adjust_timesteps_for_subjects(subjects_data, target_timesteps=180, padding_value=0):
    """Pad/truncate each subject to (T=target_timesteps, C=ROIs). Input can be list-of-arrays or ndarray(N,T,C)."""
    adjusted = []
    for subject in subjects_data:
        arr = np.asarray(subject)
        if arr.ndim != 2:
            raise ValueError(f"Each subject timeseries must be 2D (T,C); got shape {arr.shape}")
        if arr.shape[0] > target_timesteps:
            adj = arr[:target_timesteps]
        else:
            pad_len = target_timesteps - arr.shape[0]
            padding = np.full((pad_len, arr.shape[1]), padding_value, dtype=arr.dtype)
            adj = np.vstack([arr, padding])
        adjusted.append(adj)
    return np.asarray([np.asarray(i) for i in adjusted])  # (N, T, C)

def zscore_per_subject_region(X_tc: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per subject & ROI, z-score across time: (X - mean_t)/std_t. X_tc is (N, T, C)."""
    mean_t = X_tc.mean(axis=1, keepdims=True)            # (N,1,C)
    std_t  = X_tc.std(axis=1, keepdims=True)             # (N,1,C)
    return (X_tc - mean_t) / (std_t + eps)

def to_tensor_timeseries(X_tc: np.ndarray) -> torch.Tensor:
    """(N, T, C) -> (N, C, T) for Conv1d."""
    X_ct = np.transpose(X_tc, (0, 2, 1))
    return torch.as_tensor(X_ct, dtype=torch.float32)

def make_scaler_from_hcp(k: int) -> StandardScaler:
    bin_path = str(HCP_DEV_TPL).format(k=k)
    Xtr, Xva, Ytr, Yva, *_ = load_finetune_dataset(bin_path)
    return StandardScaler().fit(Ytr.reshape(-1,1))

def ensemble_predict(X_tensor: torch.Tensor) -> np.ndarray:
    """Predict age (years) using 5 fold models; if training was on z-scored age, inverse-transform per fold."""
    preds = []
    for k in range(5):
        model_path = MODEL_ROOT / f"best_outer_fold_{k}_hcp_dev_model_2_27_24.pt"
        model = ConvNet(); model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        with torch.no_grad():
            yk = model(X_tensor).squeeze().cpu().numpy()
        if USE_SCALER_FROM_HCP:
            sc = make_scaler_from_hcp(k)
            yk = sc.inverse_transform(yk.reshape(-1,1)).ravel()
        preds.append(yk)
    return np.mean(np.stack(preds, axis=0), axis=0)

def corrstats(x, y):
    """(pearson_r, pearson_p, spearman_rho, spearman_p) with NaNs if <3 valid pairs."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3: return (np.nan, np.nan, np.nan, np.nan)
    pr, pp = pearsonr(x[ok], y[ok])
    sr, sp = spearmanr(x[ok], y[ok])
    return (pr, pp, sr, sp)

# ========== LOAD STANFORD NPZ + SRS ==========
def load_stanford_npz(npz_path: Path) -> pd.DataFrame:
    z = np.load(npz_path, allow_pickle=True)
    ts   = z["data"]     # could be object array of (T_i, C) arrays OR a 3D array (N,T,C)
    age  = z["ages"].astype(float)
    lab  = z["labels"].astype(int)
    ids  = z["subjids"]

    # Normalize IDs to string
    ids = np.array([str(s).strip() for s in ids])

    # Force list-of-arrays (T,C)
    if isinstance(ts, np.ndarray) and ts.dtype == object:
        ts_list = [np.asarray(t) for t in ts]
    elif isinstance(ts, np.ndarray) and ts.ndim == 3:
        ts_list = [ts[i] for i in range(ts.shape[0])]
    else:
        raise ValueError(f"Unrecognized 'data' format in {npz_path}: dtype={ts.dtype}, ndim={getattr(ts,'ndim',None)}")

    df = pd.DataFrame({
        "subjid": ids,
        "age":    age,
        "label":  lab,
        "data":   ts_list
    })
    # restrict to <19
    df = df.loc[df["age"] < 19].reset_index(drop=True)
    return df

def load_srs_scores(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, skiprows=[0])
    df = df.drop_duplicates(subset=[SRS_ID_COL], keep="last").copy()
    df[SRS_ID_COL] = df[SRS_ID_COL].astype(str).str.strip()
    # clean sentinels to NaN
    vals = pd.to_numeric(df[SRS_SCORE_COL], errors="coerce")
    for s in SRS_SENTINELS:
        vals = vals.mask(vals == s, np.nan)
    df[SRS_SCORE_COL] = vals
    return df[[SRS_ID_COL, SRS_SCORE_COL]]

# ========== BIAS CORRECTION (TD-derived per-band, adaptive) ==========
def fit_bias_td_in_bin(df_td_pred: pd.DataFrame, lo: float, hi: float,
                       n_min=25, span_min=2.0):
    """
    Return (beta, alpha, mode, used_bins) using TD from this dataset.
    mode ∈ {'in-bin','widened','intercept-only','none'}
    """
    def pick(df, bins):
        m = pd.Series(False, index=df.index)
        for _, l, h in bins:
            m |= ((df['age'] >= l) & (df['age'] < h))
        sub = df.loc[m]
        if sub.empty: return None, None
        return sub['age'].to_numpy().astype(float), sub['pred_age_raw'].to_numpy().astype(float)

    # in-bin
    ages, preds = pick(df_td_pred, [("this", lo, hi)])
    if ages is not None:
        span = ages.max() - ages.min()
        if len(ages) >= n_min and span >= span_min:
            beta, alpha = np.polyfit(ages, preds - ages, 1)
            return float(beta), float(alpha), 'in-bin', [(lo,hi)]

    # widen symmetrically
    names = [b[0] for b in DEV_BINS]
    idx   = names.index(next(n for n,l,h in DEV_BINS if l==lo and h==hi))
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

    # intercept-only fallback
    if len(df_td_pred) > 0:
        alpha = float((df_td_pred['pred_age_raw'].to_numpy() - df_td_pred['age'].to_numpy()).mean())
        return 0.0, alpha, 'intercept-only', ['all-td']
    return None, None, 'none', []

# ========== MAIN ==========
def main():
    # 1) Load data & behavior
    df = load_stanford_npz(NPZ_FILE)
    srs = load_srs_scores(SRS_CSV)

    # attach SRS by subject id (string match)
    df = df.merge(srs, left_on="subjid", right_on=SRS_ID_COL, how="left")
    df = df.drop(columns=[SRS_ID_COL])

    # split groups
    df_asd = df.loc[df["label"] == ASD_LABEL].copy()
    df_td  = df.loc[df["label"] == TD_LABEL].copy()
    print(f"Loaded Stanford: ASD N={len(df_asd)}  TD N={len(df_td)} (age < 19)")

    # 2) QC → pad/truncate to 180 → z-score per subject/ROI across time → to tensor
    def prep_df(df_grp: pd.DataFrame):
        # drop subjects with malformed or NaN ts
        bad = df_grp["data"].apply(lambda x: (np.asarray(x).ndim != 2) or np.isnan(np.asarray(x)).any())
        kept = df_grp.loc[~bad].copy().reset_index(drop=True)
        X_tc = adjust_timesteps_for_subjects(kept["data"].values)   # (N,T,C)
        X_tc = resample_time_axis(X_tc)
        X_tc = zscore_per_subject_region(X_tc, eps=1e-8)            # per subject/ROI across time
        Xt   = to_tensor_timeseries(X_tc)                           # (N,C,T) for Conv1d
        return kept, Xt

    df_asd, X_asd = prep_df(df_asd)
    df_td,  X_td  = prep_df(df_td)

    # 3) Predict ages (raw) once
    df_asd["pred_age_raw"] = ensemble_predict(X_asd)
    df_td["pred_age_raw"]  = ensemble_predict(X_td)

    # 4) Excel writer
    with pd.ExcelWriter(EXCEL_PATH, engine="xlsxwriter") as xlw:
        summary_rows = []

        for name, lo, hi in DEV_BINS:
            asd_bin = df_asd.loc[(df_asd["age"]>=lo)&(df_asd["age"]<hi)].copy()
            td_bin  = df_td.loc[(df_td["age"]>=lo)&(df_td["age"]<hi)].copy()
            if asd_bin.empty:
                print(f"[{name}] ASD empty — skip")
                continue

            # TD-derived bias (adaptive)
            beta, alpha, mode, used = fit_bias_td_in_bin(df_td, lo, hi)
            bag_raw  = asd_bin["pred_age_raw"].to_numpy() - asd_bin["age"].to_numpy()
            if beta is None:
                beta, alpha = np.polyfit(asd_bin["age"].to_numpy(), bag_raw, 1); mode = "asd-residualize"
            bag_corr = bag_raw - (beta*asd_bin["age"].to_numpy() + alpha)
            ycorr    = asd_bin["age"].to_numpy() + bag_corr

            # accuracy stats
            if len(asd_bin) > 1:
                r_raw,  p_r_raw  = pearsonr(asd_bin["age"], asd_bin["pred_age_raw"])
                r_corr, p_r_corr = pearsonr(asd_bin["age"], ycorr)
            else:
                r_raw = p_r_raw = r_corr = p_r_corr = np.nan
            mae_raw  = mean_absolute_error(asd_bin["age"], asd_bin["pred_age_raw"])
            mae_corr = mean_absolute_error(asd_bin["age"], ycorr)
            mbag_raw = float(np.mean(bag_raw))
            mbag_corr= float(np.mean(bag_corr))

            # SRS behavior correlations (omit NaNs / sentinels)
            srs_vals = pd.to_numeric(asd_bin[SRS_SCORE_COL], errors="coerce").to_numpy()
            pr_srs_raw, pp_srs_raw, sr_srs_raw, sp_srs_raw = corrstats(bag_raw,  srs_vals)
            pr_srs_cor, pp_srs_cor, sr_srs_cor, sp_srs_cor = corrstats(bag_corr, srs_vals)

            # TD diagnostics (same correction)
            td_n = len(td_bin)
            td_mean_bag_raw = td_mean_bag_corr = np.nan
            if td_n > 0:
                td_bag_raw  = td_bin["pred_age_raw"].to_numpy() - td_bin["age"].to_numpy()
                td_bag_corr = td_bag_raw - (beta*td_bin["age"].to_numpy() + alpha)
                td_mean_bag_raw  = float(np.mean(td_bag_raw))
                td_mean_bag_corr = float(np.mean(td_bag_corr))

            # per-subject sheet (append one-row summary at bottom)
            df_sheet = pd.DataFrame({
                "row_type": "subject",
                "subjid": asd_bin.get("subjid", pd.Series(np.arange(len(asd_bin)))),
                "age": asd_bin["age"],
                "pred_age_raw": asd_bin["pred_age_raw"],
                "BAG_raw": bag_raw,
                "pred_age_corr": ycorr,
                "BAG_corr": bag_corr,
                SRS_SCORE_COL: srs_vals,
            })
            summary_row = pd.DataFrame([{
                "row_type": "summary",
                "subjid": np.nan, "age": np.nan, "pred_age_raw": np.nan,
                "BAG_raw": mbag_raw, "pred_age_corr": np.nan, "BAG_corr": mbag_corr,
                SRS_SCORE_COL: np.nan,
                # bias
                "beta": beta, "alpha": alpha, "bias_mode": mode, "td_bins_used": str(used),
                # accuracy + significance
                "r_raw": r_raw, "p_r_raw": p_r_raw, "MAE_raw": mae_raw, "mean_BAG_raw": mbag_raw,
                "r_corr": r_corr, "p_r_corr": p_r_corr, "MAE_corr": mae_corr, "mean_BAG_corr": mbag_corr,
                # BAG–SRS
                "pearson_BAG_SRS_raw_r":   pr_srs_raw, "pearson_BAG_SRS_raw_p":   pp_srs_raw,
                "spearman_BAG_SRS_raw_r":  sr_srs_raw, "spearman_BAG_SRS_raw_p":  sp_srs_raw,
                "pearson_BAG_SRS_corr_r":  pr_srs_cor, "pearson_BAG_SRS_corr_p":  pp_srs_cor,
                "spearman_BAG_SRS_corr_r": sr_srs_cor, "spearman_BAG_SRS_corr_p": sp_srs_cor,
                # TD diag
                "N_TD_in_bin": td_n, "TD_mean_BAG_raw": td_mean_bag_raw, "TD_mean_BAG_corr": td_mean_bag_corr,
            }])
            df_out = pd.concat([df_sheet, summary_row], ignore_index=True)
            df_out.to_excel(xlw, sheet_name=name[:31], index=False)  # Excel sheet name limit 31

            # console
            print(f"[{name}] N_ASD={len(asd_bin):3d} | r(raw)={r_raw:5.3f} (p={p_r_raw:.3g}) MAE(raw)={mae_raw:4.2f} "
                  f"BAG(raw)={mbag_raw:5.2f} | r(corr)={r_corr:5.3f} (p={p_r_corr:.3g}) MAE(corr)={mae_corr:4.2f} "
                  f"BAG(corr)={mbag_corr:5.2f} | bias={mode} used={used} | ρ_SRS(raw)={sr_srs_raw:.3f} "
                  f"(p={sp_srs_raw:.3g}) → corr {sr_srs_cor:.3f} (p={sp_srs_cor:.3g})")

            # add to global summary
            summary_rows.append([
                name, len(asd_bin),
                r_raw, p_r_raw, mae_raw, mbag_raw,
                r_corr, p_r_corr, mae_corr, mbag_corr,
                pr_srs_raw, pp_srs_raw, sr_srs_raw, sp_srs_raw,
                pr_srs_cor, pp_srs_cor, sr_srs_cor, sp_srs_cor,
                mode, str(used), td_n, td_mean_bag_raw, td_mean_bag_corr
            ])

        # write overall summary last
        summary_cols = [
            "bin","N_ASD",
            "r_raw","p_r_raw","MAE_raw","mean_BAG_raw",
            "r_corr","p_r_corr","MAE_corr","mean_BAG_corr",
            "pearson_BAG_SRS_raw_r","pearson_BAG_SRS_raw_p",
            "spearman_BAG_SRS_raw_r","spearman_BAG_SRS_raw_p",
            "pearson_BAG_SRS_corr_r","pearson_BAG_SRS_corr_p",
            "spearman_BAG_SRS_corr_r","spearman_BAG_SRS_corr_p",
            "bias_mode","td_bins_used","N_TD_in_bin","TD_mean_BAG_raw","TD_mean_BAG_corr"
        ]
        summary_df = pd.DataFrame(summary_rows, columns=summary_cols)
        summary_df.to_excel(xlw, sheet_name="Summary", index=False)

    print(f"\nSaved Excel → {EXCEL_PATH}")

if __name__ == "__main__":
    main()

