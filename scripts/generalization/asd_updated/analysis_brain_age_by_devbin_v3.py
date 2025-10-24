# run_end_to_end_devbands_in_memory_seeded.py
import os, math, random, numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr, spearmanr  # <-- direct imports
import sys
sys.path.append('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/train_regression_models/')
from utility_functions import *

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
ABIDE_TS_ROOT = Path("/oak/stanford/groups/menon/deriveddata/public/abide/restfmri/timeseries/group_level/brainnetome/normz/")
FINAL_SITES   = ['NYU','SDSU','STANFORD','Stanford','TCD-1','UM','USM','Yale']  # filter
ASD_LABEL = 1      # <- CHANGE if needed
TD_LABEL  = 2      # <- CHANGE if needed

DEV_BINS = [
    ("child_5_8", 5.0, 8.0),
    ("late_child_8_11", 8.0, 11.0),
    ("early_ado_11_14", 11.0, 14.0),
    ("midlate_ado_14_18", 14.0, 18.0),
    ("emerging_adult_18_21", 18.0, 21.0),
]

MODEL_ROOT = Path("/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/train_regression_models/dev")
HCP_DEV_TPL = Path("/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/"
                   "hcp_dev_age_five_fold/fold_{k}.bin")
USE_SCALER_FROM_HCP = True  # set False if network outputs age in years already

OUT_ROOT = Path("/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/analysis/asd_td_in_memory")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

BEHAV_COLS = ["ados_total", "ados_comm", "ados_social"]
ADOS_SENTINELS = {-9999}  # add more sentinels here if needed

# ========== PREP HELPERS ==========
def adjust_timesteps_for_subjects(subjects_data, target_timesteps=180, padding_value=0):
    adjusted = []
    for subject in subjects_data:
        arr = np.asarray(subject)
        if len(arr) > target_timesteps:
            adj = arr[:target_timesteps]
        else:
            pad_len = target_timesteps - len(arr)
            padding = np.full((pad_len, arr.shape[1]), padding_value)
            adj = np.vstack([arr, padding])
        adjusted.append(adj)
    return np.asarray([np.asarray(i) for i in adjusted])  # (N, T, C)

def to_tensor_timeseries(X_tc: np.ndarray) -> torch.Tensor:
    # (N, T, C) -> (N, C, T) for Conv1d
    X_ct = np.transpose(X_tc, (0, 2, 1))
    return torch.as_tensor(X_ct, dtype=torch.float32)

def make_scaler_from_hcp(k: int) -> StandardScaler:
    bin_path = str(HCP_DEV_TPL).format(k=k)
    Xtr, Xva, Ytr, Yva, *_ = load_finetune_dataset(bin_path)
    return StandardScaler().fit(Ytr.reshape(-1,1))

def ensemble_predict(X_tensor: torch.Tensor) -> np.ndarray:
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

def load_abide_df() -> pd.DataFrame:
    frames = []
    for fn in os.listdir(ABIDE_TS_ROOT):
        if any(s in fn for s in FINAL_SITES):
            arr = np.load(ABIDE_TS_ROOT / fn, allow_pickle=True)
            arr = arr[~pd.isna(arr)]
            frames.append(arr)
    df = pd.concat(frames)
    df['label'] = df['label'].astype(int)
    df['age']   = df['age'].astype(float)
    # ensure behavior columns exist (fill with NaN if absent)
    for c in BEHAV_COLS:
        if c not in df.columns: df[c] = np.nan
    # clean weird ADOS values: set sentinels to NaN
    for c in BEHAV_COLS:
        vals = df[c].to_numpy()
        for s in ADOS_SENTINELS:
            vals = np.where(vals == s, np.nan, vals)
        df[c] = vals
    return df

# ========== BIAS CORRECTION ==========
def fit_bias_td_in_bin(df_td_pred: pd.DataFrame, lo: float, hi: float,
                       n_min=30, span_min=2.0) -> tuple[float,float,str,list]:
    """Return (beta, alpha, mode, used_bins) with adaptive widening; intercept-only if needed."""
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

    # widen symmetrically across DEV_BINS order
    names = [b[0] for b in DEV_BINS]; idx = names.index(next(n for n,l,h in DEV_BINS if l==lo and h==hi))
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

# ========== BEHAVIOR CORR ==========
def corrstats(x, y):
    """Return (pearson_r, pearson_p, spearman_rho, spearman_p) with NaNs if <3 valid pairs."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return (np.nan, np.nan, np.nan, np.nan)
    pr, pp = pearsonr(x[ok], y[ok])
    sr, sp = spearmanr(x[ok], y[ok])
    return (pr, pp, sr, sp)

# ========== MAIN ==========
def main():
    # load ABIDE, restrict age, split groups
    df = load_abide_df()
    df = df.loc[df['age'] <= 21].reset_index(drop=True)
    df_asd = df.loc[df['label']==ASD_LABEL].copy()
    df_td  = df.loc[df['label']==TD_LABEL].copy()
    print(f"Loaded: ASD N={len(df_asd)}  TD N={len(df_td)} (ages ≤ 21)")

    # QC and tensor conversion
    def prep(df_grp):
        bad = df_grp['data'].apply(lambda x: (np.asarray(x).ndim < 2) or np.isnan(np.asarray(x)).any())
        kept = df_grp.loc[~bad].copy().reset_index(drop=True)
        X_tc = adjust_timesteps_for_subjects(kept['data'].values)   # (N,T,C)
        Xt   = to_tensor_timeseries(X_tc)                           # (N,C,T)
        return kept, Xt

    df_asd, X_asd = prep(df_asd)
    df_td,  X_td  = prep(df_td)

    # predictions once for all subjects
    df_asd['pred_age_raw'] = ensemble_predict(X_asd)
    df_td['pred_age_raw']  = ensemble_predict(X_td)

    # per-band analysis
    rows = []
    for name, lo, hi in DEV_BINS:
        asd_bin = df_asd.loc[(df_asd['age']>=lo)&(df_asd['age']<hi)].copy()
        td_bin  = df_td.loc[(df_td['age']>=lo)&(df_td['age']<hi)].copy()
        if asd_bin.empty:
            print(f"[{name}] ASD empty — skip"); continue

        beta, alpha, mode, used = fit_bias_td_in_bin(df_td, lo, hi)
        bag_raw  = asd_bin['pred_age_raw'].to_numpy() - asd_bin['age'].to_numpy()
        if beta is None:
            # last resort: residualize within ASD
            beta, alpha = np.polyfit(asd_bin['age'].to_numpy(), bag_raw, 1); mode="asd-residualize"
        bag_corr = bag_raw - (beta*asd_bin['age'].to_numpy() + alpha)
        ycorr    = asd_bin['age'].to_numpy() + bag_corr

        # age↔prediction stats
        if len(asd_bin) > 1:
            r_raw,  p_r_raw  = pearsonr(asd_bin['age'], asd_bin['pred_age_raw'])
            r_corr, p_r_corr = pearsonr(asd_bin['age'], ycorr)
        else:
            r_raw = p_r_raw = r_corr = p_r_corr = np.nan
        mae_raw  = mean_absolute_error(asd_bin['age'], asd_bin['pred_age_raw'])
        mae_corr = mean_absolute_error(asd_bin['age'], ycorr)
        mbag_raw = float(np.mean(bag_raw))
        mbag_corr= float(np.mean(bag_corr))

        # behavior correlations (omit sentinels because they are NaN now)
        beh_vals = {c: asd_bin[c].to_numpy().astype(float) if c in asd_bin.columns else np.full(len(asd_bin), np.nan)
                    for c in BEHAV_COLS}
        pr_tot_raw, pp_tot_raw, sr_tot_raw, sp_tot_raw = corrstats(bag_raw,  beh_vals["ados_total"])
        pr_tot_cor, pp_tot_cor, sr_tot_cor, sp_tot_cor = corrstats(bag_corr, beh_vals["ados_total"])

        pr_com_raw, pp_com_raw, sr_com_raw, sp_com_raw = corrstats(bag_raw,  beh_vals["ados_comm"])
        pr_com_cor, pp_com_cor, sr_com_cor, sp_com_cor = corrstats(bag_corr, beh_vals["ados_comm"])

        pr_soc_raw, pp_soc_raw, sr_soc_raw, sp_soc_raw = corrstats(bag_raw,  beh_vals["ados_social"])
        pr_soc_cor, pp_soc_cor, sr_soc_cor, sp_soc_cor = corrstats(bag_corr, beh_vals["ados_social"])

        # TD diagnostics (same bias applied)
        td_mean_bag_raw = td_mean_bag_corr = np.nan
        td_n = len(td_bin)
        if td_n > 0:
            td_bag_raw  = td_bin['pred_age_raw'].to_numpy() - td_bin['age'].to_numpy()
            td_bag_corr = td_bag_raw - (beta * td_bin['age'].to_numpy() + alpha)
            td_mean_bag_raw  = float(np.mean(td_bag_raw))
            td_mean_bag_corr = float(np.mean(td_bag_corr))

        # per-subject CSV + summary row
        out_csv = OUT_ROOT / f"{name}_ASD_preds.csv"
        df_out = pd.DataFrame({
            "row_type": "subject",
            "subjid": asd_bin.get("subjid", pd.Series(np.arange(len(asd_bin)))),
            "age": asd_bin['age'],
            "pred_age_raw": asd_bin['pred_age_raw'],
            "BAG_raw": bag_raw,
            "pred_age_corr": ycorr,
            "BAG_corr": bag_corr,
            "ados_total": beh_vals["ados_total"],
            "ados_comm":  beh_vals["ados_comm"],
            "ados_social": beh_vals["ados_social"],
        })

        summary_row = pd.DataFrame([{
            "row_type": "summary",
            "subjid": np.nan, "age": np.nan, "pred_age_raw": np.nan,
            "BAG_raw": mbag_raw, "pred_age_corr": np.nan, "BAG_corr": mbag_corr,
            # bias info
            "beta": beta, "alpha": alpha, "bias_mode": mode, "td_bins_used": str(used),
            # accuracy + significance
            "r_raw": r_raw, "p_r_raw": p_r_raw, "MAE_raw": mae_raw, "mean_BAG_raw": mbag_raw,
            "r_corr": r_corr, "p_r_corr": p_r_corr, "MAE_corr": mae_corr, "mean_BAG_corr": mbag_corr,
            # BAG–ADOS_total
            "pearson_BAG_total_raw_r":   pr_tot_raw, "pearson_BAG_total_raw_p":   pp_tot_raw,
            "spearman_BAG_total_raw_r":  sr_tot_raw, "spearman_BAG_total_raw_p":  sp_tot_raw,
            "pearson_BAG_total_corr_r":  pr_tot_cor, "pearson_BAG_total_corr_p":  pp_tot_cor,
            "spearman_BAG_total_corr_r": sr_tot_cor, "spearman_BAG_total_corr_p": sp_tot_cor,
            # BAG–ADOS_comm
            "pearson_BAG_comm_raw_r":   pr_com_raw, "pearson_BAG_comm_raw_p":   pp_com_raw,
            "spearman_BAG_comm_raw_r":  sr_com_raw, "spearman_BAG_comm_raw_p":  sp_com_raw,
            "pearson_BAG_comm_corr_r":  pr_com_cor, "pearson_BAG_comm_corr_p":  pp_com_cor,
            "spearman_BAG_comm_corr_r": sr_com_cor, "spearman_BAG_comm_corr_p": sp_com_cor,
            # BAG–ADOS_social
            "pearson_BAG_social_raw_r":   pr_soc_raw, "pearson_BAG_social_raw_p":   pp_soc_raw,
            "spearman_BAG_social_raw_r":  sr_soc_raw, "spearman_BAG_social_raw_p":  sp_soc_raw,
            "pearson_BAG_social_corr_r":  pr_soc_cor, "pearson_BAG_social_corr_p":  pp_soc_cor,
            "spearman_BAG_social_corr_r": sr_soc_cor, "spearman_BAG_social_corr_p": sp_soc_cor,
            # TD diagnostics
            "N_TD_in_bin": td_n, "TD_mean_BAG_raw": td_mean_bag_raw, "TD_mean_BAG_corr": td_mean_bag_corr,
        }])

        pd.concat([df_out, summary_row], ignore_index=True).to_csv(out_csv, index=False)

        print(
          f"[{name}] N_ASD={len(asd_bin):3d} | r(raw)={r_raw:5.3f} (p={p_r_raw:.3g}) "
          f"MAE(raw)={mae_raw:4.2f} BAG(raw)={mbag_raw:5.2f} | "
          f"r(corr)={r_corr:5.3f} (p={p_r_corr:.3g}) MAE(corr)={mae_corr:4.2f} BAG(corr)={mbag_corr:5.2f} | "
          f"bias={mode} used={used} | ρ(BAG↔ADOS_tot)={sr_tot_raw:.3f} (p={sp_tot_raw:.3g}) → corr {sr_tot_cor:.3f} (p={sp_tot_cor:.3g})"
        )

        rows.append([
            name, len(asd_bin),
            r_raw, p_r_raw, mae_raw, mbag_raw,
            r_corr, p_r_corr, mae_corr, mbag_corr,
            pr_tot_raw, pp_tot_raw, sr_tot_raw, sp_tot_raw,
            pr_tot_cor, pp_tot_cor, sr_tot_cor, sp_tot_cor,
            pr_com_raw, pp_com_raw, sr_com_raw, sp_com_raw,
            pr_com_cor, pp_com_cor, sr_com_cor, sp_com_cor,
            pr_soc_raw, pp_soc_raw, sr_soc_raw, sp_soc_raw,
            pr_soc_cor, pp_soc_cor, sr_soc_cor, sp_soc_cor,
            mode, str(used), td_n, td_mean_bag_raw, td_mean_bag_corr
        ])

    summary_cols = [
        "bin","N_ASD",
        "r_raw","p_r_raw","MAE_raw","mean_BAG_raw",
        "r_corr","p_r_corr","MAE_corr","mean_BAG_corr",
        # ADOS_total
        "pearson_BAG_total_raw_r","pearson_BAG_total_raw_p",
        "spearman_BAG_total_raw_r","spearman_BAG_total_raw_p",
        "pearson_BAG_total_corr_r","pearson_BAG_total_corr_p",
        "spearman_BAG_total_corr_r","spearman_BAG_total_corr_p",
        # ADOS_comm
        "pearson_BAG_comm_raw_r","pearson_BAG_comm_raw_p",
        "spearman_BAG_comm_raw_r","spearman_BAG_comm_raw_p",
        "pearson_BAG_comm_corr_r","pearson_BAG_comm_corr_p",
        "spearman_BAG_comm_corr_r","spearman_BAG_comm_corr_p",
        # ADOS_social
        "pearson_BAG_social_raw_r","pearson_BAG_social_raw_p",
        "spearman_BAG_social_raw_r","spearman_BAG_social_raw_p",
        "pearson_BAG_social_corr_r","pearson_BAG_social_corr_p",
        "spearman_BAG_social_corr_r","spearman_BAG_social_corr_p",
        # bias + TD diag
        "bias_mode","td_bins_used","N_TD_in_bin","TD_mean_BAG_raw","TD_mean_BAG_corr"
    ]
    summary = pd.DataFrame(rows, columns=summary_cols)
    summary_path = OUT_ROOT / "summary_asd_per_band_with_behavior.csv"
    summary.to_csv(summary_path, index=False)
    print("\n=== summary ===")
    print(summary.to_string(index=False))
    print(f"\nSaved per-subject CSVs (with summary rows) and overall summary to: {OUT_ROOT}")

if __name__ == "__main__":
    main()

