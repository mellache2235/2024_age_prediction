#!/usr/bin/env python3
"""
Disorder-level IG comparison: aggregate multiple cohorts per disorder (ASD, ADHD),
then compute top-50% overlap and cosine similarity between disorders.

- Supports two aggregation modes:
  1) mean_IG: concat all subjects within a disorder; mean IG per ROI.
  2) rank_based: per-subject ROI percentile ranks; median per ROI (scale/N-robust).

Outputs (in OUTDIR):
  - ranked_ASD.csv, ranked_ADHD.csv (disorder-level vectors used for selection)
  - overlap_roi_list.csv (ROIs in the intersection of top-50% sets)
  - metrics.json (Jaccard, Dice, cosine_intersection, cosine_union_zeroed, subject counts)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

# ========= FILL THESE =========
ASD_FILES = [
    "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/feature_attribution/stanford_asd_features_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS.csv",
    "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/feature_attribution/abide_asd_features_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS.csv",
]
ADHD_FILES = [
    "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/feature_attribution/adhd200_adhd_features_all_sites_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS_single_model_predictions.csv",
    "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/feature_attribution/cmihbn_adhd_weidong_cutoffs_features_all_sites_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS_single_model_predictions.csv",
]
OUTDIR = Path("ig_overlap_out_asd_adhd")
TOP_FRAC = 0.5            # top 50%
AGG_MODE = "mean_IG"      # "mean_IG" or "rank_based"

NON_ROI = {
    "subject_id","participant_id","subid","sub_id","id",
    "Unnamed: 0","Unnamed: 0.1","Unnamed: 0.2"
}

def detect_roi_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in NON_ROI]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]

def load_and_align(files):
    dfs = [pd.read_csv(f) for f in files]
    roi_sets = [set(detect_roi_cols(df)) for df in dfs]
    common = sorted(set.intersection(*roi_sets))
    if not common:
        raise RuntimeError("No common ROI columns across files.")
    # keep only common numeric ROI columns + subject_id if needed
    aligned = [df[common] for df in dfs]
    return aligned, common

def agg_mean_IG(dfs_roi: list[pd.DataFrame]) -> pd.Series:
    # concat subjects across cohorts, then mean per ROI
    X = pd.concat(dfs_roi, axis=0, ignore_index=True)
    return X.mean(axis=0).sort_values(ascending=False)

def agg_rank_based(dfs_roi: list[pd.DataFrame]) -> pd.Series:
    # per-subject ranks → percentile → median per ROI
    def rank_to_pct(df):
        ranks = df.rank(axis=1, method="average", ascending=False)
        N = df.shape[1]
        return 1.0 - (ranks - 1) / (N - 1)  # high IG -> high percentile
    pct_list = [rank_to_pct(df) for df in dfs_roi]
    pct_all = pd.concat(pct_list, axis=0, ignore_index=True)
    return pct_all.median(axis=0).sort_values(ascending=False)

def pick_top(series: pd.Series, frac: float) -> pd.Index:
    k = max(1, int(np.floor(series.shape[0]*frac)))
    return series.index[:k]

def jaccard_dice(A: set, B: set):
    inter = A & B; union = A | B
    jac = len(inter)/len(union) if union else float("nan")
    dice = (2*len(inter))/(len(A)+len(B)) if (len(A)+len(B)) else float("nan")
    return jac, dice, inter, union

def cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0: return float("nan")
    return float(np.dot(u, v)/(nu*nv))

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    asd_dfs, common = load_and_align(ASD_FILES)
    adhd_dfs, _      = load_and_align(ADHD_FILES)  # shares the same 'common' if atlases match
    # If ADHD files use the same ROI names (they should), force to 'common' for both disorders:
    adhd_dfs = [df[common] for df in adhd_dfs]

    n_asd = sum(df.shape[0] for df in asd_dfs)
    n_adhd= sum(df.shape[0] for df in adhd_dfs)

    if AGG_MODE == "mean_IG":
        asd_vec = agg_mean_IG(asd_dfs)
        adhd_vec= agg_mean_IG(adhd_dfs)
    elif AGG_MODE == "rank_based":
        asd_vec = agg_rank_based(asd_dfs)
        adhd_vec= agg_rank_based(adhd_dfs)
    else:
        raise ValueError("AGG_MODE must be 'mean_IG' or 'rank_based'.")

    # Save disorder-level rankings
    asd_vec.to_csv(OUTDIR/"ranked_ASD.csv", header=[AGG_MODE])
    adhd_vec.to_csv(OUTDIR/"ranked_ADHD.csv", header=[AGG_MODE])

    # Top-50% sets
    asd_top = set(pick_top(asd_vec, TOP_FRAC))
    adhd_top= set(pick_top(adhd_vec, TOP_FRAC))

    jacc, dice, inter, union = jaccard_dice(asd_top, adhd_top)

    # Cosine similarity using MEAN IG magnitudes (even if AGG_MODE='rank_based', for interpretability)
    # Build disorder-level mean IG (concatenate subjects per disorder)
    asd_mean = pd.concat(asd_dfs, axis=0).mean(axis=0)
    adhd_mean= pd.concat(adhd_dfs, axis=0).mean(axis=0)

    inter_list = sorted(list(inter))
    cos_inter = cosine(asd_mean[inter_list].to_numpy(), adhd_mean[inter_list].to_numpy()) if inter_list else float("nan")
    union_list = sorted(list(union))
    cos_union = cosine(
        asd_mean.reindex(union_list).fillna(0.0).to_numpy(),
        adhd_mean.reindex(union_list).fillna(0.0).to_numpy()
    )

    # Save overlap table
    rows = []
    asd_rankpos = pd.Series(range(1, len(asd_vec)+1), index=asd_vec.index, name="rank_ASD")
    adhd_rankpos= pd.Series(range(1, len(adhd_vec)+1), index=adhd_vec.index, name="rank_ADHD")
    for r in inter_list:
        rows.append({
            "ROI": r,
            "ASD_metric": float(asd_vec.get(r, np.nan)),
            "ADHD_metric": float(adhd_vec.get(r, np.nan)),
            "rank_ASD": int(asd_rankpos.get(r, np.nan)) if not pd.isna(asd_rankpos.get(r, np.nan)) else None,
            "rank_ADHD": int(adhd_rankpos.get(r, np.nan)) if not pd.isna(adhd_rankpos.get(r, np.nan)) else None,
            "meanIG_ASD": float(asd_mean.get(r, np.nan)),
            "meanIG_ADHD": float(adhd_mean.get(r, np.nan)),
        })
    pd.DataFrame(rows).to_csv(OUTDIR/"overlap_roi_list.csv", index=False)

    metrics = {
        "agg_mode": AGG_MODE,
        "top_frac": TOP_FRAC,
        "n_common_rois": len(common),
        "n_subjects_ASD": int(n_asd),
        "n_subjects_ADHD": int(n_adhd),
        "n_top_each": int(len(asd_top)),
        "overlap_count": int(len(inter)),
        "jaccard": jacc,
        "dice": dice,
        "cosine_intersection": cos_inter,
        "cosine_union_zeroed": cos_union,
    }
    (OUTDIR/"metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()

