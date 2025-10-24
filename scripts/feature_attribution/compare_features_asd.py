#!/usr/bin/env python3
"""
Compare top-50% IG ROI overlap between two wide-format CSVs and compute similarity.

This version uses fixed file paths (no CLI args) and offers two selection modes to be
robust to different numbers of subjects and scale differences between cohorts:

1) mean_IG (default):
   - Per cohort: compute the MEAN IG per ROI across subjects, rank ROIs by mean, select top 50%.
   - Different N affects variance, not the mean; still comparable.

2) rank_based (optional):
   - Per subject: rank all ROIs by IG (descending). Convert to percentile rank per subject.
   - Aggregate per cohort by the median percentile rank per ROI. Select top 50% by that.
   - Scale-invariant and less sensitive to subject-count differences.

Outputs:
- overlap_metrics.json  (counts + Jaccard/Dice + cosine over intersection & union)
- overlap_roi_list.csv  (intersecting ROIs with cohort-wise means and ranks)
- ranked_IG_A.csv, ranked_IG_B.csv (full cohort rankings)

Edit the FILE PATHS below to point to your CSVs if needed.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

# =====================
# FIXED INPUT PATHS
# =====================
FILE_A = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/feature_attribution/stanford_asd_features_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS.csv"
FILE_B = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/feature_attribution/abide_asd_features_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS.csv"
OUTDIR = Path("ig_overlap_out_ASD")
TOP_FRAC = 0.5  # top 50%
SELECTION_MODE = "mean_IG"  # "mean_IG" or "rank_based"

NON_ROI_CANDIDATES = {
    "subject_id", "participant_id", "subid", "sub_id", "id",
    "Unnamed: 0", "Unnamed: 0.1", "Unnamed: 0.2"
}

def detect_roi_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in NON_ROI_CANDIDATES]
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return numeric_cols

def pick_top(series: pd.Series, frac: float) -> pd.Series:
    n = max(1, int(np.floor(series.shape[0] * frac)))
    return series.nlargest(n)

def jaccard_dice(setA: set, setB: set):
    inter = setA & setB
    union = setA | setB
    jaccard = len(inter) / len(union) if union else float("nan")
    dice = (2 * len(inter)) / (len(setA) + len(setB)) if (len(setA)+len(setB)) else float("nan")
    return jaccard, dice, inter, union

def cosine(u: np.ndarray, v: np.ndarray) -> float:
    u = u.astype(float); v = v.astype(float)
    nu = np.linalg.norm(u); nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return float("nan")
    return float(np.dot(u, v) / (nu * nv))

def aggregate_mean_mode(A: pd.DataFrame, B: pd.DataFrame, roi_cols: list[str]):
    A_mean = A[roi_cols].mean(axis=0)
    B_mean = B[roi_cols].mean(axis=0)
    return A_mean.sort_values(ascending=False), B_mean.sort_values(ascending=False)

def aggregate_rank_mode(A: pd.DataFrame, B: pd.DataFrame, roi_cols: list[str]):
    def cohort_rank_series(df: pd.DataFrame) -> pd.Series:
        X = df[roi_cols].copy()
        ranks = X.rank(axis=1, method="average", ascending=False)
        N = len(roi_cols)
        pct = 1.0 - (ranks - 1) / (N - 1)
        return pct.median(axis=0)
    A_med = cohort_rank_series(A)
    B_med = cohort_rank_series(B)
    return A_med.sort_values(ascending=False), B_med.sort_values(ascending=False)

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    A = pd.read_csv(FILE_A)
    B = pd.read_csv(FILE_B)

    roiA = detect_roi_columns(A)
    roiB = detect_roi_columns(B)
    roi_cols = [c for c in roiA if c in roiB]
    if len(roi_cols) == 0:
        raise RuntimeError("No common ROI columns found.")

    if SELECTION_MODE == "mean_IG":
        A_ranked, B_ranked = aggregate_mean_mode(A, B, roi_cols)
    elif SELECTION_MODE == "rank_based":
        A_ranked, B_ranked = aggregate_rank_mode(A, B, roi_cols)
    else:
        raise ValueError("SELECTION_MODE must be 'mean_IG' or 'rank_based'")

    A_ranked.to_csv(OUTDIR / "ranked_IG_A.csv", header=[SELECTION_MODE])
    B_ranked.to_csv(OUTDIR / "ranked_IG_B.csv", header=[SELECTION_MODE])

    topA = pick_top(A_ranked, TOP_FRAC)
    topB = pick_top(B_ranked, TOP_FRAC)
    setA, setB = set(topA.index), set(topB.index)

    jacc, dice, inter, union = jaccard_dice(setA, setB)

    A_mean_all = A[roi_cols].mean(axis=0)
    B_mean_all = B[roi_cols].mean(axis=0)

    inter_list = sorted(list(inter))
    cos_inter = cosine(A_mean_all[inter_list].to_numpy(), B_mean_all[inter_list].to_numpy()) if inter_list else float("nan")
    union_list = sorted(list(union))
    cos_union = cosine(
        A_mean_all.reindex(union_list).fillna(0.0).to_numpy(),
        B_mean_all.reindex(union_list).fillna(0.0).to_numpy(),
    )

    A_rank_pos = pd.Series(range(1, len(A_ranked)+1), index=A_ranked.index, name="rank_A")
    B_rank_pos = pd.Series(range(1, len(B_ranked)+1), index=B_ranked.index, name="rank_B")

    rows = []
    for r in inter_list:
        rows.append({
            "ROI": r,
            "cohortA_metric": float(A_ranked.get(r, np.nan)),
            "cohortB_metric": float(B_ranked.get(r, np.nan)),
            "rank_A": int(A_rank_pos.get(r, np.nan)) if not pd.isna(A_rank_pos.get(r, np.nan)) else None,
            "rank_B": int(B_rank_pos.get(r, np.nan)) if not pd.isna(B_rank_pos.get(r, np.nan)) else None,
            "mean_IG_A": float(A_mean_all.get(r, np.nan)),
            "mean_IG_B": float(B_mean_all.get(r, np.nan)),
        })
    pd.DataFrame(rows).to_csv(OUTDIR / "overlap_roi_list.csv", index=False)

    metrics = {
        "selection_mode": SELECTION_MODE,
        "n_common_rois": len(roi_cols),
        "top_frac": TOP_FRAC,
        "n_top_A": len(topA),
        "n_top_B": len(topB),
        "overlap_count": len(inter),
        "jaccard": jacc,
        "dice": dice,
        "cosine_intersection": cos_inter,
        "cosine_union_zeroed": cos_union,
        "n_subjects_A": int(A.shape[0]),
        "n_subjects_B": int(B.shape[0]),
    }
    (OUTDIR / "overlap_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()

