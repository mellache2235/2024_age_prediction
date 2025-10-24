# td_vs_patient_bag_test_td_only_correction.py
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ttest_ind, norm

# ------------------ CONFIG ------------------
# Which clinical group are you comparing to TD?  "ADHD" or "ASD"
GROUP_FOR_PATIENT = "ADHD"   # change to "ASD" if needed

# TD files (uncorrected, to be corrected here)
CMI_TD_ACT  = Path("/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/cmihbn_updated/actual_cmihbn_td_ages.npz")
CMI_TD_PRED = Path("/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/cmihbn_updated/predicted_cmihbn_td_ages.npz")

ADHD200_TD_ACT  = Path("/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/adhd200_updated/actual_adhd200_td_ages.npz")
ADHD200_TD_PRED = Path("/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/adhd200_updated/predicted_adhd200_td_ages.npz")

# Clinical (already corrected) ACT/PRED paths — set for ADHD or ASD explicitly
# --- ADHD example (edit if your filenames differ) ---
CMI_PAT_ACT  = Path("/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/cmihbn_updated/actual_cmihbn_adhd_ages_most_updated.npz")
CMI_PAT_PRED = Path("/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/cmihbn_updated/predicted_cmihbn_adhd_ages_most_updated.npz")

ADHD200_PAT_ACT  = Path("/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/adhd200_updated/actual_adhd200_ages_most_updated.npz")
ADHD200_PAT_PRED = Path("/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/adhd200_updated/predicted_adhd200_ages_most_updated.npz")

# --- If you’re testing ASD instead, point these to ASD files and set GROUP_FOR_PATIENT="ASD" ---

# ------------------ HELPERS ------------------
def load_npz(path: Path, key: str):
    arr = np.load(path, allow_pickle=True)[key].astype(float)
    return arr

def fit_td_bias(age_td, pred_td):
    """Fit TD-only linear bias: BAG = β*Age + α."""
    bag_td = pred_td - age_td
    m = np.isfinite(age_td) & np.isfinite(bag_td)
    if m.sum() < 3:
        raise RuntimeError("Too few TD points to fit bias (need ≥3).")
    beta, alpha = np.polyfit(age_td[m], bag_td[m], 1)
    return float(beta), float(alpha)

def correct_td(age_td, pred_td, beta, alpha):
    """Return corrected TD predictions and corrected TD BAG."""
    pred_corr = pred_td - (beta * age_td + alpha)
    bag_corr  = pred_corr - age_td
    return pred_corr, bag_corr

def welch_diff(a, b):
    """a - b (means), Welch t-test; returns (diff, t, p, se)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    ma, mb = np.nanmean(a), np.nanmean(b)
    va, vb = np.nanvar(a, ddof=1), np.nanvar(b, ddof=1)
    na, nb = np.isfinite(a).sum(), np.isfinite(b).sum()
    t, p = ttest_ind(a, b, equal_var=False)
    se = np.sqrt(va/na + vb/nb)
    return (ma - mb), t, p, se

def process_one_cohort(name, td_act_path, td_pred_path, pat_act_path, pat_pred_path):
    # TD (uncorrected → correct here)
    age_td  = load_npz(td_act_path,  'actual')
    pred_td = load_npz(td_pred_path, 'predicted')
    beta, alpha = fit_td_bias(age_td, pred_td)
    td_pred_corr, td_bag_corr = correct_td(age_td, pred_td, beta, alpha)

    # Patient (already corrected) — use as is
    age_pat  = load_npz(pat_act_path,  'actual')
    pred_pat = load_npz(pat_pred_path, 'predicted')
    pat_bag_corr = pred_pat - age_pat

    # Compare means (patient − TD)
    diff, t, p, se = welch_diff(pat_bag_corr, td_bag_corr)

    print(f"{name} [{GROUP_FOR_PATIENT} vs TD]: TD-bias β={beta:.3f}, α={alpha:.3f} | "
          f"mean BAG_corr TD={np.nanmean(td_bag_corr):+.3f}, {GROUP_FOR_PATIENT}={np.nanmean(pat_bag_corr):+.3f} | "
          f"{GROUP_FOR_PATIENT}−TD diff={diff:+.3f} y, t={t:.2f}, p={p:.3g} "
          f"(n_{GROUP_FOR_PATIENT}={np.isfinite(pat_bag_corr).sum()}, n_TD={np.isfinite(td_bag_corr).sum()})")
    return diff, se

# ------------------ RUN ------------------
effects = []

# CMI-HBN
effects.append(
    process_one_cohort(
        "CMI-HBN",
        CMI_TD_ACT, CMI_TD_PRED,
        CMI_PAT_ACT, CMI_PAT_PRED
    )
)

# ADHD-200
effects.append(
    process_one_cohort(
        "ADHD-200",
        ADHD200_TD_ACT, ADHD200_TD_PRED,
        ADHD200_PAT_ACT, ADHD200_PAT_PRED
    )
)

# Pooled Welch across cohorts (optional quick check)
# You can also pool by concatenating, but here we meta-analyze per-cohort effects:

diffs = np.array([d for d,_ in effects], float)
ses   = np.array([s for _,s in effects], float)
w     = 1.0 / (ses**2)

# Fixed-effect meta-analysis
diff_FE = np.sum(w*diffs) / np.sum(w)
se_FE   = np.sqrt(1.0 / np.sum(w))
z_FE    = diff_FE / se_FE
p_FE    = 2 * norm.sf(abs(z_FE))
print(f"\nPooled effect (fixed): {GROUP_FOR_PATIENT}−TD = {diff_FE:+.3f} ± {1.96*se_FE:.3f} y, z={z_FE:.2f}, p={p_FE:.3g}")

# Random-effects (DerSimonian–Laird)
Q   = np.sum(w * (diffs - diff_FE)**2)
dfq = len(diffs) - 1
C   = np.sum(w) - (np.sum(w**2)/np.sum(w))
tau2= max(0.0, (Q - dfq) / C)
wRE = 1.0 / (ses**2 + tau2)
diff_RE = np.sum(wRE*diffs) / np.sum(wRE)
se_RE   = np.sqrt(1.0 / np.sum(wRE))
z_RE    = diff_RE / se_RE
p_RE    = 2 * norm.sf(abs(z_RE))
print(f"Pooled effect (random): {GROUP_FOR_PATIENT}−TD = {diff_RE:+.3f} ± {1.96*se_RE:.3f} y, z={z_RE:.2f}, p={p_RE:.3g}, tau²={tau2:.4f}")

