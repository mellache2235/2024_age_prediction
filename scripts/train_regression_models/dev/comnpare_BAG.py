import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ttest_ind, norm

# ---------- helper: load one cohort (actual & predicted are bias-corrected) ----------
def load_cohort(actual_path, predicted_path, cohort_name, group_name):
    a = np.load(actual_path,    allow_pickle=True)['actual'].astype(float)
    p = np.load(predicted_path, allow_pickle=True)['predicted'].astype(float)
    if a.shape[0] != p.shape[0]:
        raise ValueError(f"Length mismatch in {cohort_name}: actual={a.shape[0]} vs predicted={p.shape[0]}")
    df = pd.DataFrame({'age': a, 'pred': p})
    df['bag'] = df['pred'] - df['age']  # NOTE: predicted already bias-corrected
    df['cohort'] = cohort_name
    df['group']  = group_name           # 'ASD' or 'ADHD'
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['bag','age','pred'])
    return df

# ---------- paths (edit if your predicted file names differ) ----------
ROOT_ASD = Path("/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/asd_updated")
ROOT_ADHD= Path("/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/adhd200_updated")
ROOT_ADHD_CMIHBN = Path("/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/cmihbn_updated")
files = [
    # ASD
    (ROOT_ASD / "actual_abide_asd_ages_most_updated.npz",
     ROOT_ASD / "predicted_abide_asd_ages_most_updated.npz",   "ABIDE",    "ASD"),
    (ROOT_ASD / "actual_stanford_asd_ages_most_updated.npz",
     ROOT_ASD / "predicted_stanford_asd_ages_most_updated.npz","Stanford", "ASD"),
    # ADHD
    (ROOT_ADHD / "actual_adhd200_ages_most_updated.npz",
     ROOT_ADHD / "predicted_adhd200_ages_most_updated.npz",    "ADHD-200", "ADHD"),
    (ROOT_ADHD_CMIHBN / "actual_cmihbn_adhd_ages_most_updated.npz",
     ROOT_ADHD_CMIHBN / "predicted_cmihbn_adhd_ages_most_updated.npz","CMI-HBN",  "ADHD"),
]

df = pd.concat([load_cohort(a,p,c,g) for a,p,c,g in files], ignore_index=True)

# ---------- quick per-cohort summary ----------
print("\nPer-cohort BAG summary (bias-corrected):")
print(df.groupby(['group','cohort'])['bag'].agg(['count','mean','std']).round(3))

# ---------- 1) pooled Welch t-test (ASD vs ADHD) ----------
asd_bag   = df.loc[df['group']=='ASD','bag'].to_numpy()
adhd_bag  = df.loc[df['group']=='ADHD','bag'].to_numpy()
t_stat, p_val = ttest_ind(asd_bag, adhd_bag, equal_var=False)
diff = np.nanmean(asd_bag) - np.nanmean(adhd_bag)
print(f"\nPooled Welch t-test (ASD − ADHD mean BAG): "
      f"diff = {diff:.3f} y,  t = {t_stat:.3f},  p = {p_val:.3g}")

# ---------- 2) cohort-aware meta-analysis ----------
# compute mean & SE per cohort
cohort_stats = df.groupby(['group','cohort'])['bag'].agg(['count','mean','std']).reset_index()
cohort_stats['se'] = cohort_stats['std'] / np.sqrt(cohort_stats['count'])

# inverse-variance pooled mean per group (fixed-effect)
def ivw_mean(stats):
    w  = 1 / (stats['se']**2)
    m  = np.sum(w * stats['mean']) / np.sum(w)
    se = np.sqrt(1 / np.sum(w))
    return m, se

asd_stats  = cohort_stats[cohort_stats['group']=='ASD']
adhd_stats = cohort_stats[cohort_stats['group']=='ADHD']
m_asd, se_asd   = ivw_mean(asd_stats)
m_adhd, se_adhd = ivw_mean(adhd_stats)

diff_FE  = m_asd - m_adhd
se_diff  = np.sqrt(se_asd**2 + se_adhd**2)
z        = diff_FE / se_diff
p_FE     = 2 * norm.sf(abs(z))

print(f"\nMeta-analytic (fixed-effect) pooled means:")
print(f"  ASD pooled mean BAG = {m_asd:.3f} ± {1.96*se_asd:.3f}")
print(f"  ADHD pooled mean BAG = {m_adhd:.3f} ± {1.96*se_adhd:.3f}")
print(f"Group difference (ASD − ADHD) = {diff_FE:.3f} ± {1.96*se_diff:.3f},  z = {z:.3f},  p = {p_FE:.3g}")

# ---------- Optional: random-effects if you suspect heterogeneity ----------
# DerSimonian–Laird tau^2 across cohorts within each group, then recompute pooled means
def dersimonian_laird(stats):
    w = 1 / (stats['se']**2)
    m_FE = np.sum(w * stats['mean']) / np.sum(w)
    Q = np.sum(w * (stats['mean'] - m_FE)**2)
    df_q = len(stats) - 1
    C = np.sum(w) - (np.sum(w**2) / np.sum(w))
    tau2 = max(0.0, (Q - df_q) / C)
    w_RE = 1 / (stats['se']**2 + tau2)
    m_RE = np.sum(w_RE * stats['mean']) / np.sum(w_RE)
    se_RE = np.sqrt(1 / np.sum(w_RE))
    return m_RE, se_RE, tau2

m_asd_RE,  se_asd_RE,  tau2_asd  = dersimonian_laird(asd_stats)
m_adhd_RE, se_adhd_RE, tau2_adhd = dersimonian_laird(adhd_stats)
diff_RE  = m_asd_RE - m_adhd_RE
se_diff_RE = np.sqrt(se_asd_RE**2 + se_adhd_RE**2)
z_RE     = diff_RE / se_diff_RE
p_RE     = 2 * norm.sf(abs(z_RE))
print(f"\nRandom-effects (DL):")
print(f"  ASD pooled = {m_asd_RE:.3f} ± {1.96*se_asd_RE:.3f} (tau²={tau2_asd:.4f})")
print(f"  ADHD pooled = {m_adhd_RE:.3f} ± {1.96*se_adhd_RE:.3f} (tau²={tau2_adhd:.4f})")
print(f"  Group diff = {diff_RE:.3f} ± {1.96*se_diff_RE:.3f},  z = {z_RE:.3f},  p = {p_RE:.3g}")

# ---------- Optional: band-wise tests (uncomment if you want age windows) ----------
# bins = [(5,8),(8,11),(11,14),(14,18),(18,21)]
# for lo,hi in bins:
#     sub = df[(df['age']>=lo) & (df['age']<hi)]
#     a = sub[sub['group']=='ASD']['bag'].to_numpy()
#     d = sub[sub['group']=='ADHD']['bag'].to_numpy()
#     if len(a)>=5 and len(d)>=5:
#         t,p = ttest_ind(a,d,equal_var=False)
#         print(f"Band {lo}-{hi}: diff={np.nanmean(a)-np.nanmean(d):.3f}, t={t:.2f}, p={p:.3g}, n={len(a)}/{len(d)}")

