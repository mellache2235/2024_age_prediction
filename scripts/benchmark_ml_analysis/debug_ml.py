import pandas as pd
import numpy as np
from pathlib import Path

# ---- paths (edit if your files live elsewhere) ----
R2_CSV   = 'ml_benchmark_r2_by_dataset.csv'
DEBUG_CSV= 'ml_benchmark_debug.csv'  # optional (raw vs corr + beta/alpha)

# ---- load ----
df = pd.read_csv(R2_CSV)
dbg = None

print("Loaded:", R2_CSV)
if dbg is None:
    print("Note: no debug CSV found (ml_benchmark_debug.csv); some checks will be skipped.")
else:
    print("Loaded:", DEBUG_CSV)

# ---- pretty R² pivot for a quick look ----
print("\n=== R² by Cohort × Group × Model ===")
pvt = (df.pivot_table(index=["Cohort","Group"], columns="Model", values="R2", aggfunc="first")).round(3)
print(pvt.fillna("NA"))

# ---- merge debug (if available) to get raw R², beta, alpha, age ranges, etc. ----
if dbg is not None:
    # The debug CSV usually has extra columns; ensure common join keys exist
    common_keys = [c for c in ["Model","Cohort","Group"] if c in dbg.columns and c in df.columns]
    merged = pd.merge(df, dbg, on=common_keys, how="left", suffixes=("","_dbg"))
else:
    # fabricate minimal merged with df only
    merged = df.copy()
    for col in ["R2_raw","R2_corr","MAE_raw","MAE_corr","beta","alpha","N_fit_bias"]:
        if col not in merged.columns:
            merged[col] = np.nan

# ---- flag rules ----
def flag_row(r):
    flags = []
    # 1) R² negative (after correction)
    if pd.notna(r["R2"]) and r["R2"] < 0:
        flags.append("R2_negative")

    # 2) big drop after correction (>0.10) if we have raw/corr info
    if pd.notna(r["R2_raw"]) and pd.notna(r["R2_corr"]) and (r["R2_corr"] < r["R2_raw"] - 0.10):
        flags.append("R2_drop_after_corr>0.10")

    # 3) TD bias slope |beta| large (suggests misfit) — only evaluate for TD rows
    if r["Group"] == "TD" and pd.notna(r["beta"]) and abs(r["beta"]) > 0.20:
        flags.append(f"TD_beta_large({r['beta']:.2f})")

    # 4) very small TD used to fit bias
    if r["Group"] == "TD" and pd.notna(r["N_fit_bias"]) and r["N_fit_bias"] < 30:
        flags.append(f"small_TD_for_bias({int(r['N_fit_bias'])})")

    # 5) narrow age range (R² fragile)
    if pd.notna(r.get("AgeMax", np.nan)) and pd.notna(r.get("AgeMin", np.nan)):
        if (r["AgeMax"] - r["AgeMin"]) < 3:
            flags.append("narrow_age_range(<3y)")

    return ";".join(flags)

# Ensure AgeMin/AgeMax exist (from main CSV)
for col in ["AgeMin","AgeMax"]:
    if col not in merged.columns:
        merged[col] = np.nan

merged["flags"] = merged.apply(flag_row, axis=1)

# ---- show suspicious rows grouped by model & cohort ----
sus = merged[merged["flags"] != ""].copy()
sus_disp_cols = [c for c in [
    "Model","Cohort","Group","N","AgeMin","AgeMax",
    "R2_raw","R2_corr","R2","MAE_raw","MAE_corr","beta","alpha","N_fit_bias","flags"
] if c in merged.columns]
print("\n=== Potentially problematic rows ===")
if sus.empty:
    print("No obvious problems flagged. If values still look off, check FC feature order/Fisher-z consistency and bias application.")
else:
    # Sort and print
    sus = sus.sort_values(["Model","Cohort","Group"]).round(4)
    print(sus[sus_disp_cols].to_string(index=False))

    # quick suggestions
    print("\nSuggested actions per flag:")
    print(" - R2_negative: check FC feature order/Fisher-z consistency; consider PCA (fit on HCP-Dev) for linear/SVR/KNN; verify no double scaling.")
    print(" - R2_drop_after_corr>0.10: likely wrong TD β/α applied or double correction; ensure β/α are fit on TD of the SAME cohort and SAME model (once).")
    print(" - TD_beta_large(...): refit bias with more TD or by developmental bands; verify bias fit used TD only and ≥30 subjects.")
    print(" - small_TD_for_bias(...): widen TD pool for bias (or fit per-band then pool); unstable β/α can destroy r² after correction.")
    print(" - narrow_age_range(<3y): R² is fragile; lean on MAE and residual plots; consider pooling adjacent bands.")

# ---- extra: ensure clinical rows inherited the SAME TD β/α within cohort (if debug had beta(TD)/alpha(TD)) ----
if dbg is not None and "beta(TD)" in dbg.columns:
    clin = dbg[dbg["Group"].isin(["ADHD","ASD"])].copy()
    miss = clin[(~np.isclose(clin["beta(TD)"], clin["beta"], atol=1e-8)) |
                (~np.isclose(clin["alpha(TD)"], clin["alpha"], atol=1e-8))]
    if not miss.empty:
        print("\n=== Clinical rows where β/α do not match the cohort's TD bias (fix mapping / avoid double correction) ===")
        print(miss[["Model","Cohort","Group","beta(TD)","alpha(TD)","beta","alpha"]].round(6).to_string(index=False))

# ---- optional: print a compact “top offenders” list by lowest R² ----
print("\n=== Lowest R² rows (top 10) ===")
print(merged.sort_values("R2", na_position="last").head(10)[["Model","Cohort","Group","R2","flags"]].to_string(index=False))
