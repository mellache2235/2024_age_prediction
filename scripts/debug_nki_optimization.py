#!/usr/bin/env python3
"""
Debug script to compare NKI enhanced vs optimized data loading.
This will help identify why optimized performs worse than enhanced.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

# Add to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from logging_utils import print_section_header, print_step, print_info

# Import both loading functions
from run_nki_brain_behavior_enhanced import load_nki_behavioral_data as load_enhanced
from run_nki_brain_behavior_optimized import (load_nki_ig_scores, load_nki_behavioral_data as load_optimized,
                                                merge_data)

IG_CSV = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/integrated_gradients/nki_cog_dev_wIDS_features_IG_convnet_regressor_single_model_fold_0.csv"
BEHAVIORAL_DIR = Path("/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/FLUX/assessment_data")

print_section_header("NKI DATA LOADING COMPARISON: ENHANCED VS OPTIMIZED")

# Test measure
TEST_MEASURE = "B T-SCORE (HYPERACTIVITY/RESTLESSNESS)"  # Baseline ρ = 0.410

# Load enhanced version
print_step("Loading data using ENHANCED script logic", "")
enhanced_ig_df = pd.read_csv(IG_CSV)
enhanced_beh_df, enhanced_cols = load_enhanced(BEHAVIORAL_DIR)

print_info(f"Enhanced IG subjects: {len(enhanced_ig_df)}")
print_info(f"Enhanced behavioral subjects: {len(enhanced_beh_df)}")

# Merge enhanced
enhanced_merged = pd.merge(enhanced_ig_df, enhanced_beh_df, on='subject_id', how='inner')
print_info(f"Enhanced merged subjects: {len(enhanced_merged)}")

# Load optimized version
print()
print_step("Loading data using OPTIMIZED script logic", "")
opt_ig_df, opt_roi_cols = load_nki_ig_scores(IG_CSV)
opt_beh_df, opt_cols = load_optimized(BEHAVIORAL_DIR)

print_info(f"Optimized IG subjects: {len(opt_ig_df)}")
print_info(f"Optimized behavioral subjects: {len(opt_beh_df)}")

# Merge optimized
opt_merged = merge_data(opt_ig_df, opt_beh_df)
print_info(f"Optimized merged subjects: {len(opt_merged)}")

# Compare merged subjects
print()
print_step("Comparing merged data", "")
enhanced_ids = set(enhanced_merged['subject_id'])
opt_ids = set(opt_merged['subject_id'])
common_ids = enhanced_ids.intersection(opt_ids)

print_info(f"Common IDs between enhanced and optimized: {len(common_ids)}")
print_info(f"Only in enhanced: {len(enhanced_ids - opt_ids)}")
print_info(f"Only in optimized: {len(opt_ids - enhanced_ids)}")

# Check if test measure exists
print()
print_step(f"Checking test measure: {TEST_MEASURE}", "")
print_info(f"In enhanced columns: {TEST_MEASURE in enhanced_cols}")
print_info(f"In optimized columns: {TEST_MEASURE in opt_cols}")

if TEST_MEASURE in enhanced_merged.columns and TEST_MEASURE in opt_merged.columns:
    # Get behavioral scores
    enhanced_scores = pd.to_numeric(enhanced_merged[TEST_MEASURE], errors='coerce').values
    opt_scores = pd.to_numeric(opt_merged[TEST_MEASURE], errors='coerce').values
    
    print()
    print(f"Enhanced {TEST_MEASURE}:")
    print(f"  N = {len(enhanced_scores)}")
    print(f"  Non-NaN = {(~np.isnan(enhanced_scores)).sum()}")
    print(f"  Range = [{np.nanmin(enhanced_scores):.2f}, {np.nanmax(enhanced_scores):.2f}]")
    print(f"  Mean = {np.nanmean(enhanced_scores):.2f}")
    
    print()
    print(f"Optimized {TEST_MEASURE}:")
    print(f"  N = {len(opt_scores)}")
    print(f"  Non-NaN = {(~np.isnan(opt_scores)).sum()}")
    print(f"  Range = [{np.nanmin(opt_scores):.2f}, {np.nanmax(opt_scores):.2f}]")
    print(f"  Mean = {np.nanmean(opt_scores):.2f}")
    
    # Test with simple PCA + LinearRegression (same as enhanced)
    print()
    print_step("Testing with same method as enhanced (PCA 80% + LinearRegression)", "")
    
    # Get IG features from merged df
    roi_cols_common = [col for col in opt_roi_cols if col in opt_merged.columns]
    X_opt = opt_merged[roi_cols_common].values
    y_opt = opt_scores
    
    # Remove NaN
    valid_mask = ~np.isnan(y_opt)
    X_valid = X_opt[valid_mask]
    y_valid = y_opt[valid_mask]
    
    print_info(f"Valid subjects for optimization: {len(y_valid)}")
    
    # Simple PCA + LinearRegression (mimics enhanced)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)
    
    pca = PCA(n_components=min(50, len(y_valid), X_valid.shape[1]))
    pca_scores = pca.fit_transform(X_scaled)
    
    # Use first N components that explain 80% variance
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    n_components_80 = np.argmax(cumsum_var >= 0.80) + 1
    
    print_info(f"PCA components for 80% variance: {n_components_80}")
    
    pca_scores_subset = pca_scores[:, :n_components_80]
    
    # Fit LinearRegression
    model = LinearRegression()
    model.fit(pca_scores_subset, y_valid)
    y_pred = model.predict(pca_scores_subset)
    
    rho, p_value = spearmanr(y_valid, y_pred)
    
    print()
    print("="*80)
    print("BASELINE TEST (PCA 80% + LinearRegression)")
    print("="*80)
    print(f"  N subjects: {len(y_valid)}")
    print(f"  N components: {n_components_80}")
    print(f"  Spearman ρ: {rho:.3f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Predicted range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
    print(f"  Actual range: [{y_valid.min():.2f}, {y_valid.max():.2f}]")
    print("="*80)
    print()
    
    if rho < 0.3:
        print("⚠️  WARNING: Even baseline is poor! Data may be different from enhanced script.")
        print("   Check that you're analyzing the same behavioral measure.")
    elif rho >= 0.35:
        print("✅ Baseline matches enhanced script! Optimization should work now.")
    else:
        print("⚠️  Baseline is moderate. Double-check data matches enhanced script.")

else:
    print_warning(f"Test measure '{TEST_MEASURE}' not found in both datasets!")
    print(f"Enhanced columns: {enhanced_cols[:10]}...")
    print(f"Optimized columns: {opt_cols[:10]}...")

print()
print_section_header("DIAGNOSTIC COMPLETE")
print("If baseline ρ ≈ 0.41, optimization should work correctly.")
print("If baseline ρ is poor, there's a data loading difference to investigate.")

