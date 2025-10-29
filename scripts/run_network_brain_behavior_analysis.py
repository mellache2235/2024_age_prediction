#!/usr/bin/env python3
"""
Network-Level Brain-Behavior Analysis

Dedicated script for analyzing brain-behavior relationships at the NETWORK level
rather than ROI level. Uses Yeo network parcellation to aggregate 246 ROIs into
7-17 functional networks.

Tests multiple aggregation methods:
- mean: Simple average of IGs
- abs_mean: Average of absolute IGs  
- pos_share: Positive IG mass fraction
- neg_share: Negative IG mass fraction
- signed_share: Net IG mass fraction

Usage:
    python run_network_brain_behavior_analysis.py --cohort nki_rs_td
    python run_network_brain_behavior_analysis.py --cohort cmihbn_td --method signed_share
    python run_network_brain_behavior_analysis.py --all

Author: Brain-Behavior Network Analysis Team
Date: 2024
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import argparse
import warnings
warnings.filterwarnings('ignore')

# Add to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from logging_utils import (print_section_header, print_step, print_success, 
                           print_warning, print_error, print_info, print_completion)
from plot_styles import create_standardized_scatter, get_dataset_title, setup_arial_font, DPI, FIGURE_FACECOLOR
from optimized_brain_behavior_core import (load_yeo_network_mapping, aggregate_rois_to_networks,
                                           spearman_scorer, remove_outliers, apply_fdr_correction)

# Setup Arial font
setup_arial_font()

# ============================================================================
# COHORT CONFIGURATIONS  
# ============================================================================

COHORTS = {
    'nki_rs_td': {
        'name': 'NKI-RS TD',
        'ig_csv': '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/integrated_gradients/nki_cog_dev_wIDS_features_IG_convnet_regressor_single_model_fold_0.csv',
        'beh_dir': '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/FLUX/assessment_data',
        'beh_type': 'nki_multi',
        'output_dir': '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/nki_rs_td_networks'
    },
    'cmihbn_td': {
        'name': 'CMI-HBN TD',
        'ig_csv': '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/integrated_gradients/cmihbn_td_features_all_sites_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS_single_model_predictions.csv',
        'beh_file': '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/prepare_data/adhd/C3SR.csv',
        'beh_type': 'c3sr',
        'output_dir': '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/cmihbn_td_networks'
    },
    'cmihbn_adhd': {
        'name': 'CMI-HBN ADHD',
        'ig_csv': '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/integrated_gradients/cmihbn_adhd_no_cutoffs_features_all_sites_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS_single_model_predictions.csv',
        'beh_file': '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/prepare_data/adhd/C3SR.csv',
        'beh_type': 'c3sr',
        'output_dir': '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/cmihbn_adhd_networks'
    }
}

AGGREGATION_METHODS = ['mean', 'abs_mean', 'pos_share', 'neg_share', 'signed_share']

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_network_features(X_net, y, measure_name, method_name, output_dir):
    """
    Analyze brain-behavior relationship using network-level features.
    
    Tests Linear, Ridge, and Lasso regression.
    """
    print_step(f"Network analysis: {measure_name}", f"Method: {method_name}")
    
    # Remove NaN
    valid_mask = ~np.isnan(y)
    X_valid = X_net[valid_mask]
    y_valid = y[valid_mask]
    
    n_invalid = np.sum(~valid_mask)
    if n_invalid > 0:
        print_info(f"Removed {n_invalid} subjects with missing data")
    
    if len(y_valid) < 20:
        print_warning(f"Insufficient data: {len(y_valid)} subjects")
        return None
    
    # Remove outliers
    X_valid, y_valid, n_outliers = remove_outliers(X_valid, y_valid)
    if n_outliers > 0:
        print_info(f"Removed {n_outliers} outliers")
    
    if len(y_valid) < 20:
        print_warning(f"Insufficient data after outlier removal")
        return None
    
    print_info(f"Valid subjects: {len(y_valid)}, Network features: {X_valid.shape[1]}")
    
    # Test models
    models = {
        'Linear': LinearRegression(),
        'Ridge': [0.001, 0.01, 0.1, 1.0, 10.0],
        'Lasso': [0.001, 0.01, 0.1, 1.0, 10.0]
    }
    
    best_score = -np.inf
    best_model = None
    best_model_name = None
    best_alpha = None
    
    spearman_score = make_scorer(spearman_scorer)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Test Linear
    pipe = Pipeline([('scaler', StandardScaler()), ('regressor', LinearRegression())])
    scores = cross_val_score(pipe, X_valid, y_valid, cv=cv, scoring=spearman_score)
    mean_score = np.mean(scores)
    
    if mean_score > best_score:
        best_score = mean_score
        best_model = pipe
        best_model.fit(X_valid, y_valid)
        best_model_name = 'Linear'
        best_alpha = None
    
    # Test Ridge
    for alpha in models['Ridge']:
        pipe = Pipeline([('scaler', StandardScaler()), ('regressor', Ridge(alpha=alpha))])
        scores = cross_val_score(pipe, X_valid, y_valid, cv=cv, scoring=spearman_score)
        mean_score = np.mean(scores)
        
        if mean_score > best_score:
            best_score = mean_score
            best_model = pipe
            best_model.fit(X_valid, y_valid)
            best_model_name = 'Ridge'
            best_alpha = alpha
    
    # Test Lasso
    for alpha in models['Lasso']:
        pipe = Pipeline([('scaler', StandardScaler()), ('regressor', Lasso(alpha=alpha, max_iter=10000))])
        scores = cross_val_score(pipe, X_valid, y_valid, cv=cv, scoring=spearman_score)
        mean_score = np.mean(scores)
        
        if mean_score > best_score:
            best_score = mean_score
            best_model = pipe
            best_model.fit(X_valid, y_valid)
            best_model_name = 'Lasso'
            best_alpha = alpha
    
    # Final evaluation
    y_pred = best_model.predict(X_valid)
    rho, p_value = spearmanr(y_valid, y_pred)
    r2 = r2_score(y_valid, y_pred)
    mae = mean_absolute_error(y_valid, y_pred)
    
    print_success(f"Best: {best_model_name}" + (f"(α={best_alpha})" if best_alpha else ""))
    print_info(f"ρ = {rho:.3f}, p = {p_value:.4f}")
    
    return {
        'Measure': measure_name,
        'Aggregation_Method': method_name,
        'N_Subjects': len(y_valid),
        'N_Networks': X_valid.shape[1],
        'Best_Model': best_model_name,
        'Best_Alpha': best_alpha,
        'CV_Spearman': best_score,
        'Final_Spearman': rho,
        'Final_P_Value': p_value,
        'Final_R2': r2,
        'Final_MAE': mae,
        'y_actual': y_valid,
        'y_pred': y_pred
    }


def run_cohort_analysis(cohort_key, aggregation_methods=None):
    """Run network analysis for a cohort."""
    config = COHORTS[cohort_key]
    
    print_section_header(f"NETWORK BRAIN-BEHAVIOR ANALYSIS - {config['name']}")
    
    # Import cohort-specific data loading from optimized scripts
    # For now, simplified - you can enhance based on cohort
    
    print_warning("Network-only analysis script - Implementation in progress")
    print_info(f"Will analyze: {config['name']}")
    print_info(f"Network features: {aggregation_methods or AGGREGATION_METHODS}")
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Network-level brain-behavior analysis")
    parser.add_argument('--cohort', choices=list(COHORTS.keys()), help="Cohort to analyze")
    parser.add_argument('--all', action='store_true', help="Run all cohorts")
    parser.add_argument('--method', choices=AGGREGATION_METHODS, 
                       help="Specific aggregation method (default: test all)")
    
    args = parser.parse_args()
    
    if not args.cohort and not args.all:
        parser.error("Must specify --cohort or --all")
    
    # Determine methods to test
    methods = [args.method] if args.method else AGGREGATION_METHODS
    
    # Determine cohorts
    cohorts = list(COHORTS.keys()) if args.all else [args.cohort]
    
    # Run analyses
    results = {}
    for cohort in cohorts:
        success = run_cohort_analysis(cohort, methods)
        results[cohort] = success
    
    # Summary
    print_section_header("SUMMARY")
    for cohort, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {COHORTS[cohort]['name']:.<50} {status}")


if __name__ == "__main__":
    main()

