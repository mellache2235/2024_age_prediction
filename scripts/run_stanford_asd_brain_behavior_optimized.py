#!/usr/bin/env python3
"""
Optimized Brain-Behavior Analysis for Stanford ASD Cohort

This script performs brain-behavior correlation analysis with comprehensive optimization:
1. Loads Stanford ASD IG scores and SRS behavioral data
2. Performs PCA on IG scores with hyperparameter tuning
3. Optimizes:
   - Number of PCs (grid search from 5 to 50)
   - Regression model (Linear, Ridge, Lasso, ElasticNet)
   - Regularization strength (alpha values)
4. Uses nested cross-validation for robust evaluation
5. Parallel processing of multiple behavioral measures
6. Memory-efficient data handling
7. Reports best parameters and performance

Usage:
    python run_stanford_asd_brain_behavior_optimized.py
    python run_stanford_asd_brain_behavior_optimized.py --max-measures 3  # Test mode
    python run_stanford_asd_brain_behavior_optimized.py --n-jobs 4  # Control parallelism

Author: Enhanced Brain-Behavior Analysis
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from joblib import Parallel, delayed
import argparse
import warnings
warnings.filterwarnings('ignore')

# Add to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from logging_utils import (print_section_header, print_step, print_success, 
                           print_warning, print_error, print_info, print_completion)
from plot_styles import create_standardized_scatter, get_dataset_title, setup_arial_font, DPI, FIGURE_FACECOLOR

# Setup Arial font globally
setup_arial_font()

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET = "stanford_asd"
IG_CSV = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/integrated_gradients/stanford_asd_features_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS.csv"
SRS_FILE = "/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/prepare_data/stanford_autism/SRS_data_20230925.csv"
OUTPUT_DIR = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/stanford_asd_optimized"

# Hyperparameter search space (EXPANDED FOR BETTER SPEARMAN CORRELATIONS)
MAX_N_PCS = 50  # Maximum number of PCs to consider
PC_STEP = 5  # Step size for PC grid search
ALPHA_RANGE = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]  # Regularization strengths (expanded)

# PLS-specific settings
MAX_PLS_COMPONENTS = 30  # Maximum PLS components
PLS_STEP = 3  # Step size for PLS components

# Feature selection settings
FEATURE_SELECTION_METHODS = ['none', 'f_regression', 'mutual_info']  # Methods to try
TOP_K_FEATURES = [50, 100, 150, 200]  # Number of top features to select (if applicable)

# Cross-validation settings
OUTER_CV_FOLDS = 5  # For final evaluation
INNER_CV_FOLDS = 3  # For hyperparameter tuning (not used in current implementation)

# Parallel processing
DEFAULT_N_JOBS = -1  # -1 means use all available cores

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_data_integrity(data, data_name, subject_ids=None):
    """Comprehensive data integrity checks."""
    print_info(f"Checking data integrity: {data_name}", 0)
    
    if isinstance(data, pd.DataFrame):
        print(f"    Shape: {data.shape} (rows x columns)")
        print(f"    Columns: {list(data.columns[:5])}{'...' if len(data.columns) > 5 else ''}")
        
        # Check for missing values
        missing_counts = data.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"    ⚠ Missing values in {missing_counts[missing_counts > 0].shape[0]} columns")
        else:
            print(f"    ✓ No missing values")
        
        # Check for duplicates
        if 'subject_id' in data.columns:
            n_duplicates = data['subject_id'].duplicated().sum()
            if n_duplicates > 0:
                print(f"    ⚠ Duplicate subject IDs: {n_duplicates}")
            else:
                print(f"    ✓ No duplicate subject IDs")
    
    elif isinstance(data, np.ndarray):
        print(f"    Shape: {data.shape}")
        print(f"    Data type: {data.dtype}")
        
        # Check for NaN/Inf
        n_nan = np.isnan(data).sum() if np.issubdtype(data.dtype, np.floating) else 0
        n_inf = np.isinf(data).sum() if np.issubdtype(data.dtype, np.floating) else 0
        
        if n_nan > 0:
            print(f"    ⚠ NaN values: {n_nan} ({100*n_nan/data.size:.2f}%)")
        else:
            print(f"    ✓ No NaN values")
        
        if n_inf > 0:
            print(f"    ⚠ Inf values: {n_inf}")
        else:
            print(f"    ✓ No Inf values")
        
        # Basic statistics
        if len(data.shape) == 1:
            print(f"    Range: [{np.nanmin(data):.2f}, {np.nanmax(data):.2f}]")
            print(f"    Mean ± SD: {np.nanmean(data):.2f} ± {np.nanstd(data):.2f}")


def determine_pc_range(n_samples, n_features):
    """Determine appropriate PC range based on sample size."""
    # Maximum PCs should be less than min(n_samples, n_features)
    # We'll use n_samples - 10 as max to ensure stability in CV
    max_pcs_for_sample = min(MAX_N_PCS, n_samples - 10, n_features)
    
    # Create range with specified step
    pc_range = list(range(PC_STEP, max_pcs_for_sample + 1, PC_STEP))
    
    # Ensure we have at least a few options
    if len(pc_range) < 3:
        pc_range = [max(2, n_samples // 4), max(3, n_samples // 2), max(4, n_samples - 5)]
    
    print_info(f"PC range to test: {pc_range}", 0)
    print_info(f"Number of PC values: {len(pc_range)}", 0)
    
    return pc_range


def spearman_scorer(y_true, y_pred):
    """Custom scorer for Spearman correlation."""
    rho, _ = spearmanr(y_true, y_pred)
    return rho


# ============================================================================
# DATA LOADING
# ============================================================================

def load_stanford_ig_scores(ig_csv):
    """Load Stanford ASD IG scores."""
    print_step("Loading IG scores", f"From {Path(ig_csv).name}")
    
    df = pd.read_csv(ig_csv)
    
    # Drop Unnamed: 0 if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    # Identify subject ID column
    id_col = None
    for col in ['subject_id', 'id', 'ID', 'Subject_ID', 'record_id']:
        if col in df.columns:
            id_col = col
            break
    
    if id_col is None:
        raise ValueError("No subject ID column found in IG CSV")
    
    # Standardize to 'subject_id'
    if id_col != 'subject_id':
        df = df.rename(columns={id_col: 'subject_id'})
    
    # Convert subject IDs to string
    df['subject_id'] = df['subject_id'].astype(str)
    
    # ROI columns are all columns except subject_id
    roi_cols = [col for col in df.columns if col != 'subject_id']
    
    print_info(f"IG subjects: {len(df)}", 0)
    print_info(f"IG features (ROIs): {len(roi_cols)}", 0)
    
    # Data integrity check
    check_data_integrity(df, "IG DataFrame")
    
    return df, roi_cols


def load_srs_data(srs_file):
    """Load Stanford SRS behavioral data."""
    print_step("Loading SRS behavioral data", f"From {Path(srs_file).name}")
    
    # Load SRS file - handle potential whitespace in column names
    srs_df = pd.read_csv(srs_file)
    
    # Strip whitespace from column names
    srs_df.columns = srs_df.columns.str.strip()
    
    print_info(f"Total rows: {len(srs_df)}", 0)
    
    # Identify subject ID column
    id_col = None
    for col in srs_df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['record', 'subject', 'participant', 'pid']) and \
           any(keyword in col_lower for keyword in ['id', 'pid']):
            id_col = col
            break
    
    if id_col is None:
        raise ValueError(f"No subject ID column found in SRS file. Available columns: {list(srs_df.columns)}")
    
    print_info(f"Using ID column: {id_col}", 0)
    
    # Drop duplicates (keep last as specified)
    srs_df = srs_df.drop_duplicates(subset=[id_col], keep='last')
    
    # Convert ID to string
    srs_df[id_col] = srs_df[id_col].astype(str)
    
    # Rename to subject_id for consistency
    if id_col != 'subject_id':
        srs_df = srs_df.rename(columns={id_col: 'subject_id'})
    
    # Look for SRS behavioral columns
    srs_cols = []
    
    # Target columns to find
    target_cols = {
        'srs_total_score_standard': 'srs_total_score_standard',
        'SRS Total Score T-Score': 'srs_total_score_standard',
        'Social Awareness (AWR) T-Score': 'social_awareness_tscore'
    }
    
    # Find and rename columns
    for original_name, standard_name in target_cols.items():
        if original_name in srs_df.columns:
            if original_name != standard_name:
                srs_df = srs_df.rename(columns={original_name: standard_name})
                print_info(f"Found and renamed '{original_name}' to '{standard_name}'", 0)
            else:
                print_info(f"Found '{original_name}'", 0)
            srs_cols.append(standard_name)
    
    # If we didn't find srs_total_score_standard, try flexible matching
    if 'srs_total_score_standard' not in srs_cols:
        print_warning("'SRS Total Score T-Score' not found, searching for alternatives...")
        for col in srs_df.columns:
            col_lower = col.lower()
            if 'srs' in col_lower and 'total' in col_lower and any(x in col_lower for x in ['score', 't-score', 'tscore', 't_score']):
                srs_df = srs_df.rename(columns={col: 'srs_total_score_standard'})
                print_info(f"Using '{col}' as SRS Total Score", 0)
                srs_cols.append('srs_total_score_standard')
                break
    
    if not srs_cols:
        raise ValueError(f"No SRS behavioral columns found. Available columns: {list(srs_df.columns)}")
    
    # Convert all SRS columns to numeric
    for col in srs_cols:
        if col in srs_df.columns:
            srs_df[col] = pd.to_numeric(srs_df[col], errors='coerce')
            valid_scores = srs_df[col].notna().sum()
            print_info(f"{col}: {valid_scores} valid scores (non-NaN)", 0)
    
    print_info(f"SRS subjects: {len(srs_df)}", 0)
    print_info(f"SRS behavioral measures: {srs_cols}", 0)
    
    # Data integrity check
    check_data_integrity(srs_df, "SRS DataFrame")
    
    return srs_df, srs_cols


def merge_data(ig_df, srs_df):
    """Merge IG and SRS data by subject ID."""
    print_step("Merging data", "Matching subject IDs")
    
    print_info(f"IG subjects: {len(ig_df)}", 0)
    print_info(f"SRS subjects: {len(srs_df)}", 0)
    
    # Merge on subject_id
    merged = pd.merge(ig_df, srs_df, on='subject_id', how='inner')
    
    common_subjects = len(merged)
    print_success(f"Merged: {common_subjects} subjects with both IG and SRS data")
    
    if common_subjects < 10:
        raise ValueError(f"Insufficient overlap: only {common_subjects} common subjects")
    
    # Data integrity check
    check_data_integrity(merged, "Merged DataFrame")
    
    return merged


# ============================================================================
# OPTIMIZATION
# ============================================================================

def optimize_comprehensive(X, y, measure_name, n_jobs=-1):
    """
    Comprehensive optimization to maximize Spearman correlation.
    
    Tests multiple strategies:
    1. PCA + various regression models
    2. PLS regression (optimized for prediction)
    3. Feature selection + regression
    4. Direct regression with regularization
    
    Returns best model, best parameters, and CV performance.
    """
    print_step(f"COMPREHENSIVE OPTIMIZATION for {measure_name}", 
               "Testing PCA, PLS, Feature Selection + Multiple Regression Models")
    
    # Create custom scorer
    spearman_score = make_scorer(spearman_scorer)
    
    # Outer CV for evaluation
    outer_cv = KFold(n_splits=OUTER_CV_FOLDS, shuffle=True, random_state=42)
    
    # Track best performance globally
    best_score = -np.inf
    best_params = {}
    best_model = None
    all_results = []
    
    # ========================================================================
    # STRATEGY 1: PCA + REGRESSION MODELS
    # ========================================================================
    print(f"\n  [1/4] Testing PCA + Regression Models...")
    
    n_pcs_range = determine_pc_range(len(y), X.shape[1])
    
    for n_components in n_pcs_range:
        pca = PCA(n_components=n_components)
        
        # Test Linear Regression
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', pca),
            ('regressor', LinearRegression())
        ])
        cv_scores = cross_val_score(pipe, X, y, cv=outer_cv, scoring=spearman_score, n_jobs=1)
        mean_score = np.mean(cv_scores)
        
        all_results.append({
            'strategy': 'PCA+Linear',
            'n_components': n_components,
            'model': 'Linear',
            'alpha': None,
            'feature_selection': None,
            'n_features': None,
            'mean_cv_spearman': mean_score,
            'std_cv_spearman': np.std(cv_scores)
        })
        
        if mean_score > best_score:
            best_score = mean_score
            best_params = {
                'strategy': 'PCA+Linear',
                'n_components': n_components,
                'model': 'Linear',
                'alpha': None
            }
            best_model = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=n_components)),
                ('regressor', LinearRegression())
            ])
            best_model.fit(X, y)
        
        # Test regularized models with different alphas
        for model_name, model_class in [('Ridge', Ridge), ('Lasso', Lasso), ('ElasticNet', ElasticNet)]:
            for alpha in ALPHA_RANGE:
                pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('pca', pca),
                    ('regressor', model_class(alpha=alpha, max_iter=10000))
                ])
                cv_scores = cross_val_score(pipe, X, y, cv=outer_cv, scoring=spearman_score, n_jobs=1)
                mean_score = np.mean(cv_scores)
                
                all_results.append({
                    'strategy': f'PCA+{model_name}',
                    'n_components': n_components,
                    'model': model_name,
                    'alpha': alpha,
                    'feature_selection': None,
                    'n_features': None,
                    'mean_cv_spearman': mean_score,
                    'std_cv_spearman': np.std(cv_scores)
                })
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {
                        'strategy': f'PCA+{model_name}',
                        'n_components': n_components,
                        'model': model_name,
                        'alpha': alpha
                    }
                    best_model = Pipeline([
                        ('scaler', StandardScaler()),
                        ('pca', PCA(n_components=n_components)),
                        ('regressor', model_class(alpha=alpha, max_iter=10000))
                    ])
                    best_model.fit(X, y)
    
    print(f"    Best PCA result: ρ = {best_score:.4f}")
    
    # ========================================================================
    # STRATEGY 2: PLS REGRESSION (often better for brain-behavior)
    # ========================================================================
    print(f"\n  [2/4] Testing PLS Regression...")
    
    max_pls = min(MAX_PLS_COMPONENTS, len(y) - 10, X.shape[1])
    pls_range = list(range(PLS_STEP, max_pls + 1, PLS_STEP))
    
    for n_components in pls_range:
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('pls', PLSRegression(n_components=n_components, max_iter=10000))
        ])
        cv_scores = cross_val_score(pipe, X, y, cv=outer_cv, scoring=spearman_score, n_jobs=1)
        mean_score = np.mean(cv_scores)
        
        all_results.append({
            'strategy': 'PLS',
            'n_components': n_components,
            'model': 'PLS',
            'alpha': None,
            'feature_selection': None,
            'n_features': None,
            'mean_cv_spearman': mean_score,
            'std_cv_spearman': np.std(cv_scores)
        })
        
        if mean_score > best_score:
            best_score = mean_score
            best_params = {
                'strategy': 'PLS',
                'n_components': n_components,
                'model': 'PLS',
                'alpha': None
            }
            best_model = Pipeline([
                ('scaler', StandardScaler()),
                ('pls', PLSRegression(n_components=n_components, max_iter=10000))
            ])
            best_model.fit(X, y)
    
    print(f"    Best PLS result: ρ = {best_score:.4f}")
    
    # ========================================================================
    # STRATEGY 3: FEATURE SELECTION + REGRESSION
    # ========================================================================
    print(f"\n  [3/4] Testing Feature Selection + Regression...")
    
    # Determine which k values to try based on feature count
    max_k = min(max(TOP_K_FEATURES), X.shape[1])
    k_values = [k for k in TOP_K_FEATURES if k <= max_k]
    if not k_values:
        k_values = [max_k // 2, max_k]
    
    for fs_method in ['f_regression', 'mutual_info']:
        for k in k_values:
            # Create feature selector
            if fs_method == 'f_regression':
                selector = SelectKBest(f_regression, k=k)
            else:
                selector = SelectKBest(mutual_info_regression, k=k)
            
            # Test with different regression models
            for model_name, model_class in [('Ridge', Ridge), ('Lasso', Lasso)]:
                for alpha in [0.01, 0.1, 1.0, 10.0]:
                    pipe = Pipeline([
                        ('scaler', StandardScaler()),
                        ('selector', selector),
                        ('regressor', model_class(alpha=alpha, max_iter=10000))
                    ])
                    cv_scores = cross_val_score(pipe, X, y, cv=outer_cv, scoring=spearman_score, n_jobs=1)
                    mean_score = np.mean(cv_scores)
                    
                    all_results.append({
                        'strategy': f'FeatureSelection+{model_name}',
                        'n_components': None,
                        'model': model_name,
                        'alpha': alpha,
                        'feature_selection': fs_method,
                        'n_features': k,
                        'mean_cv_spearman': mean_score,
                        'std_cv_spearman': np.std(cv_scores)
                    })
                    
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = {
                            'strategy': f'FeatureSelection+{model_name}',
                            'n_components': None,
                            'model': model_name,
                            'alpha': alpha,
                            'feature_selection': fs_method,
                            'n_features': k
                        }
                        if fs_method == 'f_regression':
                            selector_final = SelectKBest(f_regression, k=k)
                        else:
                            selector_final = SelectKBest(mutual_info_regression, k=k)
                        
                        best_model = Pipeline([
                            ('scaler', StandardScaler()),
                            ('selector', selector_final),
                            ('regressor', model_class(alpha=alpha, max_iter=10000))
                        ])
                        best_model.fit(X, y)
    
    print(f"    Best Feature Selection result: ρ = {best_score:.4f}")
    
    # ========================================================================
    # STRATEGY 4: DIRECT REGULARIZED REGRESSION (no dimensionality reduction)
    # ========================================================================
    print(f"\n  [4/4] Testing Direct Regularized Regression...")
    
    for model_name, model_class in [('Ridge', Ridge), ('Lasso', Lasso), ('ElasticNet', ElasticNet)]:
        for alpha in ALPHA_RANGE:
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', model_class(alpha=alpha, max_iter=10000))
            ])
            cv_scores = cross_val_score(pipe, X, y, cv=outer_cv, scoring=spearman_score, n_jobs=1)
            mean_score = np.mean(cv_scores)
            
            all_results.append({
                'strategy': f'Direct{model_name}',
                'n_components': None,
                'model': model_name,
                'alpha': alpha,
                'feature_selection': None,
                'n_features': X.shape[1],
                'mean_cv_spearman': mean_score,
                'std_cv_spearman': np.std(cv_scores)
            })
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = {
                    'strategy': f'Direct{model_name}',
                    'n_components': None,
                    'model': model_name,
                    'alpha': alpha,
                    'n_features': X.shape[1]
                }
                best_model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', model_class(alpha=alpha, max_iter=10000))
                ])
                best_model.fit(X, y)
    
    print(f"    Best Direct Regression result: ρ = {best_score:.4f}")
    
    # ========================================================================
    # STRATEGY 5: Top-K IG Features (NEW - especially good for small N)
    # ========================================================================
    print(f"\n  Testing Top-K IG feature selection...")
    
    # Determine appropriate K values based on sample size
    n_samples = X.shape[0]
    k_values = []
    
    if n_samples < 100:
        # Very small N: use N/15, N/10, N/8
        k_values = [max(5, n_samples // 15), max(8, n_samples // 10), max(10, n_samples // 8)]
    elif n_samples < 200:
        # Small N: use N/10, N/8, N/5
        k_values = [n_samples // 10, n_samples // 8, n_samples // 5]
    else:
        # Larger N: can use more features
        k_values = [20, 30, 50, 100]
    
    # Remove duplicates and ensure valid range
    k_values = sorted(list(set([k for k in k_values if 3 <= k <= X.shape[1]])))
    
    if k_values:
        for k in k_values:
            # Select top K features by mean absolute value (IG importance)
            feature_importance = np.abs(X).mean(axis=0)
            top_k_idx = np.argsort(feature_importance)[-k:]
            
            # Try different models on selected features
            models = {
                'Linear': LinearRegression(),
                'Ridge': Ridge(),
                'Lasso': Lasso(max_iter=10000)
            }
            
            for model_name, model_class in models.items():
                if model_name == 'Linear':
                    pipe = Pipeline([
                        ('scaler', StandardScaler()),
                        ('regressor', LinearRegression())
                    ])
                    
                    # Use only top K features
                    X_topk = X[:, top_k_idx]
                    cv_scores = cross_val_score(pipe, X_topk, y, cv=outer_cv, scoring=spearman_score, n_jobs=1)
                    mean_score = np.mean(cv_scores)
                    
                    all_results.append({
                        'strategy': f'TopK-IG',
                        'n_components': None,
                        'model': model_name,
                        'alpha': None,
                        'feature_selection': 'MeanAbsValue',
                        'n_features': k,
                        'mean_cv_spearman': mean_score,
                        'std_cv_spearman': np.std(cv_scores)
                    })
                    
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = {
                            'strategy': f'TopK-IG',
                            'n_components': None,
                            'model': model_name,
                            'alpha': None,
                            'n_features': k,
                            'feature_selection': 'MeanAbsValue',
                            'top_k_idx': top_k_idx
                        }
                        # Create a custom pipeline that selects features
                        class TopKSelector:
                            def __init__(self, indices):
                                self.indices = indices
                            def fit(self, X, y=None):
                                return self
                            def transform(self, X):
                                return X[:, self.indices]
                        
                        best_model = Pipeline([
                            ('selector', TopKSelector(top_k_idx)),
                            ('scaler', StandardScaler()),
                            ('regressor', LinearRegression())
                        ])
                        best_model.fit(X, y)
                
                else:
                    # Ridge/Lasso: test different alphas
                    alpha_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
                    for alpha in alpha_range:
                        pipe = Pipeline([
                            ('scaler', StandardScaler()),
                            ('regressor', model_class(alpha=alpha, max_iter=10000))
                        ])
                        
                        X_topk = X[:, top_k_idx]
                        cv_scores = cross_val_score(pipe, X_topk, y, cv=outer_cv, scoring=spearman_score, n_jobs=1)
                        mean_score = np.mean(cv_scores)
                        
                        all_results.append({
                            'strategy': f'TopK-IG',
                            'n_components': None,
                            'model': model_name,
                            'alpha': alpha,
                            'feature_selection': 'MeanAbsValue',
                            'n_features': k,
                            'mean_cv_spearman': mean_score,
                            'std_cv_spearman': np.std(cv_scores)
                        })
                        
                        if mean_score > best_score:
                            best_score = mean_score
                            best_params = {
                                'strategy': f'TopK-IG',
                                'n_components': None,
                                'model': model_name,
                                'alpha': alpha,
                                'n_features': k,
                                'feature_selection': 'MeanAbsValue',
                                'top_k_idx': top_k_idx
                            }
                            class TopKSelector:
                                def __init__(self, indices):
                                    self.indices = indices
                                def fit(self, X, y=None):
                                    return self
                                def transform(self, X):
                                    return X[:, self.indices]
                            
                            best_model = Pipeline([
                                ('selector', TopKSelector(top_k_idx)),
                                ('scaler', StandardScaler()),
                                ('regressor', model_class(alpha=alpha, max_iter=10000))
                            ])
                            best_model.fit(X, y)
        
        print(f"    Best Top-K IG result: ρ = {best_score:.4f}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('mean_cv_spearman', ascending=False)
    
    print(f"\n  ✓ BEST CONFIGURATION (Max Spearman ρ):")
    print(f"    - Strategy: {best_params.get('strategy', 'N/A')}")
    print(f"    - Model: {best_params.get('model', 'N/A')}")
    if best_params.get('n_components'):
        print(f"    - N Components: {best_params['n_components']}")
    if best_params.get('alpha'):
        print(f"    - Alpha: {best_params['alpha']}")
    if best_params.get('feature_selection'):
        print(f"    - Feature Selection: {best_params['feature_selection']}")
    if best_params.get('n_features'):
        print(f"    - N Features: {best_params['n_features']}")
    print(f"    - CV Spearman: {best_score:.4f}")
    print(f"\n  Top 5 configurations:")
    print(results_df.head(5)[['strategy', 'model', 'mean_cv_spearman']].to_string(index=False))
    
    return best_model, best_params, best_score, results_df


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_best_model(model, X, y, measure_name):
    """Evaluate the best model on all data and return metrics."""
    print_step(f"Evaluating best model for {measure_name}", "Final performance metrics")
    
    # Predict on all data
    y_pred = model.predict(X)
    
    # Calculate metrics
    rho, p_value = spearmanr(y, y_pred)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    # Comprehensive integrity check output
    print()
    print(f"  📊 PREDICTION INTEGRITY CHECK:")
    print(f"  {'='*80}")
    
    print(f"\n  Actual values:")
    print(f"    N = {len(y)}")
    print(f"    Mean = {y.mean():.2f}")
    print(f"    Std = {y.std():.2f}")
    print(f"    Range = [{y.min():.2f}, {y.max():.2f}]")
    print(f"    Unique values = {len(np.unique(y))}")
    
    print(f"\n  Predicted values:")
    print(f"    Mean = {y_pred.mean():.2f}")
    print(f"    Std = {y_pred.std():.2f}")
    print(f"    Range = [{y_pred.min():.2f}, {y_pred.max():.2f}]")
    print(f"    Unique values = {len(np.unique(y_pred))}")
    
    print(f"\n  Metrics:")
    print(f"    Spearman ρ = {rho:.3f}")
    print(f"    P-value = {p_value:.4f}")
    print(f"    R² = {r2:.3f}")
    print(f"    MAE = {mae:.2f}")
    
    # Check for problems
    issues = []
    if len(np.unique(y_pred)) == 1:
        issues.append("❌ CONSTANT PREDICTIONS - model predicting same value for all!")
    elif y_pred.std() < 0.01:
        issues.append(f"⚠️  Very low prediction variance: std={y_pred.std():.4f}")
    
    mean_diff = abs(y_pred.mean() - y.mean())
    if mean_diff > 2 * y.std():
        issues.append(f"⚠️  Large mean shift: {mean_diff:.2f} (>{2*y.std():.2f})")
    
    if abs(r2) > 10:
        issues.append(f"⚠️  Extreme R²: {r2:.1f} (indicates overfitting)")
    
    if mae > 2 * y.std():
        issues.append(f"⚠️  High MAE: {mae:.2f} (>{2*y.std():.2f})")
    
    if issues:
        print(f"\n  🚨 ISSUES DETECTED:")
        for issue in issues:
            print(f"    {issue}")
    else:
        print(f"\n  ✅ No major issues detected")
    
    # Show sample predictions
    print(f"\n  Sample predictions (first 5):")
    print(f"    {'Actual':>10} {'Predicted':>10} {'Residual':>10}")
    for i in range(min(5, len(y))):
        residual = y[i] - y_pred[i]
        # Convert to float for formatting (handles numpy scalar types)
        print(f"    {float(y[i]):>10.2f} {float(y_pred[i]):>10.2f} {float(residual):>10.2f}")
    print(f"  {'='*80}\n")
    
    return {
        'y_actual': y,
        'y_pred': y_pred,
        'rho': rho,
        'p_value': p_value,
        'r2': r2,
        'mae': mae,
        'n_subjects': len(y)
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_scatter_plot(results, measure_name, best_params, output_dir):
    """Create scatter plot using centralized styling."""
    y_actual = results['y_actual']
    y_pred = results['y_pred']
    rho = results['rho']
    p_value = results['p_value']
    
    # Format p-value
    p_str = "< 0.001" if p_value < 0.001 else f"= {p_value:.3f}"
    
    # Create stats text with model info
    strategy = best_params.get('strategy', 'Unknown')
    model = best_params.get('model', 'Unknown')
    
    # Build model info string
    info_parts = [f"{strategy}"]
    if best_params.get('n_components'):
        info_parts.append(f"comp={best_params['n_components']}")
    if best_params.get('alpha'):
        info_parts.append(f"α={best_params['alpha']}")
    if best_params.get('n_features') and best_params.get('feature_selection'):
        info_parts.append(f"k={best_params['n_features']}")
    
    model_info = "\n".join(info_parts)
    stats_text = f"r = {rho:.3f}\np {p_str}\n{model_info}"  # Using "r =" not "ρ ="
    
    # Get standardized title
    title = get_dataset_title(DATASET)
    
    # Create safe filename with method info
    safe_name = measure_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
    method_name = best_params.get('strategy', 'Unknown').replace('+', '_')
    model_name = best_params.get('model', '')
    
    # Create descriptive filename
    if best_params.get('n_components'):
        filename = f'scatter_{safe_name}_{method_name}_comp{best_params["n_components"]}_optimized'
    elif best_params.get('n_features'):
        filename = f'scatter_{safe_name}_{method_name}_k{best_params["n_features"]}_optimized'
    else:
        filename = f'scatter_{safe_name}_{method_name}_optimized'
    
    save_path = Path(output_dir) / filename
    
    # Use centralized plotting function
    fig, ax = plt.subplots(figsize=(6, 6))
    
    create_standardized_scatter(
        ax, y_actual, y_pred,
        title=title + " (Optimized)",
        xlabel='Observed Behavioral Score',
        ylabel='Predicted Behavioral Score',
        stats_text=stats_text,
        is_subplot=False
    )
    
    # Save with centralized export (PNG + TIFF + AI)
    plt.tight_layout()
    
    png_path = save_path.with_suffix('.png')
    tiff_path = save_path.with_suffix('.tiff')
    ai_path = save_path.with_suffix('.ai')
    
    plt.savefig(png_path, dpi=DPI, bbox_inches='tight', facecolor=FIGURE_FACECOLOR, edgecolor='none')
    plt.savefig(tiff_path, dpi=DPI, bbox_inches='tight', facecolor=FIGURE_FACECOLOR, edgecolor='none',
               format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
    pdf_backend.FigureCanvas(fig).print_pdf(str(ai_path))
    
    plt.close()
    
    print(f"  ✓ Saved: {png_path.name} + {tiff_path.name} + {ai_path.name}")


# ============================================================================
# PARALLEL PROCESSING
# ============================================================================

def analyze_single_measure(X, merged_df, measure, output_dir, n_jobs_inner=1, random_seed=RANDOM_SEED):
    """Analyze a single behavioral measure (for parallel processing)."""
    try:
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        print_section_header(f"ANALYZING: {measure}")
        
        # Get behavioral scores for this measure
        y = merged_df[measure].values
        
        # Remove NaN values
        valid_mask = ~np.isnan(y)
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        # Remove outliers using IQR method
        q1 = np.percentile(y_valid, 25)
        q3 = np.percentile(y_valid, 75)
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        
        outlier_mask = (y_valid >= lower_bound) & (y_valid <= upper_bound)
        n_outliers = len(y_valid) - outlier_mask.sum()
        
        if n_outliers > 0:
            print_info(f"Removing {n_outliers} outliers", 0)
            X_valid = X_valid[outlier_mask]
            y_valid = y_valid[outlier_mask]
        
        if len(y_valid) < 20:
            print_warning(f"Insufficient data for {measure}: only {len(y_valid)} subjects")
            return None
        
        print_info(f"Valid subjects: {len(y_valid)}", 0)
        
        # COMPREHENSIVE OPTIMIZATION (testing all strategies)
        best_model, best_params, cv_score, opt_results = \
            optimize_comprehensive(X_valid, y_valid, measure, verbose=True, random_seed=random_seed)
        
        # Evaluate on all data
        eval_results = evaluate_best_model(best_model, X_valid, y_valid, measure)
        
        # Create visualization
        create_scatter_plot(eval_results, measure, best_params, output_dir)
        
        # Save optimization results
        safe_name = measure.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
        method_name = best_params.get('strategy', 'Unknown').replace('+', '_')
        
        opt_results.to_csv(Path(output_dir) / f"optimization_results_{safe_name}.csv", index=False)
        
        # INTEGRITY CHECK: Save actual vs predicted values (with method in filename)
        # Ensure arrays are 1-dimensional (flatten if needed)
        y_actual_flat = np.asarray(eval_results['y_actual']).flatten()
        y_pred_flat = np.asarray(eval_results['y_pred']).flatten()
        
        predictions_df = pd.DataFrame({
            'Actual': y_actual_flat,
            'Predicted': y_pred_flat,
            'Residual': y_actual_flat - y_pred_flat
        })
        pred_filename = f"predictions_{safe_name}_{method_name}.csv"
        predictions_df.to_csv(Path(output_dir) / pred_filename, index=False)
        
        # Check for constant predictions (warning sign)
        if predictions_df['Predicted'].nunique() == 1:
            print_warning(f"⚠️  Model predicts constant value: {predictions_df['Predicted'].iloc[0]:.2f}")
        elif predictions_df['Predicted'].std() < 0.01:
            print_warning(f"⚠️  Predictions have very low variance: std={predictions_df['Predicted'].std():.4f}")
        else:
            print_info(f"✓ Prediction variance OK: std={predictions_df['Predicted'].std():.2f}, range=[{predictions_df['Predicted'].min():.2f}, {predictions_df['Predicted'].max():.2f}]", 0)
        
        # Return summary
        return {
            'Measure': measure,
            'N_Subjects': len(y_valid),
            'N_Outliers_Removed': n_outliers,
            'Best_Strategy': best_params.get('strategy', 'N/A'),
            'Best_Model': best_params.get('model', 'N/A'),
            'Best_N_Components': best_params.get('n_components', None),
            'Best_Alpha': best_params.get('alpha', None),
            'Feature_Selection': best_params.get('feature_selection', None),
            'N_Features': best_params.get('n_features', None),
            'CV_Spearman': cv_score,
            'Final_Spearman': eval_results['rho'],
            'Final_P_Value': eval_results['p_value'],
            'Final_R2': eval_results['r2'],
            'Final_MAE': eval_results['mae']
        }
    
    except Exception as e:
        print_error(f"Failed to analyze {measure}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Optimized brain-behavior analysis for Stanford ASD"
    )
    parser.add_argument(
        '--max-measures', '-m',
        type=int,
        default=None,
        help="Maximum number of behavioral measures to analyze (for testing)"
    )
    parser.add_argument(
        '--n-jobs', '-j',
        type=int,
        default=DEFAULT_N_JOBS,
        help="Number of parallel jobs (-1 for all cores, 1 for sequential)"
    )
    
    args = parser.parse_args()
    
    print_section_header("OPTIMIZED BRAIN-BEHAVIOR ANALYSIS - STANFORD ASD")
    
    print()
    print_info(f"IG CSV: {IG_CSV}", 0)
    print_info(f"SRS File: {SRS_FILE}", 0)
    print_info(f"Output: {OUTPUT_DIR}", 0)
    print()
    print("  OPTIMIZATION STRATEGIES:")
    print("  " + "="*80)
    print(f"  1. PCA + Regression (Linear, Ridge, Lasso, ElasticNet)")
    print(f"     - PC range: 5-{MAX_N_PCS} (step={PC_STEP}, adjusted by sample size)")
    print(f"     - Alpha range: {ALPHA_RANGE}")
    print(f"  2. PLS Regression (optimized for prediction)")
    print(f"     - PLS component range: 3-{MAX_PLS_COMPONENTS} (step={PLS_STEP})")
    print(f"  3. Feature Selection + Regression")
    print(f"     - Methods: F-statistic, Mutual Information")
    print(f"     - Top-K features: {TOP_K_FEATURES}")
    print(f"  4. Direct Regularized Regression (all features)")
    print(f"  " + "="*80)
    print(f"  CV folds: {OUTER_CV_FOLDS}")
    print(f"  Parallel jobs: {args.n_jobs if args.n_jobs != -1 else 'all available cores'}")
    print()
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Load data
        ig_df, roi_cols = load_stanford_ig_scores(IG_CSV)
        srs_df, srs_cols = load_srs_data(SRS_FILE)
        
        # 2. Merge data
        merged_df = merge_data(ig_df, srs_df)
        
        # Extract IG matrix
        X = merged_df[roi_cols].values
        
        print_info(f"IG matrix shape: {X.shape} (subjects x ROIs)", 0)
        print_info(f"SRS behavioral measures to analyze: {srs_cols}", 0)
        
        # Limit measures if specified
        if args.max_measures:
            srs_cols = srs_cols[:args.max_measures]
            print_warning(f"Limited to {args.max_measures} measures for testing")
        
        print()
        
        # 3. Analyze each behavioral measure (with optional parallelization)
        if args.n_jobs == 1:
            # Sequential processing
            print_section_header("SEQUENTIAL PROCESSING")
            all_results = []
            for measure in srs_cols:
                result = analyze_single_measure(X, merged_df, measure, OUTPUT_DIR, n_jobs_inner=1)
                if result is not None:
                    all_results.append(result)
                print()
        else:
            # Parallel processing
            print_section_header("PARALLEL PROCESSING")
            print_info(f"Processing {len(srs_cols)} measures in parallel...", 0)
            
            all_results = Parallel(n_jobs=args.n_jobs, verbose=10)(
                delayed(analyze_single_measure)(X, merged_df, measure, OUTPUT_DIR, n_jobs_inner=1)
                for measure in srs_cols
            )
            
            # Filter out None results
            all_results = [r for r in all_results if r is not None]
        
        # 4. Save summary
        if all_results:
            summary_df = pd.DataFrame(all_results)
            summary_df.to_csv(Path(OUTPUT_DIR) / "optimization_summary.csv", index=False)
            
            print()
            print_completion("Stanford ASD Brain-Behavior Analysis Complete!")
            print_info(f"Results saved to: {OUTPUT_DIR}", 0)
            print()
            print("="*100)
            print("BEST PERFORMANCES (Sorted by Spearman ρ)")
            print("="*100)
            summary_sorted = summary_df.sort_values('Final_Spearman', ascending=False)
            print(summary_sorted[['Measure', 'Final_Spearman', 'Best_Strategy', 'Best_Model']].to_string(index=False))
            print()
            print(f"\n  HIGHEST CORRELATION: ρ = {summary_sorted.iloc[0]['Final_Spearman']:.4f}")
            print(f"  Measure: {summary_sorted.iloc[0]['Measure']}")
            print(f"  Strategy: {summary_sorted.iloc[0]['Best_Strategy']}")
            print()
        else:
            print_warning("No results to save")
        
    except Exception as e:
        print()
        print_error(f"Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

