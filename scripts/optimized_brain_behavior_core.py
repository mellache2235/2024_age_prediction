#!/usr/bin/env python3
"""
Core Optimization Module for Brain-Behavior Analysis

This module contains the universal optimization logic that can be used
by all cohort-specific brain-behavior scripts.

Comprehensive optimization strategies:
1. PCA + Regression Models  
2. PLS Regression
3. Feature Selection + Regression
4. Direct Regularized Regression

Author: Brain-Behavior Optimization Team
Date: 2024
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

# Configuration constants
MAX_N_PCS = 50
PC_STEP = 5
ALPHA_RANGE = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
MAX_PLS_COMPONENTS = 30
PLS_STEP = 3
TOP_K_FEATURES = [50, 100, 150, 200]
OUTER_CV_FOLDS = 5

# Random seed for reproducibility
RANDOM_SEED = 42


def spearman_scorer(y_true, y_pred):
    """Custom scorer for Spearman correlation."""
    # Handle constant predictions gracefully
    if len(np.unique(y_pred)) == 1 or len(np.unique(y_true)) == 1:
        return 0.0  # Return 0 for constant predictions instead of NaN
    
    rho, _ = spearmanr(y_true, y_pred)
    
    # Handle NaN results
    if np.isnan(rho):
        return 0.0
    
    return rho


def determine_pc_range(n_samples, n_features):
    """Determine appropriate PC range based on sample size."""
    max_pcs_for_sample = min(MAX_N_PCS, n_samples - 10, n_features)
    pc_range = list(range(PC_STEP, max_pcs_for_sample + 1, PC_STEP))
    
    if len(pc_range) < 3:
        pc_range = [max(2, n_samples // 4), max(3, n_samples // 2), max(4, n_samples - 5)]
    
    return pc_range


def optimize_comprehensive(X, y, measure_name, verbose=True, random_seed=None):
    """
    Comprehensive optimization to maximize Spearman correlation.
    
    Args:
        X: Feature matrix
        y: Target variable
        measure_name: Name of behavioral measure
        verbose: Print progress
        random_seed: Random seed for reproducibility (default: 42)
    
    Returns best model, best parameters, and CV performance.
    """
    if random_seed is None:
        random_seed = RANDOM_SEED
    
    if verbose:
        print(f"\n  Optimizing {measure_name}... (seed={random_seed})")
    
    # Set random seeds for reproducibility
    np.random.seed(random_seed)
    
    spearman_score = make_scorer(spearman_scorer)
    outer_cv = KFold(n_splits=OUTER_CV_FOLDS, shuffle=True, random_state=random_seed)
    
    best_score = -np.inf
    best_params = {}
    best_model = None
    all_results = []
    
    # ========================================================================
    # STRATEGY 1: PCA + REGRESSION MODELS
    # ========================================================================
    n_pcs_range = determine_pc_range(len(y), X.shape[1])
    
    for n_components in n_pcs_range:
        pca = PCA(n_components=n_components)
        
        # Linear Regression
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
        
        # Regularized models
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
    
    # ========================================================================
    # STRATEGY 2: PLS REGRESSION
    # ========================================================================
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
    
    # ========================================================================
    # STRATEGY 3: FEATURE SELECTION + REGRESSION
    # ========================================================================
    max_k = min(max(TOP_K_FEATURES), X.shape[1])
    k_values = [k for k in TOP_K_FEATURES if k <= max_k]
    if not k_values:
        k_values = [max_k // 2, max_k]
    
    for fs_method in ['f_regression', 'mutual_info']:
        for k in k_values:
            selector = SelectKBest(f_regression if fs_method == 'f_regression' else mutual_info_regression, k=k)
            
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
                        selector_final = SelectKBest(
                            f_regression if fs_method == 'f_regression' else mutual_info_regression, k=k
                        )
                        best_model = Pipeline([
                            ('scaler', StandardScaler()),
                            ('selector', selector_final),
                            ('regressor', model_class(alpha=alpha, max_iter=10000))
                        ])
                        best_model.fit(X, y)
    
    # ========================================================================
    # STRATEGY 4: DIRECT REGULARIZED REGRESSION
    # ========================================================================
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
    
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('mean_cv_spearman', ascending=False)
    
    if verbose:
        print(f"    Best: {best_params.get('strategy', 'N/A')} (ρ={best_score:.4f})")
    
    return best_model, best_params, best_score, results_df


def evaluate_model(model, X, y, verbose=True):
    """Evaluate the best model on all data and return metrics."""
    y_pred = model.predict(X)
    rho, p_value = spearmanr(y, y_pred)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    if verbose:
        print(f"\n  📊 PREDICTION INTEGRITY CHECK:")
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


def remove_outliers(X, y, iqr_multiplier=3, random_seed=None):
    """
    Remove outliers using IQR method.
    
    Args:
        X: Feature matrix
        y: Target variable
        iqr_multiplier: IQR multiplier for outlier detection (default: 3)
        random_seed: Random seed (unused here, but kept for API consistency)
    
    Returns: X_clean, y_clean, n_outliers
    """
    q1 = np.percentile(y, 25)
    q3 = np.percentile(y, 75)
    iqr = q3 - q1
    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr
    
    mask = (y >= lower_bound) & (y <= upper_bound)
    n_outliers = len(y) - mask.sum()
    
    return X[mask], y[mask], n_outliers

