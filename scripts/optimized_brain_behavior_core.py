#!/usr/bin/env python3
"""
Core Optimization Module for Brain-Behavior Analysis

This module contains the universal optimization logic that can be used
by all cohort-specific brain-behavior scripts.

Comprehensive optimization strategies:
1. PCA + Regression Models  
2. PLS Regression (with adaptive component limits for numerical stability)
3. Feature Selection + Regression
4. Direct Regularized Regression
5. Top-K IG Features (especially effective for small N)
   - Selects top K ROIs by IG importance (mean absolute value)
   - Uses conservative K based on sample size (N/10 to N/15)
   - Fits simple models to avoid overfitting
6. Network Aggregation (NEW: 246 ROIs → 7-17 Yeo networks)
   - Aggregates ROIs to network-level features
   - Excellent for small N: 84:7 ratio instead of 84:246
   - Highly interpretable: network-level brain-behavior relationships
   - Tests 'mean' and 'abs_mean' aggregation methods

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
import os

# ============================================================================
# FDR CORRECTION
# ============================================================================

def apply_fdr_correction(p_values, alpha=0.05):
    """
    Apply Benjamini-Hochberg FDR correction to p-values.
    
    Args:
        p_values: Array of p-values
        alpha: Significance level (default: 0.05)
    
    Returns:
        corrected_p_values: FDR-corrected p-values
        rejected: Boolean array indicating significance after correction
    """
    try:
        # Try to use statsmodels if available
        from statsmodels.stats.multitest import multipletests
        rejected, corrected_p, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
        return corrected_p, rejected
    except ImportError:
        # Fallback: Benjamini-Hochberg implementation
        p_values = np.array(p_values)
        n = len(p_values)
        
        if n == 0:
            return np.array([]), np.array([])
        
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]
        
        # BH critical values
        bh_critical = (np.arange(1, n + 1) / n) * alpha
        
        # Find rejected hypotheses
        rejected = np.zeros(n, dtype=bool)
        corrected_p = np.ones(n)
        
        # Find largest k where p(k) <= (k/n)*alpha
        max_k = -1
        for i in range(n):
            if sorted_p_values[i] <= bh_critical[i]:
                max_k = i
        
        # Mark all up to max_k as rejected
        if max_k >= 0:
            for i in range(max_k + 1):
                idx = sorted_indices[i]
                rejected[idx] = True
                corrected_p[idx] = min(1.0, sorted_p_values[i] * n / (i + 1))
        
        return corrected_p, rejected


# Configuration constants
MAX_N_PCS = 50
PC_STEP = 5
ALPHA_RANGE = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
MAX_PLS_COMPONENTS = 30
PLS_STEP = 3
TOP_K_FEATURES = [50, 100, 150, 200]
OUTER_CV_FOLDS = 5

# Yeo atlas path for network aggregation
YEO_ATLAS_PATH = "/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/feature_attribution/csv_files/subregion_func_network_Yeo_updated_yz.csv"

# Random seed for reproducibility
RANDOM_SEED = 42


# ============================================================================
# NETWORK AGGREGATION HELPERS
# ============================================================================

def load_yeo_network_mapping(yeo_atlas_path=YEO_ATLAS_PATH):
    """
    Load Yeo network mapping for 246 Brainnetome ROIs.
    
    Returns:
        network_map: dict mapping ROI index (0-245) to network name/ID
    """
    if not os.path.exists(yeo_atlas_path):
        return None  # Fallback: network aggregation unavailable
    
    try:
        yeo_atlas = pd.read_csv(yeo_atlas_path)
        
        # Find network column (Yeo_17network or similar)
        network_col = None
        for col in yeo_atlas.columns:
            if 'network' in col.lower() and 'yeo' in col.lower():
                network_col = col
                break
        
        if network_col is None:
            return None
        
        # Create mapping: ROI index (0-based) → network ID
        network_map = {}
        for idx in range(min(len(yeo_atlas), 246)):
            network_map[idx] = yeo_atlas.iloc[idx][network_col]
        
        return network_map
    except:
        return None


def aggregate_rois_to_networks(X, network_map, method='mean'):
    """
    Aggregate ROI-level IG scores to network-level scores.
    
    Args:
        X: Feature matrix (N subjects × 246 ROIs)
        network_map: Dict mapping ROI index → network ID
        method: Aggregation method
            'mean': Simple mean of IGs within network
            'abs_mean': Mean of absolute IGs
            'pos_share': Positive mass share (sum of positive IGs / total absolute)
            'neg_share': Negative mass share (sum of negative IGs / total absolute)
            'signed_share': Net share ((positive - negative) / total absolute)
    
    Returns:
        X_networks: Feature matrix (N subjects × K networks)
        network_names: List of network names
    """
    if network_map is None:
        return None, None
    
    # Get unique networks
    unique_networks = sorted(set(network_map.values()))
    n_networks = len(unique_networks)
    n_subjects = X.shape[0]
    
    # Create network feature matrix
    X_networks = np.zeros((n_subjects, n_networks))
    
    for net_idx, network in enumerate(unique_networks):
        # Find all ROIs belonging to this network
        roi_indices = [roi_idx for roi_idx, net in network_map.items() if net == network]
        
        if roi_indices:
            X_network_rois = X[:, roi_indices]  # (subjects × ROIs in network)
            
            # Aggregate based on method
            if method == 'mean':
                X_networks[:, net_idx] = X_network_rois.mean(axis=1)
                
            elif method == 'abs_mean':
                X_networks[:, net_idx] = np.abs(X_network_rois).mean(axis=1)
                
            elif method == 'pos_share':
                # Positive mass share: sum(max(IG, 0)) / sum(|IG|) across ALL ROIs
                P_g = np.maximum(X_network_rois, 0).sum(axis=1)  # Positive mass in network
                A = np.abs(X).sum(axis=1)  # Total absolute mass (all ROIs)
                X_networks[:, net_idx] = P_g / (A + 1e-10)  # Avoid division by zero
                
            elif method == 'neg_share':
                # Negative mass share: sum(max(-IG, 0)) / sum(|IG|) across ALL ROIs
                N_g = np.maximum(-X_network_rois, 0).sum(axis=1)  # Negative mass in network
                A = np.abs(X).sum(axis=1)  # Total absolute mass
                X_networks[:, net_idx] = N_g / (A + 1e-10)
                
            elif method == 'signed_share':
                # Net share: (positive - negative) / total
                P_g = np.maximum(X_network_rois, 0).sum(axis=1)
                N_g = np.maximum(-X_network_rois, 0).sum(axis=1)
                A = np.abs(X).sum(axis=1)
                X_networks[:, net_idx] = (P_g - N_g) / (A + 1e-10)
    
    network_names = [f"Network_{net}" for net in unique_networks]
    
    return X_networks, network_names


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
        X: Feature matrix (N × 246 ROIs)
        y: Target variable
        measure_name: Name of behavioral measure
        verbose: Print progress
        random_seed: Random seed for reproducibility (default: 42)
    
    Tests models on TWO feature representations:
    - ROI-level (246 features): PCA, PLS, TopK-IG, Direct
    - Network-level (7-17 features): Simple models on aggregated networks
    
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
    # PREPROCESSING: Create Network-Level Features (if available)
    # ========================================================================
    network_map = load_yeo_network_mapping()
    network_features = {}  # Store all network aggregation variants
    n_networks = 0
    
    if network_map is not None:
        # Test multiple aggregation methods (recommended method from research)
        aggregation_methods = ['mean', 'abs_mean', 'pos_share', 'neg_share', 'signed_share']
        
        for method in aggregation_methods:
            X_net, network_names = aggregate_rois_to_networks(X, network_map, method=method)
            if X_net is not None:
                network_features[method] = X_net
                n_networks = X_net.shape[1]
        
        if verbose and n_networks > 0:
            print(f"    Network features created: 246 ROIs → {n_networks} networks ({len(network_features)} methods)")
    
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
    # CRITICAL: PLS needs MUCH stricter limits than PCA to avoid numerical instability
    # For numerical stability: max components = N/5 (better) or N/4 (aggressive)
    n_samples = len(y)
    if n_samples < 100:
        max_pls_safe = n_samples // 5  # Very conservative for small N
    elif n_samples < 200:
        max_pls_safe = n_samples // 4  # Moderate for medium N
    else:
        max_pls_safe = min(30, n_samples // 3)  # Can be more aggressive with large N
    
    max_pls = min(MAX_PLS_COMPONENTS, max_pls_safe, X.shape[1])
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
    
    # ========================================================================
    # STRATEGY 5: Top-K IG Features (especially good for small N)
    # ========================================================================
    # Select features based on mean absolute value (IG importance)
    # Use conservative K based on sample size
    
    n_samples = X.shape[0]
    k_values = []
    
    # Determine appropriate K values based on sample size
    # Rule: K should be at most N/10 for small samples, N/5 for larger
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
    
    if k_values:  # Only if we have valid K values
        for k in k_values:
            # Select top K features by mean absolute value (IG importance)
            feature_importance = np.abs(X).mean(axis=0)
            top_k_idx = np.argsort(feature_importance)[-k:]
            
            # Try different models on selected features
            models = {
                'Linear': LinearRegression,  # Pass class, not instance
                'Ridge': Ridge,              # Pass class, not instance
                'Lasso': Lasso               # Pass class, not instance
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
                            'top_k_idx': top_k_idx  # Store indices for later use
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
                    for alpha in ALPHA_RANGE:
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
    
    # ========================================================================
    # STRATEGY 6: Network-Level Features (Pre-aggregated in preprocessing)
    # ========================================================================
    # Test models on network-aggregated features (7-17 networks vs 246 ROIs)
    # Uses features created in preprocessing step above
    
    if network_features:
        # Test all aggregation methods created in preprocessing
        for agg_method, X_net in network_features.items():
            if X_net is None:
                continue
                
            # Test simple models on network features (they're already compact!)
            models_network = {
                'Linear': LinearRegression,
                'Ridge': Ridge,
                'Lasso': Lasso
            }
            
            for model_name, model_class in models_network.items():
                if model_name == 'Linear':
                    pipe = Pipeline([
                        ('scaler', StandardScaler()),
                        ('regressor', LinearRegression())
                    ])
                    
                    cv_scores = cross_val_score(pipe, X_net, y, cv=outer_cv, scoring=spearman_score, n_jobs=1)
                    mean_score = np.mean(cv_scores)
                    
                    all_results.append({
                        'strategy': f'Network-{agg_method}',
                        'n_components': None,
                        'model': model_name,
                        'alpha': None,
                        'feature_selection': 'YeoNetworks',
                        'n_features': n_networks,
                        'mean_cv_spearman': mean_score,
                        'std_cv_spearman': np.std(cv_scores)
                    })
                    
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = {
                            'strategy': f'Network-{agg_method}',
                            'n_components': None,
                            'model': model_name,
                            'alpha': None,
                            'n_features': n_networks,
                            'feature_selection': 'YeoNetworks',
                            'aggregation_method': agg_method,
                            'network_map': network_map
                        }
                        # Create network aggregator for pipeline
                        class NetworkAggregator:
                            def __init__(self, network_map, method):
                                self.network_map = network_map
                                self.method = method
                            def fit(self, X, y=None):
                                return self
                            def transform(self, X):
                                X_net, _ = aggregate_rois_to_networks(X, self.network_map, self.method)
                                return X_net
                        
                        best_model = Pipeline([
                            ('aggregator', NetworkAggregator(network_map, agg_method)),
                            ('scaler', StandardScaler()),
                            ('regressor', LinearRegression())
                        ])
                        best_model.fit(X, y)
                
                else:
                    # Ridge/Lasso: test alphas
                    for alpha in ALPHA_RANGE:
                        pipe = Pipeline([
                            ('scaler', StandardScaler()),
                            ('regressor', model_class(alpha=alpha, max_iter=10000))
                        ])
                        
                        cv_scores = cross_val_score(pipe, X_net, y, cv=outer_cv, scoring=spearman_score, n_jobs=1)
                        mean_score = np.mean(cv_scores)
                        
                        all_results.append({
                            'strategy': f'Network-{agg_method}',
                            'n_components': None,
                            'model': model_name,
                            'alpha': alpha,
                            'feature_selection': 'YeoNetworks',
                            'n_features': n_networks,
                            'mean_cv_spearman': mean_score,
                            'std_cv_spearman': np.std(cv_scores)
                        })
                        
                        if mean_score > best_score:
                            best_score = mean_score
                            best_params = {
                                'strategy': f'Network-{agg_method}',
                                'n_components': None,
                                'model': model_name,
                                'alpha': alpha,
                                'n_features': n_networks,
                                'feature_selection': 'YeoNetworks',
                                'aggregation_method': agg_method,
                                'network_map': network_map
                            }
                            class NetworkAggregator:
                                def __init__(self, network_map, method):
                                    self.network_map = network_map
                                    self.method = method
                                def fit(self, X, y=None):
                                    return self
                                def transform(self, X):
                                    X_net, _ = aggregate_rois_to_networks(X, self.network_map, self.method)
                                    return X_net
                            
                            best_model = Pipeline([
                                ('aggregator', NetworkAggregator(network_map, agg_method)),
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
        
        # CRITICAL: Check for numerical instability (catastrophic failure)
        if y_pred.std() > 1e6 or abs(y_pred.mean()) > 1e6:
            issues.append("❌ NUMERICAL EXPLOSION - Model is numerically unstable!")
            issues.append(f"    → Predicted std: {y_pred.std():.2e}, mean: {y_pred.mean():.2e}")
            issues.append(f"    → This indicates matrix near-singularity (too many components/features)")
        elif abs(r2) > 1e6:
            issues.append("❌ NUMERICAL INSTABILITY - Extreme R² indicates matrix issues")
            issues.append(f"    → R² = {r2:.2e}")
        
        # Regular checks (only if not catastrophic)
        if len(issues) == 0:
            if len(np.unique(y_pred)) == 1:
                issues.append("❌ CONSTANT PREDICTIONS - model predicting same value for all!")
            elif y_pred.std() < 0.01:
                issues.append(f"⚠️  Very low prediction variance: std={y_pred.std():.4f}")
            elif y_pred.std() < 0.2 * y.std():
                # Prediction variance is less than 20% of actual variance
                variance_ratio = (y_pred.std() / y.std()) * 100
                issues.append(f"⚠️  Low prediction variance: {y_pred.std():.2f} ({variance_ratio:.1f}% of actual {y.std():.2f})")
                issues.append(f"    → Model is predicting near-constant values (model collapse)")
            
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

