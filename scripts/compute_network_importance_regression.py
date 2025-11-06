#!/usr/bin/env python3
"""
Compute network importance via multivariate regression and leave-one-out analysis.

This script:
1. Aggregates IG attributions across 500 folds to Yeo networks
2. Fits multivariate regression (all networks → age/behavior)
3. Performs leave-one-network-out to quantify each network's contribution
4. Outputs effect sizes (performance drop when network is omitted)

Effect sizes can be visualized in radar plots to show network importance.

Usage:
    python compute_network_importance_regression.py --preset brain_age_td --effect-metric rho
    python compute_network_importance_regression.py --preset brain_age_td --effect-metric r2
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, zscore
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import yaml

try:
    from netneurotools import stats as nnt_stats
except ImportError:
    nnt_stats = None

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

sys.path.append(str(PROJECT_ROOT / "scripts"))
sys.path.append(str(PROJECT_ROOT / "utils"))

from optimized_brain_behavior_core import aggregate_rois_to_networks  # noqa: E402
from data_utils import load_finetune_dataset, load_finetune_dataset_w_ids  # noqa: E402

# Default Yeo atlas path
DEFAULT_YEO_ATLAS = Path(
    "/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/"
    "scripts/dnn/feature_attribution/csv_files/subregion_func_network_Yeo_updated_yz.csv"
)

FOLD_PATTERN = re.compile(r"_(\d+)[^/\\]*_ig\.npz$", re.IGNORECASE)

# Yeo-17 network semantic names (from reference table)
# Maps atlas numeric IDs to standardized network names
YEO17_NETWORK_NAMES = {
    "0": "Yeo17_0",           # Unassigned/other
    "1": "VisPeri",           # Visual peripheral
    "2": "VisCent",           # Visual central
    "3": "SomMot-1",          # Somato-motor A
    "4": "SomMot-2",          # Somato-motor B
    "5": "DorsAttn-1",        # Dorsal attention A
    "6": "DorsAttn-2",        # Dorsal attention B
    "7": "SalVentAttn-1",     # Ventral attention / Salience A
    "8": "SalVentAttn-2",     # Salience B
    "9": "Limbic-2",          # Limbic B
    "10": "Limbic-1",         # Limbic A
    "11": "FPN-1",            # Fronto-parietal / Control C (FPA)
    "12": "FPN-2",            # Control A (FPB)
    "13": "FPN-3",            # Control B (FPC)
    "14": "AudLang",          # Default D (Auditory/Language)
    "15": "DMN-3",            # Default C
    "16": "DMN-1",            # Default A
    "17": "DMN-2",            # Default B / TempPar
    "18": "AmyHip",           # Amygdala/Hippocampus
    "19": "Striatum",         # Striatum
    "20": "Thalamus",         # Thalamus
}

YEO7_NETWORK_NAMES = {
    "0": "Visual",
    "1": "SomMot",
    "2": "DorsAttn",
    "3": "SalVentAttn",
    "4": "Limbic",
    "5": "FrontoPar",
    "6": "Default",
    "7": "Cont",
}


def extract_fold_index(path: Path) -> Optional[int]:
    """Extract fold index from filenames like *_5_*_ig.npz."""
    match = FOLD_PATTERN.search(path.name)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    numbers = re.findall(r"\d+", path.stem)
    if numbers:
        try:
            return int(numbers[-1])
        except ValueError:
            return None
    return None


def ig_file_sort_key(path: Path) -> Tuple[int, str]:
    fold_idx = extract_fold_index(path)
    if fold_idx is None:
        return (10**9, path.name)
    return (fold_idx, path.name)


def collapse_time_dimension(attr_data: np.ndarray) -> np.ndarray:
    """Collapse time dimension of IG tensor to ROI-level values."""
    if attr_data.ndim == 3:
        return np.median(attr_data, axis=2)
    if attr_data.ndim == 2:
        return attr_data
    raise ValueError(
        f"Unexpected IG array shape {attr_data.shape}; expected 2D or 3D."
    )


def load_age_source(path: Path, value_key: Optional[str] = None) -> Tuple[Optional[List[str]], np.ndarray]:
    """Load chronological ages from .bin or NPZ files."""
    suffix = path.suffix.lower()

    if suffix == ".bin":
        try:
            _, _, id_train, y_train, y_test, id_test = load_finetune_dataset_w_ids(str(path))
            ids_train = np.asarray(id_train).ravel() if id_train is not None else None
            ids_test = np.asarray(id_test).ravel() if id_test is not None else None
            if ids_train is not None and ids_test is not None:
                subjects_array = np.concatenate([ids_train, ids_test])
            elif ids_train is not None:
                subjects_array = ids_train
            elif ids_test is not None:
                subjects_array = ids_test
            else:
                subjects_array = None

            ages = np.concatenate([np.asarray(y_train, dtype=float), np.asarray(y_test, dtype=float)])

            if subjects_array is not None:
                return [str(s) for s in subjects_array], ages
            return None, ages
        except (KeyError, AttributeError):
            pass

        _, _, y_train, y_test = load_finetune_dataset(str(path))
        ages = np.concatenate([np.asarray(y_train, dtype=float), np.asarray(y_test, dtype=float)])
        return None, ages

    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            if value_key and value_key in data:
                arr = np.asarray(data[value_key], dtype=float)
                return None, arr
            for candidate in ("ages", "age", "y", "targets", "labels", "actual"):
                if candidate in data:
                    arr = np.asarray(data[candidate], dtype=float)
                    return None, arr
        raise ValueError(
            f"Could not locate age array in NPZ file {path}."
        )

    raise ValueError(f"Unsupported age source file format for {path}.")


def load_network_mapping(atlas_path: Path, parcellation: str) -> Dict[int, str]:
    """Load ROI→network mapping for the requested parcellation."""
    parcellation = parcellation.lower()
    if not atlas_path.exists():
        raise FileNotFoundError(f"Yeo atlas CSV not found at {atlas_path}.")

    atlas_df = pd.read_csv(atlas_path)

    if parcellation not in {"yeo7", "yeo-7", "yeo17", "yeo-17"}:
        raise ValueError("Parcellation must be 'yeo7' or 'yeo17'.")

    target_key = "17" if "17" in parcellation else "7"

    candidate_cols = [
        col for col in atlas_df.columns if "yeo" in col.lower() and target_key in col.lower()
    ]
    if not candidate_cols:
        raise ValueError(
            f"Could not locate a Yeo-{target_key} network column in atlas {atlas_path}."
        )

    network_col = candidate_cols[0]
    network_map: Dict[int, str] = {}

    max_rois = min(len(atlas_df), 246)
    for row_idx in range(max_rois):
        value = atlas_df.iloc[row_idx][network_col]
        if pd.notna(value):
            network_map[row_idx] = str(value)

    if not network_map:
        raise ValueError(
            f"No network assignments found in column '{network_col}' of {atlas_path}."
        )

    unique_nets = sorted(set(network_map.values()), key=lambda x: int(x) if x.isdigit() else x)
    print(f"  Loaded {len(network_map)} ROIs → {len(unique_nets)} networks from '{network_col}'")

    return network_map


def aggregate_network_ig_across_folds(
    ig_files: Sequence[Path],
    network_map: Dict[int, str],
    aggregation_method: str,
    parcellation: str,
    verbose: bool = False,
) -> Tuple[List[str], np.ndarray]:
    """
    Aggregate IG to networks and average across folds.
    
    Returns:
        network_names: List of network labels
        network_matrix: (N_subjects, N_networks) averaged across folds
    """
    subject_network_store: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    network_names_reference: Optional[List[str]] = None

    for ig_path in ig_files:
        fold_idx = extract_fold_index(ig_path)
        if verbose and fold_idx is not None and fold_idx % 100 == 0:
            print(f"    Processing fold {fold_idx}")

        with np.load(ig_path, allow_pickle=True) as data:
            if "arr_0" not in data:
                warnings.warn(f"{ig_path.name} missing 'arr_0'; skipping.")
                continue

            attr_data = data["arr_0"]
            roi_matrix = collapse_time_dimension(attr_data)
            n_subjects, n_features = roi_matrix.shape

            if n_features < 246:
                warnings.warn(f"{ig_path.name}: expected 246 ROIs, found {n_features}.")

            if verbose and fold_idx is not None and fold_idx <= 1:
                print(f"      ROI matrix shape before aggregation: {roi_matrix.shape}")
            
            net_matrix, network_names = aggregate_rois_to_networks(
                roi_matrix, network_map, method=aggregation_method
            )

            if net_matrix is None or network_names is None:
                raise RuntimeError("Failed to aggregate ROIs to networks.")
            
            if verbose and fold_idx is not None and fold_idx <= 1:
                print(f"      Network matrix shape after aggregation: {net_matrix.shape}")
                print(f"      Network names: {network_names}")

            # Strip Network_ prefix and convert to semantic names
            network_names_numeric = [name.replace("Network_", "") for name in network_names]
            
            if network_names_reference is None:
                network_names_reference = network_names_numeric
            elif network_names_numeric != network_names_reference:
                if verbose:
                    print(f"    ⚠︎ Network mismatch in {ig_path.name}:")
                    print(f"      Expected: {network_names_reference}")
                    print(f"      Got: {network_names_numeric}")
                # Use the first fold's network list as canonical
                # Ensure consistent ordering by mapping to reference
                pass

            # Store per subject (row index as subject key since IDs not in NPZ)
            # Use numeric names for storage to ensure consistency across folds
            for subj_idx in range(n_subjects):
                for net_idx, net_name_numeric in enumerate(network_names_numeric):
                    value = net_matrix[subj_idx, net_idx]
                    if np.isfinite(value):
                        subject_network_store[subj_idx][net_name_numeric].append(float(value))

    if not subject_network_store:
        raise RuntimeError("No network data aggregated across folds.")

    network_names_numeric = network_names_reference or []
    n_subjects = max(subject_network_store.keys()) + 1

    # Convert numeric labels to semantic names
    name_map = YEO17_NETWORK_NAMES if "17" in parcellation else YEO7_NETWORK_NAMES
    
    # Filter out network "0" (unassigned ROIs) for Yeo-17
    if "17" in parcellation:
        filtered_numeric = [num for num in network_names_numeric if num != "0"]
        network_names_semantic = [name_map.get(num, num) for num in filtered_numeric]
    else:
        filtered_numeric = network_names_numeric
        network_names_semantic = [name_map.get(num, num) for num in filtered_numeric]

    # Average across folds
    network_matrix_full = np.zeros((n_subjects, len(network_names_numeric)))
    for subj_idx in range(n_subjects):
        for net_idx, net_name_numeric in enumerate(network_names_numeric):
            values = subject_network_store[subj_idx].get(net_name_numeric, [])
            network_matrix_full[subj_idx, net_idx] = np.mean(values) if values else np.nan
    
    # Filter columns for network "0" if needed
    if "17" in parcellation:
        keep_indices = [i for i, num in enumerate(network_names_numeric) if num != "0"]
        network_matrix = network_matrix_full[:, keep_indices]
    else:
        network_matrix = network_matrix_full

    return network_names_semantic, network_matrix


def compute_network_importance_dominance(
    X_networks: np.ndarray,
    y: np.ndarray,
    network_names: List[str],
    n_permutations: int = 5000,
) -> pd.DataFrame:
    """
    Compute network importance via dominance analysis (manual implementation).
    
    Computes general dominance (average incremental R² across all subset models)
    for each network, then performs permutation testing to assess significance.
    
    Args:
        X_networks: (N_subjects, N_networks) network feature matrix
        y: (N_subjects,) target values
        network_names: Network labels
        n_permutations: Number of permutations for significance testing
    
    Returns:
        DataFrame with Network, Total_Dominance, Dominance_Pct, Model_R2_Adj, P_Value
    """
    from itertools import combinations
    
    # Remove NaN rows
    mask = np.all(np.isfinite(X_networks), axis=1) & np.isfinite(y)
    X_clean = X_networks[mask]
    y_clean = y[mask]

    if X_clean.shape[0] < 10:
        raise ValueError(f"Insufficient samples after removing NaNs: {X_clean.shape[0]}")

    n_networks = len(network_names)
    print(f"  Computing dominance analysis (manual implementation)...")
    print(f"  This will fit 2^{n_networks} = {2**n_networks:,} subset models...")
    
    # Z-score networks and age
    X_z = zscore(X_clean, axis=0, nan_policy='omit')
    y_z = zscore(y_clean)
    
    # Cache for subset model R² values
    r2_cache: Dict[frozenset, float] = {}
    
    def get_r2_for_subset(predictor_indices: Sequence[int]) -> float:
        """Fit LinearRegression on subset and return R²."""
        key = frozenset(predictor_indices)
        if key in r2_cache:
            return r2_cache[key]
        
        if len(predictor_indices) == 0:
            # Null model: R² = 0
            r2_cache[key] = 0.0
            return 0.0
        
        X_subset = X_z[:, list(predictor_indices)]
        lin_reg = LinearRegression()
        lin_reg.fit(X_subset, y_z)
        yhat = lin_reg.predict(X_subset)
        
        SS_Residual = np.sum((y_z - yhat) ** 2)
        SS_Total = np.sum((y_z - np.mean(y_z)) ** 2)
        r2 = 1 - (SS_Residual / SS_Total)
        
        r2_cache[key] = r2
        return r2
    
    # Compute general dominance for each predictor
    general_dominances = []
    
    for target_idx in range(n_networks):
        incremental_r2_list = []
        
        # All other predictors (excluding target)
        others = [i for i in range(n_networks) if i != target_idx]
        
        # Iterate over all subset sizes
        for k in range(n_networks):
            # All k-sized subsets of 'others'
            for subset in combinations(others, k):
                r2_without = get_r2_for_subset(subset)
                r2_with = get_r2_for_subset(tuple(subset) + (target_idx,))
                incremental_r2 = r2_with - r2_without
                incremental_r2_list.append(incremental_r2)
        
        # General dominance = average incremental R²
        general_dom = np.mean(incremental_r2_list)
        general_dominances.append(general_dom)
    
    total_dominance = np.array(general_dominances)
    
    # Get adjusted R² of full model
    r2_full = get_r2_for_subset(tuple(range(n_networks)))
    adj_r2 = 1 - (1 - r2_full) * (len(y_z) - 1) / (len(y_z) - n_networks - 1)
    
    print(f"  Full model R²: {r2_full:.4f}, Adjusted R²: {adj_r2:.4f}")
    print(f"  Total models fitted: {len(r2_cache)}")
    
    # Permutation test for significance
    print(f"  Running permutation test ({n_permutations} permutations)...")
    null_r2 = np.zeros(n_permutations)
    
    np.random.seed(42)
    for i in range(n_permutations):
        y_perm = np.random.permutation(y_z)
        lin_reg_null = LinearRegression()
        lin_reg_null.fit(X_z, y_perm)
        yhat_null = lin_reg_null.predict(X_z)
        
        SS_Residual_null = np.sum((y_perm - yhat_null) ** 2)
        SS_Total_null = np.sum((y_perm - np.mean(y_perm)) ** 2)
        r2_null = 1 - (SS_Residual_null / SS_Total_null)
        adj_r2_null = 1 - (1 - r2_null) * (len(y_z) - 1) / (len(y_z) - n_networks - 1)
        null_r2[i] = adj_r2_null
    
    p_value = np.mean(null_r2 >= adj_r2)
    print(f"  Permutation p-value: {p_value:.4f}")
    
    # Format results
    total_dom_sum = np.sum(total_dominance)
    
    results = []
    for net_idx, net_name in enumerate(network_names):
        dom = total_dominance[net_idx]
        dom_pct = (dom / total_dom_sum * 100) if total_dom_sum != 0 else 0.0
        
        results.append({
            "Network": net_name,
            "Total_Dominance": float(dom),
            "Dominance_Pct": float(dom_pct),
            "Effect_Size": float(dom),  # Alias for radar
            "Effect_Size_Pct": float(dom_pct),
            "Model_R2_Adj": float(adj_r2),
            "P_Value": float(p_value),
            "N_Subjects": int(X_clean.shape[0]),
        })

    return pd.DataFrame(results)


def compute_network_importance_coefficients(
    X_networks: np.ndarray,
    y: np.ndarray,
    network_names: List[str],
    alpha: float = 1.0,
    as_percentage: bool = False,
) -> pd.DataFrame:
    """
    Compute network importance via standardized Ridge coefficients.
    
    After standardizing all networks to unit variance, the Ridge coefficients
    reflect each network's weight in the multivariate prediction.
    
    Args:
        X_networks: (N_subjects, N_networks) network feature matrix
        y: (N_subjects,) target values
        network_names: Network labels
        alpha: Ridge regularization parameter
        as_percentage: Express coefficient as % of sum(|β|)
    
    Returns:
        DataFrame with Network, Coefficient, Coefficient_Abs, Coefficient_Pct
    """
    # Remove NaN rows
    mask = np.all(np.isfinite(X_networks), axis=1) & np.isfinite(y)
    X_clean = X_networks[mask]
    y_clean = y[mask]

    if X_clean.shape[0] < 10:
        raise ValueError(f"Insufficient samples after removing NaNs: {X_clean.shape[0]}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # Fit Ridge regression
    model = Ridge(alpha=alpha)
    model.fit(X_scaled, y_clean)
    y_pred = model.predict(X_scaled)
    
    # Compute performance
    rho, p_value = spearmanr(y_clean, y_pred)
    r2 = r2_score(y_clean, y_pred)
    mae = mean_absolute_error(y_clean, y_pred)
    
    print(f"  Model performance: ρ={rho:.4f}, R²={r2:.4f}, MAE={mae:.2f}")

    coefficients = model.coef_
    
    # Compute percentages (normalize by sum of absolute coefficients)
    total_abs_coef = np.sum(np.abs(coefficients))
    
    results = []
    for net_idx, net_name in enumerate(network_names):
        coef = float(coefficients[net_idx])
        coef_abs = abs(coef)
        
        if as_percentage and total_abs_coef > 0:
            coef_pct = (coef_abs / total_abs_coef) * 100
        else:
            coef_pct = np.nan

        results.append({
            "Network": net_name,
            "Coefficient": coef,
            "Coefficient_Abs": coef_abs,
            "Effect_Size": coef_abs,  # Alias for radar compatibility
            "Effect_Size_Pct": float(coef_pct) if not np.isnan(coef_pct) else coef_abs,
            "Model_Spearman_rho": float(rho),
            "Model_R2": float(r2),
            "Model_MAE": float(mae),
            "N_Subjects": int(X_clean.shape[0]),
        })

    return pd.DataFrame(results)


def compute_network_importance_permutation(
    X_networks: np.ndarray,
    y: np.ndarray,
    network_names: List[str],
    effect_metric: str = "rho",
    alpha: float = 1.0,
    n_permutations: int = 20,
    use_absolute: bool = True,
    as_percentage: bool = False,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Compute network importance via permutation.
    
    For each network, shuffle its values n_permutations times, refit the model,
    and measure the average performance drop.
    
    Args:
        X_networks: (N_subjects, N_networks) network feature matrix
        y: (N_subjects,) target values
        network_names: Network labels
        effect_metric: 'rho' (Spearman), 'r2', or 'mae'
        alpha: Ridge regularization parameter
        n_permutations: Number of permutations per network
        use_absolute: Take absolute value of effect size
        as_percentage: Express effect size as % of baseline
        random_seed: Random seed for reproducibility
    
    Returns:
        DataFrame with Network, Baseline_Performance, Mean_Permuted_Performance, Effect_Size, Effect_Size_Std
    """
    np.random.seed(random_seed)
    
    # Remove NaN rows
    mask = np.all(np.isfinite(X_networks), axis=1) & np.isfinite(y)
    X_clean = X_networks[mask]
    y_clean = y[mask]

    if X_clean.shape[0] < 10:
        raise ValueError(f"Insufficient samples after removing NaNs: {X_clean.shape[0]}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # Baseline: all networks
    model = Ridge(alpha=alpha)
    model.fit(X_scaled, y_clean)
    y_pred_baseline = model.predict(X_scaled)

    if effect_metric == "rho":
        baseline_perf = spearmanr(y_clean, y_pred_baseline)[0]
    elif effect_metric == "r2":
        baseline_perf = r2_score(y_clean, y_pred_baseline)
    elif effect_metric == "mae":
        baseline_perf = mean_absolute_error(y_clean, y_pred_baseline)
    else:
        raise ValueError("effect_metric must be 'rho', 'r2', or 'mae'.")

    print(f"  Baseline performance (all networks): {effect_metric}={baseline_perf:.4f}")

    results = []
    for net_idx, net_name in enumerate(network_names):
        permuted_perfs = []
        
        for perm_idx in range(n_permutations):
            # Permute this network's values
            X_perm = X_scaled.copy()
            X_perm[:, net_idx] = np.random.permutation(X_perm[:, net_idx])
            
            model_perm = Ridge(alpha=alpha)
            model_perm.fit(X_perm, y_clean)
            y_pred_perm = model_perm.predict(X_perm)

            if effect_metric == "rho":
                perm_perf = spearmanr(y_clean, y_pred_perm)[0]
            elif effect_metric == "r2":
                perm_perf = r2_score(y_clean, y_pred_perm)
            elif effect_metric == "mae":
                perm_perf = mean_absolute_error(y_clean, y_pred_perm)
            
            permuted_perfs.append(perm_perf)
        
        mean_perm_perf = np.mean(permuted_perfs)
        std_perm_perf = np.std(permuted_perfs)
        
        if effect_metric == "mae":
            effect_size = mean_perm_perf - baseline_perf  # Increase in MAE
        else:
            effect_size = baseline_perf - mean_perm_perf  # Drop in rho/R²
        
        # Apply transformations
        # Keep signed effect size for CSV, but zero out negatives for percentage
        effect_size_signed = effect_size
        effect_size_for_pct = max(0.0, effect_size)  # Only positive contributions
        
        if use_absolute:
            effect_size = abs(effect_size)
        if as_percentage and baseline_perf != 0:
            effect_size_pct = (effect_size_for_pct / abs(baseline_perf)) * 100
        else:
            effect_size_pct = np.nan

        results.append({
            "Network": net_name,
            "Baseline_Performance": float(baseline_perf),
            "Mean_Permuted_Performance": float(mean_perm_perf),
            "Permuted_Performance_Std": float(std_perm_perf),
            "Effect_Size": float(effect_size),
            "Effect_Size_Signed": float(effect_size_signed),
            "Effect_Size_Pct": float(effect_size_pct),
            "N_Subjects": int(X_clean.shape[0]),
        })

    return pd.DataFrame(results)


def compute_network_importance_loo(
    X_networks: np.ndarray,
    y: np.ndarray,
    network_names: List[str],
    effect_metric: str = "rho",
    alpha: float = 1.0,
    use_absolute: bool = True,
    as_percentage: bool = False,
) -> pd.DataFrame:
    """
    Compute network importance via leave-one-out regression.
    
    Args:
        X_networks: (N_subjects, N_networks) network feature matrix
        y: (N_subjects,) target values
        network_names: Network labels
        effect_metric: 'rho' (Spearman), 'r2', or 'mae'
        alpha: Ridge regularization parameter
    
    Returns:
        DataFrame with Network, Baseline_Performance, LOO_Performance, Effect_Size columns
    """
    # Remove NaN rows
    mask = np.all(np.isfinite(X_networks), axis=1) & np.isfinite(y)
    X_clean = X_networks[mask]
    y_clean = y[mask]

    if X_clean.shape[0] < 10:
        raise ValueError(f"Insufficient samples after removing NaNs: {X_clean.shape[0]}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # Baseline: all networks
    model = Ridge(alpha=alpha)
    model.fit(X_scaled, y_clean)
    y_pred_baseline = model.predict(X_scaled)

    if effect_metric == "rho":
        baseline_perf = spearmanr(y_clean, y_pred_baseline)[0]
    elif effect_metric == "r2":
        baseline_perf = r2_score(y_clean, y_pred_baseline)
    elif effect_metric == "mae":
        baseline_perf = mean_absolute_error(y_clean, y_pred_baseline)
    else:
        raise ValueError("effect_metric must be 'rho', 'r2', or 'mae'.")

    print(f"  Baseline performance (all networks): {effect_metric}={baseline_perf:.4f}")

    results = []
    for net_idx, net_name in enumerate(network_names):
        # Leave out this network
        mask_loo = np.ones(X_scaled.shape[1], dtype=bool)
        mask_loo[net_idx] = False
        X_loo = X_scaled[:, mask_loo]

        model_loo = Ridge(alpha=alpha)
        model_loo.fit(X_loo, y_clean)
        y_pred_loo = model_loo.predict(X_loo)

        if effect_metric == "rho":
            loo_perf = spearmanr(y_clean, y_pred_loo)[0]
            effect_size = baseline_perf - loo_perf  # Drop in correlation
        elif effect_metric == "r2":
            loo_perf = r2_score(y_clean, y_pred_loo)
            effect_size = baseline_perf - loo_perf  # Drop in R²
        elif effect_metric == "mae":
            loo_perf = mean_absolute_error(y_clean, y_pred_loo)
            effect_size = loo_perf - baseline_perf  # Increase in MAE (positive = worse)

        # Apply transformations
        if use_absolute:
            effect_size = abs(effect_size)
        if as_percentage and baseline_perf != 0:
            effect_size_pct = (effect_size / abs(baseline_perf)) * 100
        else:
            effect_size_pct = np.nan

        results.append({
            "Network": net_name,
            "Baseline_Performance": float(baseline_perf),
            "LOO_Performance": float(loo_perf),
            "Effect_Size": float(effect_size),
            "Effect_Size_Pct": float(effect_size_pct),
            "N_Subjects": int(X_clean.shape[0]),
        })

    return pd.DataFrame(results)


def load_presets(preset_names: Sequence[str]) -> OrderedDict[str, Dict[str, Any]]:
    """Load preset configurations from YAML."""
    presets: OrderedDict[str, Dict[str, Any]] = OrderedDict()
    if not preset_names:
        return presets

    config_path = PROJECT_ROOT / "config" / "network_importance_presets.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Preset configuration file not found at {config_path}.")

    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    available = raw.get("presets", {})
    if not available:
        raise ValueError(f"No presets defined in {config_path}.")

    for preset_name in preset_names:
        if preset_name not in available:
            raise KeyError(f"Preset '{preset_name}' not found in {config_path}.")
        preset_block = available[preset_name] or {}
        datasets_cfg = preset_block.get("datasets", {})
        for dataset_key, dataset_cfg in datasets_cfg.items():
            if dataset_key in presets:
                warnings.warn(f"Dataset '{dataset_key}' specified multiple times; using '{preset_name}'.")
            presets[dataset_key] = {
                "ig_dir": dataset_cfg.get("ig_dir"),
                "age_source": dataset_cfg.get("age_source"),
                "parcellation": preset_block.get("parcellation", "yeo7"),
                "aggregation_method": preset_block.get("aggregation_method", "abs_mean"),
                "effect_metric": preset_block.get("effect_metric", "rho"),
                "ridge_alpha": preset_block.get("ridge_alpha", 1.0),
            }

    return presets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute network importance via multivariate regression and LOO analysis."
    )

    parser.add_argument(
        "--preset",
        nargs="+",
        default=[],
        help="Load dataset configuration presets from config/network_importance_presets.yaml.",
    )
    parser.add_argument(
        "--parcellation",
        choices=["yeo7", "yeo17"],
        default=None,
        help="Yeo parcellation (overrides preset if provided).",
    )
    parser.add_argument(
        "--aggregation-method",
        choices=["mean", "abs_mean", "pos_share", "neg_share", "signed_share"],
        default=None,
        help="IG aggregation method (overrides preset if provided).",
    )
    parser.add_argument(
        "--effect-metric",
        choices=["rho", "r2", "mae"],
        default=None,
        help="Metric for effect size: rho=Spearman, r2=R², mae=MAE (overrides preset).",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=None,
        help="Ridge regularization parameter (overrides preset).",
    )
    parser.add_argument(
        "--importance-method",
        choices=["dominance", "coefficients", "permutation", "loo"],
        default=None,
        help="Importance computation method (overrides preset if provided).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("network_importance"),
        help="Directory where summary CSV files will be written.",
    )
    parser.add_argument(
        "--atlas-path",
        type=Path,
        default=DEFAULT_YEO_ATLAS,
        help="Path to the Yeo atlas CSV file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file progress information.",
    )
    parser.add_argument(
        "--save-subject-level",
        action="store_true",
        help="Save per-subject network IG matrices with ages for each dataset.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    preset_configs = load_presets(args.preset)

    if not preset_configs:
        raise ValueError("No datasets specified. Provide --preset.")

    all_results: List[pd.DataFrame] = []

    for dataset, cfg in preset_configs.items():
        ig_dir_str = cfg.get("ig_dir")
        age_source_str = cfg.get("age_source")

        if not ig_dir_str or not age_source_str:
            print(f"✗ Skipping {dataset}: missing ig_dir or age_source in preset.")
            continue

        ig_dir = Path(ig_dir_str).expanduser()
        
        # Parse age_source (handle optional ::value_key suffix)
        age_source_parts = str(age_source_str).split("::")
        age_source = Path(age_source_parts[0]).expanduser()
        
        if not ig_dir.exists():
            print(f"✗ Skipping {dataset}: IG directory not found: {ig_dir}")
            continue

        if not age_source.exists():
            print(f"✗ Skipping {dataset}: age source not found: {age_source}")
            continue

        parcellation = args.parcellation or cfg.get("parcellation", "yeo7")
        aggregation_method = args.aggregation_method or cfg.get("aggregation_method", "abs_mean")
        effect_metric = args.effect_metric or cfg.get("effect_metric", "rho")
        ridge_alpha = args.ridge_alpha if args.ridge_alpha is not None else cfg.get("ridge_alpha", 1.0)

        ig_files = sorted(ig_dir.glob("*_ig*.npz"), key=ig_file_sort_key)

        if not ig_files:
            print(f"✗ Skipping {dataset}: no *_ig*.npz files in {ig_dir}")
            continue

        print(f"\n{'='*80}\nDATASET: {dataset}")
        print(f"IG directory: {ig_dir}")
        print(f"Found {len(ig_files)} IG files")
        print(f"Parcellation: {parcellation}, Aggregation: {aggregation_method}, Metric: {effect_metric}")
        print(f"{'='*80}")

        # Load network mapping
        network_map = load_network_mapping(args.atlas_path, parcellation)

        # Load ages (handle optional ::value_key suffix in age_source string)
        age_source_str_parts = str(age_source).split("::")
        age_source_path = Path(age_source_str_parts[0])
        age_value_key = age_source_str_parts[1] if len(age_source_str_parts) == 2 else None
        
        if not age_source_path.exists():
            print(f"✗ Skipping {dataset}: age source not found: {age_source_path}")
            continue
            
        _, ages = load_age_source(age_source_path, value_key=age_value_key)
        print(f"  Loaded {len(ages)} ages from {age_source_path.name}")

        # Aggregate IG across folds
        print(f"  Aggregating {len(ig_files)} folds...")
        network_names, network_matrix = aggregate_network_ig_across_folds(
            ig_files,
            network_map,
            aggregation_method,
            parcellation,
            verbose=args.verbose,
        )
        print(f"  Aggregated: {network_matrix.shape[0]} subjects × {len(network_names)} networks")

        if network_matrix.shape[0] != len(ages):
            print(
                f"  ⚠︎ Subject count mismatch: IG has {network_matrix.shape[0]} subjects, ages has {len(ages)}."
            )
            # Truncate to smaller length
            min_n = min(network_matrix.shape[0], len(ages))
            network_matrix = network_matrix[:min_n]
            ages = ages[:min_n]
            print(f"  Truncated to {min_n} subjects for alignment.")

        # Compute network importance
        importance_method = args.importance_method or cfg.get("importance_method", "coefficients")
        
        if importance_method == "dominance":
            print(f"  Computing network importance via dominance analysis...")
            importance_df = compute_network_importance_dominance(
                network_matrix,
                ages,
                network_names,
                n_permutations=5000,
            )
        elif importance_method == "coefficients":
            print(f"  Computing network importance via standardized Ridge coefficients...")
            importance_df = compute_network_importance_coefficients(
                network_matrix,
                ages,
                network_names,
                alpha=ridge_alpha,
                as_percentage=True,
            )
        elif importance_method == "permutation":
            print(f"  Computing network importance via permutation (20 shuffles/network)...")
            importance_df = compute_network_importance_permutation(
                network_matrix,
                ages,
                network_names,
                effect_metric=effect_metric,
                alpha=ridge_alpha,
                n_permutations=20,
                use_absolute=True,
                as_percentage=True,
            )
        else:
            print(f"  Computing network importance via LOO regression...")
            importance_df = compute_network_importance_loo(
                network_matrix,
                ages,
                network_names,
                effect_metric=effect_metric,
                alpha=ridge_alpha,
                use_absolute=True,
                as_percentage=True,
            )

        importance_df.insert(0, "Dataset", dataset)
        importance_df.insert(1, "Parcellation", parcellation)
        importance_df.insert(2, "Aggregation", aggregation_method)
        importance_df.insert(3, "Effect_Metric", effect_metric)

        all_results.append(importance_df)

        dataset_csv = output_dir / f"{dataset}_network_importance.csv"
        importance_df.to_csv(dataset_csv, index=False)
        print(f"  ✓ Saved: {dataset_csv}")
        
        # Also save in radar-compatible format (Network, Dominance %)
        if "Dominance_Pct" in importance_df.columns:
            radar_df = pd.DataFrame({
                "Network": importance_df["Network"],
                "Dominance (%)": importance_df["Dominance_Pct"] / 100,  # Convert back to fraction for compatibility
            })
            radar_csv = output_dir / f"dominance_multivariate_network_age_{dataset}.csv"
            radar_df.to_csv(radar_csv, index=False)
            print(f"  ✓ Saved radar-ready CSV: {radar_csv}")
        
        if args.save_subject_level:
            # Save subject-level matrix: Age + Network_0, Network_1, ..., Network_20
            subject_df = pd.DataFrame(network_matrix, columns=network_names)
            subject_df.insert(0, "Age", ages[:network_matrix.shape[0]])
            subject_df.insert(0, "Subject_Index", range(network_matrix.shape[0]))
            
            subject_csv = output_dir / f"{dataset}_subject_network_values.csv"
            subject_df.to_csv(subject_csv, index=False)
            print(f"  ✓ Saved subject-level data: {subject_csv}")

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined_path = output_dir / f"network_importance_combined.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\n✓ Combined summary saved: {combined_path}")
        
        # Create permutation p-value summary if dominance was used
        if "P_Value" in combined.columns:
            pval_summary = combined[["Dataset", "Model_R2_Adj", "P_Value", "N_Subjects"]].drop_duplicates()
            pval_summary_path = output_dir / "dominance_permutation_pvalues.csv"
            pval_summary.to_csv(pval_summary_path, index=False)
            print(f"✓ Permutation p-value summary saved: {pval_summary_path}")
            print("\nPermutation Test Results:")
            print(pval_summary.to_string(index=False))
            
            # Create pooled dominance for radar overlap plots
            if "Dominance_Pct" in combined.columns:
                pooled = combined.groupby("Network")["Dominance_Pct"].mean().reset_index()
                pooled_radar = pd.DataFrame({
                    "Network": pooled["Network"],
                    "Dominance (%)": pooled["Dominance_Pct"] / 100,
                })
                pooled_csv = output_dir / "dominance_multivariate_network_age_pooled.csv"
                pooled_radar.to_csv(pooled_csv, index=False)
                print(f"✓ Pooled dominance (averaged across cohorts) saved: {pooled_csv}")
    else:
        print("\n✗ No datasets processed successfully.")


if __name__ == "__main__":
    main()

