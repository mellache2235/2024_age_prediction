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
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import yaml

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

            net_matrix, network_names = aggregate_rois_to_networks(
                roi_matrix, network_map, method=aggregation_method
            )

            if net_matrix is None or network_names is None:
                raise RuntimeError("Failed to aggregate ROIs to networks.")

            # Strip Network_ prefix
            network_names = [name.replace("Network_", "") for name in network_names]

            if network_names_reference is None:
                network_names_reference = network_names
            elif network_names != network_names_reference:
                raise ValueError(f"Network name mismatch in {ig_path.name}.")

            # Store per subject (row index as subject key since IDs not in NPZ)
            for subj_idx in range(n_subjects):
                for net_idx, net_name in enumerate(network_names):
                    value = net_matrix[subj_idx, net_idx]
                    if np.isfinite(value):
                        subject_network_store[subj_idx][net_name].append(float(value))

    if not subject_network_store:
        raise RuntimeError("No network data aggregated across folds.")

    network_names_final = network_names_reference or []
    n_subjects = max(subject_network_store.keys()) + 1

    # Average across folds
    network_matrix = np.zeros((n_subjects, len(network_names_final)))
    for subj_idx in range(n_subjects):
        for net_idx, net_name in enumerate(network_names_final):
            values = subject_network_store[subj_idx].get(net_name, [])
            network_matrix[subj_idx, net_idx] = np.mean(values) if values else np.nan

    return network_names_final, network_matrix


def compute_network_importance_loo(
    X_networks: np.ndarray,
    y: np.ndarray,
    network_names: List[str],
    effect_metric: str = "rho",
    alpha: float = 1.0,
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

        results.append({
            "Network": net_name,
            "Baseline_Performance": float(baseline_perf),
            "LOO_Performance": float(loo_perf),
            "Effect_Size": float(effect_size),
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
        age_source = Path(age_source_str).expanduser()

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

        # Load ages
        _, ages = load_age_source(age_source)
        print(f"  Loaded {len(ages)} ages from {age_source.name}")

        # Aggregate IG across folds
        print(f"  Aggregating {len(ig_files)} folds...")
        network_names, network_matrix = aggregate_network_ig_across_folds(
            ig_files,
            network_map,
            aggregation_method,
            verbose=args.verbose,
        )
        print(f"  Aggregated: {network_matrix.shape[0]} subjects × {len(network_names)} networks")

        if network_matrix.shape[0] != len(ages):
            raise ValueError(
                f"Subject count mismatch: IG has {network_matrix.shape[0]} subjects, ages has {len(ages)}."
            )

        # Compute network importance
        print(f"  Computing network importance via LOO regression...")
        importance_df = compute_network_importance_loo(
            network_matrix,
            ages,
            network_names,
            effect_metric=effect_metric,
            alpha=ridge_alpha,
        )

        importance_df.insert(0, "Dataset", dataset)
        importance_df.insert(1, "Parcellation", parcellation)
        importance_df.insert(2, "Aggregation", aggregation_method)
        importance_df.insert(3, "Effect_Metric", effect_metric)

        all_results.append(importance_df)

        dataset_csv = output_dir / f"{dataset}_network_importance_{effect_metric}.csv"
        importance_df.to_csv(dataset_csv, index=False)
        print(f"  ✓ Saved: {dataset_csv}")

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined_path = output_dir / f"network_importance_combined.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\n✓ Combined summary saved: {combined_path}")
    else:
        print("\n✗ No datasets processed successfully.")


if __name__ == "__main__":
    main()

