#!/usr/bin/env python3
"""Compute network-level IG correlations against age- or behavior-related targets.

This utility aggregates integrated gradients (IG) outputs that were saved per-fold
(`*_ig.npz` files). For each dataset it:

1. Loads every fold's IG file (default: 500 folds).
2. Collapses ROI-level IG values to Yeo networks (7- or 17-network options).
3. Averages each subject's network IG across folds (when subject IDs are
   available).
4. Correlates the resulting network-wise IG scores with one or more subject-level
   targets (chronological age by default, with optional predicted ages or
   behavioral scores provided via NPZ keys).
5. Optionally applies Benjamini–Hochberg FDR correction to the resulting p-values.

Usage example:

```
python scripts/compute_network_age_correlations.py \
    --datasets cmihbn_td nki_rs_td \
    --root-dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/figures \
    --parcellation yeo7 \
    --target-key Predicted_Brain_Age:brain_age_pred \
    --output-dir /tmp/network_age_summary
```

The script writes one summary CSV (per target per dataset) plus optional
per-subject network matrices. It attempts to infer subject IDs and chronological
ages from each NPZ file but also allows manual overrides via command-line
arguments if needed. Additional targets can be surfaced by specifying the NPZ
array keys to extract.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import warnings
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from statsmodels.stats.multitest import multipletests
except ImportError:
    multipletests = None

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
import yaml

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

sys.path.append(str(PROJECT_ROOT / "scripts"))
sys.path.append(str(PROJECT_ROOT / "utils"))

from optimized_brain_behavior_core import (  # noqa: E402
    aggregate_rois_to_networks,
)

from plot_styles import (  # noqa: E402
    create_standardized_scatter,
    get_dataset_title,
    setup_arial_font,
    FIGSIZE_SINGLE,
    FIGURE_FACECOLOR,
    DPI,
)

from data_utils import (  # noqa: E402
    load_finetune_dataset,
    load_finetune_dataset_w_ids,
)


# Default location of the Yeo atlas CSV used throughout the project
DEFAULT_YEO_ATLAS = Path(
    "/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/"
    "scripts/dnn/feature_attribution/csv_files/subregion_func_network_Yeo_updated_yz.csv"
)

FOLD_PATTERN = re.compile(r"_(\d+)[^/\\]*_ig\.npz$", re.IGNORECASE)


UPPER_TOKENS = {
    "ADHD",
    "ASD",
    "TD",
    "SRS",
    "ADOS",
    "IQ",
    "CBCL",
    "DSM",
    "CAARS",
    "T",
    "RS",
    "B",
    "A",
    "C",
    "CDI",
    "BASC",
    "SCQ",
}


def discover_ig_directory(root: Path, dataset: str, prefer_td: bool = True) -> Optional[Path]:
    """Return the directory containing IG npz files for a dataset."""

    dataset_dir = root / dataset
    if not dataset_dir.exists():
        return None

    td_dir = dataset_dir / "ig_files_td"
    main_dir = dataset_dir / "ig_files"

    if prefer_td and td_dir.exists():
        return td_dir
    if main_dir.exists():
        return main_dir
    if td_dir.exists():
        return td_dir
    return None


def extract_fold_index(path: Path) -> Optional[int]:
    """Extract fold index from filenames like *_5_*_ig.npz."""

    match = FOLD_PATTERN.search(path.name)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None

    # Fallback: last integer in stem
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


def sanitize_filename(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return sanitized or "value"


def load_age_source(path: Path) -> Tuple[Optional[List[str]], np.ndarray]:
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
            for candidate in ("ages", "age", "y", "targets", "labels"):
                if candidate in data:
                    arr = np.asarray(data[candidate], dtype=float)
                    return None, arr
        raise ValueError(
            f"Could not locate age array in NPZ file {path}. Expected keys include 'ages', 'age', 'y', 'targets', or 'labels'."
        )

    if suffix == ".npy":
        arr = np.load(path)
        return None, np.asarray(arr, dtype=float)

    if suffix in {".csv", ".tsv"}:
        df = pd.read_csv(path)
        for candidate in ("age", "Age", "AGES", "chronological_age"):
            if candidate in df.columns:
                arr = df[candidate].to_numpy(dtype=float)
                return None, arr
        raise ValueError(
            f"CSV/TSV file {path} does not contain a recognizable age column."
        )

    raise ValueError(f"Unsupported age source file format for {path}.")


def parse_age_sources(entries: Optional[Sequence[str]]) -> Dict[str, Tuple[Optional[List[str]], np.ndarray]]:
    sources: Dict[str, Tuple[Optional[List[str]], np.ndarray]] = {}
    if not entries:
        return sources

    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                f"Invalid --age-source '{entry}'. Expected format DATASET=/absolute/path/to/ages.bin"
            )
        dataset_key, raw_path = entry.split("=", 1)
        dataset_key = dataset_key.strip()
        age_path = Path(raw_path).expanduser()
        if not dataset_key:
            raise ValueError(f"Invalid --age-source '{entry}': dataset key cannot be empty.")
        if not age_path.exists():
            raise FileNotFoundError(f"Age source for dataset '{dataset_key}' not found: {age_path}")
        subjects, ages = load_age_source(age_path)
        sources[dataset_key] = (subjects, ages)

    return sources


def load_target_source(path: Path, id_key: str, value_key: str) -> Tuple[List[str], np.ndarray]:
    suffix = path.suffix.lower()

    if suffix in {".csv", ".tsv"}:
        df = pd.read_csv(path)
        if id_key not in df.columns or value_key not in df.columns:
            raise KeyError(
                f"Columns '{id_key}' and/or '{value_key}' not found in {path}."
            )
        ids = df[id_key].astype(str).tolist()
        values = df[value_key].to_numpy(dtype=float)
        return ids, values

    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            if id_key not in data or value_key not in data:
                raise KeyError(
                    f"Keys '{id_key}' and/or '{value_key}' not present in {path}."
                )
            ids = np.asarray(data[id_key]).astype(str).tolist()
            values = np.asarray(data[value_key], dtype=float)
            return ids, values

    if suffix == ".npy":
        arr = np.load(path, allow_pickle=True)
        if arr.dtype.names is None:
            raise ValueError(
                f"Target source {path} is a plain array; provide a structured array with named fields."
            )
        if id_key not in arr.dtype.names or value_key not in arr.dtype.names:
            raise KeyError(
                f"Fields '{id_key}' and/or '{value_key}' not found in structured array {path}."
            )
        ids = arr[id_key].astype(str).tolist()
        values = arr[value_key].astype(float)
        return ids, values

    raise ValueError(f"Unsupported target source file format for {path}.")


def parse_target_sources(entries: Optional[Sequence[str]]) -> Dict[str, List[Tuple[str, List[str], np.ndarray]]]:
    sources: Dict[str, List[Tuple[str, List[str], np.ndarray]]] = defaultdict(list)
    if not entries:
        return sources

    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                f"Invalid --target-source '{entry}'. Expected format LABEL=DATASET::/path::id_col::value_col"
            )

        label, remainder = entry.split("=", 1)
        label = label.strip()
        parts = remainder.split("::")
        if len(parts) != 4:
            raise ValueError(
                f"Invalid --target-source '{entry}'. Expected format LABEL=DATASET::/path::id_col::value_col"
            )

        dataset_key = parts[0].strip()
        path = Path(parts[1]).expanduser()
        id_key = parts[2].strip()
        value_key = parts[3].strip()

        if not label:
            raise ValueError(f"Invalid --target-source '{entry}': label cannot be empty.")
        if not dataset_key:
            raise ValueError(f"Invalid --target-source '{entry}': dataset key cannot be empty.")
        if not path.exists():
            raise FileNotFoundError(
                f"Target source file for dataset '{dataset_key}' not found: {path}"
            )

        subject_ids, values = load_target_source(path, id_key, value_key)
        sources[dataset_key].append((label, subject_ids, values))

    return sources


def load_presets(preset_names: Sequence[str]) -> "OrderedDict[str, Dict[str, Any]]":
    presets: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    if not preset_names:
        return presets

    config_path = PROJECT_ROOT / "config" / "network_correlation_presets.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Preset configuration file not found at {config_path}."
        )

    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    available = raw.get("presets", {})
    if not available:
        raise ValueError(
            f"No presets defined in {config_path}."
        )

    for preset_name in preset_names:
        if preset_name not in available:
            raise KeyError(
                f"Preset '{preset_name}' not found in {config_path}."
            )
        preset_block = available[preset_name] or {}
        datasets_cfg = preset_block.get("datasets", {})
        for dataset_key, dataset_cfg in datasets_cfg.items():
            if dataset_key in presets:
                warnings.warn(
                    f"Dataset '{dataset_key}' specified multiple times across presets; using configuration from '{preset_name}'."
                )
            presets[dataset_key] = dataset_cfg or {}

    return presets


def format_target_axis_label(label: str, chronological_label: str) -> str:
    if not label:
        return "Target"

    label_clean = label.strip()

    if label_clean == chronological_label:
        return "Chronological Age (years)"

    normalized = label_clean.replace("_", " ").replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized)

    lower_norm = normalized.lower()
    if lower_norm.endswith(" observed"):
        normalized = normalized[:-8].rstrip() + " (Observed)"
    elif lower_norm.endswith(" predicted"):
        normalized = normalized[:-9].rstrip() + " (Predicted)"

    words: List[str] = []
    for word in normalized.split():
        upper_word = word.upper()
        if upper_word in UPPER_TOKENS:
            words.append(upper_word)
        else:
            words.append(word.capitalize())

    formatted = " ".join(words)
    lower_formatted = formatted.lower()
    if "age" in lower_formatted and "(years)" not in lower_formatted:
        formatted += " (years)"
    return formatted


def format_network_axis_label(network_name: str) -> str:
    return f"{network_name} Mean IG"


def create_scatter_figure(
    dataset_key: str,
    target_label: str,
    network_name: str,
    x_values: np.ndarray,
    y_values: np.ndarray,
    rho: float,
    p_value: float,
    output_dir: Path,
    chronological_label: str,
    title_suffix: str = "",
) -> None:
    if x_values.size < 3:
        return

    p_str = "< 0.001" if p_value < 0.001 else f"= {p_value:.3f}"
    stats_text = f"ρ = {rho:.3f}\np {p_str}"

    dataset_title = get_dataset_title(dataset_key)
    title = dataset_title if not title_suffix else f"{dataset_title} – {title_suffix}"

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE, dpi=DPI)
    fig.patch.set_facecolor(FIGURE_FACECOLOR)

    create_standardized_scatter(
        ax,
        x_values,
        y_values,
        title=title,
        xlabel=format_target_axis_label(target_label, chronological_label),
        ylabel=format_network_axis_label(network_name),
        stats_text=stats_text,
        is_subplot=False,
    )

    plt.tight_layout()

    base_name = "_".join(
        [
            sanitize_filename(dataset_key),
            sanitize_filename(target_label),
            sanitize_filename(network_name),
            "network_scatter",
        ]
    )
    png_path = output_dir / f"{base_name}.png"
    tiff_path = output_dir / f"{base_name}.tiff"
    ai_path = output_dir / f"{base_name}.ai"

    fig.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight",
        facecolor=FIGURE_FACECOLOR,
        edgecolor="none",
    )
    fig.savefig(
        tiff_path,
        dpi=DPI,
        bbox_inches="tight",
        facecolor=FIGURE_FACECOLOR,
        edgecolor="none",
        format="tiff",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    pdf_backend.FigureCanvas(fig).print_pdf(str(ai_path))

    plt.close(fig)

    print(f"    ✓ Scatter saved: {png_path.name}")


def load_network_mapping(atlas_path: Path, parcellation: str) -> Dict[int, str]:
    """Load ROI→network mapping for the requested parcellation."""

    parcellation = parcellation.lower()
    if not atlas_path.exists():
        raise FileNotFoundError(
            f"Yeo atlas CSV not found at {atlas_path}. Provide --atlas-path explicitly."
        )

    atlas_df = pd.read_csv(atlas_path)

    if parcellation not in {"yeo7", "yeo-7", "yeo17", "yeo-17"}:
        raise ValueError("Parcellation must be 'yeo7' or 'yeo17'.")

    target_key = "7" if "7" in parcellation else "17"

    candidate_cols = [
        col for col in atlas_df.columns if "yeo" in col.lower() and target_key in col.lower()
    ]
    if not candidate_cols:
        raise ValueError(
            f"Could not locate a Yeo-{target_key} network column in atlas {atlas_path}."
        )

    network_col = candidate_cols[0]
    network_map: Dict[int, str] = {}

    for idx in range(min(len(atlas_df), 246)):
        value = atlas_df.iloc[idx][network_col]
        if pd.isna(value):
            continue
        network_map[idx] = str(value)

    if not network_map:
        raise ValueError(
            f"No network assignments found in column '{network_col}' of {atlas_path}."
        )

    return network_map


def ensure_network_names(network_names: Sequence[str]) -> List[str]:
    """Normalize network labels returned by aggregate_rois_to_networks."""

    normalized = []
    for name in network_names:
        if not isinstance(name, str):
            name = str(name)
        if name.startswith("Network_"):
            name = name[len("Network_") :]
        normalized.append(name)
    return normalized


def collapse_time_dimension(attr_data: np.ndarray) -> np.ndarray:
    """Collapse time dimension of IG tensor to ROI-level values."""

    if attr_data.ndim == 3:
        return np.median(attr_data, axis=2)
    if attr_data.ndim == 2:
        return attr_data
    raise ValueError(
        f"Unexpected IG array shape {attr_data.shape}; expected 2D or 3D (subjects×ROIs×time)."
    )


def flatten_subject_array(arr: np.ndarray, n_subjects: int) -> Optional[np.ndarray]:
    """Return a 1D array of length `n_subjects` when possible."""

    if arr is None:
        return None

    if arr.shape == (n_subjects,):
        return arr

    if arr.ndim == 2 and 1 in arr.shape:
        other_dim = arr.shape[0] if arr.shape[1] == 1 else arr.shape[1]
        if other_dim == n_subjects:
            return arr.reshape(n_subjects)

    return None


def infer_subjects_and_ages(
    npz: np.lib.npyio.NpzFile,
    n_subjects: int,
    age_key: Optional[str] = None,
    subject_key: Optional[str] = None,
    require_age: bool = True,
) -> Tuple[np.ndarray, np.ndarray, bool, bool]:
    """Infer subject IDs and chronological ages from an NPZ bundle."""

    detected_subjects: Optional[np.ndarray] = None
    subjects_from_file = False

    if subject_key:
        if subject_key not in npz:
            raise KeyError(
                f"Requested subject key '{subject_key}' not found in {npz.files}."
            )
        subject_arr = flatten_subject_array(npz[subject_key], n_subjects)
        if subject_arr is None:
            raise ValueError(
                f"Subject key '{subject_key}' does not have shape {n_subjects}."
            )
        detected_subjects = subject_arr.astype(str)
        subjects_from_file = True

    detected_ages: Optional[np.ndarray] = None
    ages_from_file = False

    if age_key:
        if age_key not in npz:
            raise KeyError(f"Requested age key '{age_key}' not present in NPZ file.")
        age_arr = flatten_subject_array(npz[age_key], n_subjects)
        if age_arr is None:
            raise ValueError(
                f"Age key '{age_key}' does not align with {n_subjects} subjects."
            )
        detected_ages = age_arr.astype(float)
        ages_from_file = True

    numeric_candidates: List[Tuple[str, np.ndarray]] = []

    for key in npz.files:
        if key == "arr_0":
            continue

        arr = flatten_subject_array(npz[key], n_subjects)
        if arr is None:
            continue

        key_lower = key.lower()

        if detected_subjects is None and arr.dtype.kind in {"U", "S", "O"}:
            detected_subjects = arr.astype(str)
            subjects_from_file = True
            continue

        if arr.dtype.kind in {"i", "f"}:
            arr = arr.astype(float)
            if "pred" in key_lower or "prob" in key_lower or "logit" in key_lower:
                continue

            numeric_candidates.append((key, arr))

            if detected_ages is None and any(
                token in key_lower for token in ("age", "label", "target", "y")
            ):
                detected_ages = arr
                ages_from_file = True

    if detected_subjects is None:
        detected_subjects = np.array([f"subject_{idx:05d}" for idx in range(n_subjects)])
        subjects_from_file = False

    if detected_ages is None:
        plausible: List[Tuple[str, np.ndarray]] = []
        for key, arr in numeric_candidates:
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                continue
            min_val, max_val = float(np.min(finite)), float(np.max(finite))
            if 0 <= min_val <= 120 and 0 < max_val <= 120:
                plausible.append((key, arr))

        if plausible:
            detected_ages = plausible[0][1]
            ages_from_file = True
        elif numeric_candidates:
            detected_ages = numeric_candidates[0][1]
            ages_from_file = True

    if detected_ages is None:
        if require_age:
            raise ValueError(
                "Could not infer chronological ages from NPZ. Supply --age-key explicitly."
            )
        detected_ages = np.full(n_subjects, np.nan, dtype=float)
        ages_from_file = False

    return detected_subjects, detected_ages, subjects_from_file, ages_from_file


def aggregate_subject_networks(
    ig_files: Sequence[Path],
    network_map: Dict[int, str],
    aggregation_method: str,
    age_key: Optional[str],
    subject_key: Optional[str],
    target_key_map: Optional[Dict[str, str]] = None,
    collect_chronological: bool = True,
    verbose: bool = False,
    override_subjects: Optional[Sequence[str]] = None,
    override_ages: Optional[np.ndarray] = None,
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Aggregate ROI-level IG to networks and align by subject.

    Returns
    -------
    network_names
        Ordered list of network labels.
    ordered_subjects
        Subject IDs aligned with the rows of the returned matrices.
    ages
        Chronological ages (NaN when unavailable or collect_chronological=False).
    network_matrix
        Averaged network-level IG values for each subject.
    target_arrays
        Dictionary mapping target labels to subject-aligned arrays (averaged across folds).
    """

    subject_store: Dict[str, Dict[str, List[float]]] = {}
    subject_ages: Dict[str, float] = {}
    network_names_reference: Optional[List[str]] = None
    target_key_map = target_key_map or {}
    target_store: Dict[str, Dict[str, List[float]]] = {
        label: defaultdict(list) for label in target_key_map
    }
    missing_target_reported: Dict[str, bool] = {label: False for label in target_key_map}

    override_subjects_array: Optional[np.ndarray] = (
        np.asarray(override_subjects, dtype=str) if override_subjects is not None else None
    )
    override_ages_array: Optional[np.ndarray] = (
        np.asarray(override_ages, dtype=float) if override_ages is not None else None
    )

    for ig_path in ig_files:
        fold_idx = extract_fold_index(ig_path)
        if verbose:
            if fold_idx is not None:
                print(f"  Processing fold {fold_idx:03d}: {ig_path.name}")
            else:
                print(f"  Processing {ig_path.name}")

        with np.load(ig_path, allow_pickle=True) as data:
            if "arr_0" not in data:
                raise KeyError(f"File {ig_path} does not contain 'arr_0' IG data.")

            attr_data = data["arr_0"]
            roi_matrix = collapse_time_dimension(attr_data)
            n_subjects, n_features = roi_matrix.shape

            if n_features < 246:
                warnings.warn(
                    f"{ig_path.name}: expected 246 ROIs, found {n_features}. Continuing with available features."
                )

            subjects, ages, subj_from_file, age_from_file = infer_subjects_and_ages(
                data,
                n_subjects,
                age_key=age_key,
                subject_key=subject_key,
                require_age=collect_chronological,
            )

            if override_subjects_array is not None:
                if override_subjects_array.shape[0] != n_subjects:
                    raise ValueError(
                        f"Override subjects length ({override_subjects_array.shape[0]}) does not match IG subjects ({n_subjects}) in {ig_path.name}."
                    )
                subjects = override_subjects_array
                subj_from_file = True

            if override_ages_array is not None:
                if override_ages_array.shape[0] != n_subjects:
                    raise ValueError(
                        f"Override ages length ({override_ages_array.shape[0]}) does not match IG subjects ({n_subjects}) in {ig_path.name}."
                    )
                ages = override_ages_array
                age_from_file = True

            if not subj_from_file and verbose:
                print(
                    f"    ⚠︎ Subject IDs not found in {ig_path.name}; using temporary placeholders."
                )

            if collect_chronological and not age_from_file and verbose:
                print(
                    f"    ⚠︎ Ages inferred heuristically for {ig_path.name}. Consider --age-key."
                )

            file_targets: Dict[str, np.ndarray] = {}

            for label, key in target_key_map.items():
                if key not in data:
                    if not missing_target_reported[label]:
                        warnings.warn(
                            f"Target key '{key}' (label '{label}') not present in {ig_path.name}."
                        )
                        missing_target_reported[label] = True
                    continue

                arr = flatten_subject_array(data[key], n_subjects)
                if arr is None:
                    raise ValueError(
                        f"Target key '{key}' (label '{label}') from {ig_path.name} does not align with {n_subjects} subjects."
                    )

                file_targets[label] = arr.astype(float)

            net_matrix, network_names = aggregate_rois_to_networks(
                roi_matrix, network_map, method=aggregation_method
            )

            if net_matrix is None or network_names is None:
                raise RuntimeError("Failed to aggregate ROIs to networks; check network map.")

            network_names = ensure_network_names(network_names)

            if network_names_reference is None:
                network_names_reference = network_names
            elif network_names != network_names_reference:
                raise ValueError(
                    f"Network name mismatch encountered in {ig_path.name}."
                )

            for subj_idx, (subj, age_value, net_row) in enumerate(
                zip(subjects, ages, net_matrix)
            ):
                subj_key = str(subj)
                age_float = float(age_value)

                if subj_key not in subject_store:
                    subject_store[subj_key] = defaultdict(list)
                    subject_ages[subj_key] = age_float
                else:
                    existing_age = subject_ages.get(subj_key, np.nan)
                    if collect_chronological:
                        if np.isfinite(existing_age) and np.isfinite(age_float):
                            if not math.isclose(
                                existing_age,
                                age_float,
                                rel_tol=1e-3,
                                abs_tol=1e-3,
                            ):
                                warnings.warn(
                                    f"Subject {subj_key} age mismatch across folds: "
                                    f"{existing_age:.3f} vs {age_float:.3f}. Using earliest value."
                                )
                        elif np.isfinite(age_float) and not np.isfinite(existing_age):
                            subject_ages[subj_key] = age_float
                    else:
                        if not np.isfinite(existing_age) and np.isfinite(age_float):
                            subject_ages[subj_key] = age_float

                for net_name, value in zip(network_names, net_row):
                    if np.isfinite(value):
                        subject_store[subj_key][net_name].append(float(value))

                for label, arr in file_targets.items():
                    value = float(arr[subj_idx])
                    if np.isfinite(value):
                        target_store[label][subj_key].append(value)

    if not subject_store:
        raise RuntimeError("No subject data aggregated – check IG directories.")

    network_names_final = network_names_reference or []
    ordered_subjects = list(subject_store.keys())

    ages = np.array([subject_ages.get(subj, np.nan) for subj in ordered_subjects], dtype=float)
    network_matrix = np.zeros((len(ordered_subjects), len(network_names_final)))

    for i, subj in enumerate(ordered_subjects):
        values_dict = subject_store[subj]
        for j, net_name in enumerate(network_names_final):
            values = values_dict.get(net_name, [])
            network_matrix[i, j] = np.mean(values) if values else np.nan

    target_arrays: Dict[str, np.ndarray] = {}
    for label, subj_values in target_store.items():
        arr: List[float] = []
        for subj in ordered_subjects:
            values = subj_values.get(subj)
            arr.append(float(np.mean(values)) if values else np.nan)
        target_arrays[label] = np.array(arr, dtype=float)

    return network_names_final, ordered_subjects, ages, network_matrix, target_arrays


def compute_correlations(
    network_names: Sequence[str],
    target_values: np.ndarray,
    network_matrix: np.ndarray,
    target_label: str,
    apply_fdr: bool = False,
) -> pd.DataFrame:
    """Compute Pearson and Spearman correlations for each network and target."""

    results: List[Dict[str, float]] = []

    for idx, net_name in enumerate(network_names):
        values = network_matrix[:, idx]
        mask = np.isfinite(values) & np.isfinite(target_values)
        n = int(mask.sum())

        if n < 3:
            continue

        target_subset = target_values[mask]
        value_subset = values[mask]

        pearson_r, pearson_p = pearsonr(target_subset, value_subset)
        spearman_r, spearman_p = spearmanr(target_subset, value_subset)

        results.append(
            {
                "Network": net_name,
                "N_Subjects": n,
                "Pearson_r": float(pearson_r),
                "Pearson_p": float(pearson_p),
                "Spearman_rho": float(spearman_r),
                "Spearman_p": float(spearman_p),
                "Mean_IG": float(np.nanmean(value_subset)),
                "Std_IG": float(np.nanstd(value_subset)),
            },
        )

    df = pd.DataFrame(results)

    if df.empty:
        return df

    df.insert(0, "Target", target_label)

    if apply_fdr:
        if multipletests is None:
            warnings.warn(
                "statsmodels is not installed; skipping FDR correction despite --apply-fdr."
            )
        else:
            for col in ("Pearson_p", "Spearman_p"):
                pvals = df[col].to_numpy()
                finite_mask = np.isfinite(pvals)
                if finite_mask.sum() == 0:
                    continue
                _, corrected, _, _ = multipletests(pvals[finite_mask], method="fdr_bh")
                df.loc[finite_mask, f"{col}_FDR"] = corrected

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate network-level IG values and correlate with age (TD)."
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[],
        help="Dataset names under the root results path (optional when using --preset).",
    )
    parser.add_argument(
        "--preset",
        nargs="+",
        default=[],
        help="Load dataset configuration presets defined in config/network_correlation_presets.yaml.",
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path(
            "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/figures"
        ),
        help="Root directory containing dataset sub-folders with IG files.",
    )
    parser.add_argument(
        "--parcellation",
        choices=["yeo7", "yeo17"],
        default="yeo7",
        help="Yeo parcellation to use when aggregating ROIs to networks.",
    )
    parser.add_argument(
        "--aggregation-method",
        choices=["mean", "abs_mean", "pos_share", "neg_share", "signed_share"],
        default="mean",
        help="Aggregation method passed to aggregate_rois_to_networks.",
    )
    parser.add_argument(
        "--target-key",
        action="append",
        default=[],
        metavar="LABEL:KEY",
        help=(
            "Additional subject-level arrays inside each NPZ to correlate. "
            "Provide as label:key (e.g., Predicted_Brain_Age:pred_age)."
        ),
    )
    parser.add_argument(
        "--target-source",
        action="append",
        default=[],
        metavar="LABEL=DATASET::/path::id_col::value_col",
        help="External target source aligned by subject ID (e.g., observed/predicted behavior).",
    )
    parser.add_argument(
        "--atlas-path",
        type=Path,
        default=DEFAULT_YEO_ATLAS,
        help="Path to the Yeo atlas CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("network_correlations"),
        help="Directory where summary CSV files will be written.",
    )
    parser.add_argument(
        "--dataset-path",
        action="append",
        metavar="DATASET=PATH",
        help=(
            "Override automatic directory discovery for a dataset. "
            "Provide the dataset label used with --datasets followed by an absolute path to the "
            "directory containing *_ig.npz files (e.g., nki_rs_td=/oak/.../integrated_gradients/nki_rs_td)."
        ),
    )
    parser.add_argument(
        "--age-source",
        action="append",
        metavar="DATASET=PATH",
        help=(
            "Provide an explicit age source for a dataset (e.g., fold_0.bin). "
            "The loader reads chronological ages from the file and aligns them with IG rows."
        ),
    )
    parser.add_argument(
        "--age-key",
        type=str,
        default=None,
        help="Optional explicit array key for chronological ages inside each NPZ.",
    )
    parser.add_argument(
        "--subject-key",
        type=str,
        default=None,
        help="Optional explicit array key for subject IDs inside each NPZ.",
    )
    parser.add_argument(
        "--skip-chronological",
        action="store_true",
        help="Disable chronological age correlations (useful for behavior-only runs).",
    )
    parser.add_argument(
        "--chronological-label",
        type=str,
        default="Chronological_Age",
        help="Label to use for chronological age correlations in the output tables.",
    )
    parser.add_argument(
        "--save-subject-level",
        action="store_true",
        help="If set, save per-subject network IG matrices for each dataset.",
    )
    parser.set_defaults(prefer_td=True)
    parser.add_argument(
        "--no-prefer-td",
        dest="prefer_td",
        action="store_false",
        help="Use 'ig_files' even when a TD directory exists (default: use TD directory).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file progress information.",
    )
    parser.add_argument(
        "--apply-fdr",
        action="store_true",
        help="Apply Benjamini–Hochberg FDR correction to p-values per dataset/target.",
    )
    parser.add_argument(
        "--scatter-plots",
        action="store_true",
        help="Generate standardized scatter plots for significant network correlations.",
    )
    parser.add_argument(
        "--scatter-alpha",
        type=float,
        default=0.05,
        help="P-value threshold for scatter plot generation (default: 0.05).",
    )
    parser.add_argument(
        "--scatter-min-n",
        type=int,
        default=20,
        help="Minimum subjects required to generate a scatter plot (default: 20).",
    )
    parser.add_argument(
        "--scatter-min-abs-rho",
        type=float,
        default=0.0,
        help="Minimum absolute Spearman ρ required for scatter plots (default: 0).",
    )
    parser.add_argument(
        "--scatter-output-dir",
        type=Path,
        default=None,
        help="Optional root directory for scatter plots. Defaults to <output-dir>/plots.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    scatter_root: Optional[Path] = None
    if args.scatter_plots:
        setup_arial_font()
        scatter_root = args.scatter_output_dir or (output_dir / "plots")
        scatter_root.mkdir(parents=True, exist_ok=True)

    network_map = load_network_mapping(args.atlas_path, args.parcellation)

    dataset_path_overrides: Dict[str, Path] = {}
    if args.dataset_path:
        for entry in args.dataset_path:
            if "=" not in entry:
                raise ValueError(
                    f"Invalid --dataset-path '{entry}'. Expected format DATASET=/absolute/path."
                )
            dataset_key, raw_path = entry.split("=", 1)
            dataset_key = dataset_key.strip()
            path = Path(raw_path).expanduser()
            if not dataset_key:
                raise ValueError(
                    f"Invalid --dataset-path '{entry}': dataset key cannot be empty."
                )
            if not path.exists():
                raise FileNotFoundError(
                    f"Override directory for dataset '{dataset_key}' does not exist: {path}"
                )
            if not path.is_dir():
                raise NotADirectoryError(
                    f"Override path for dataset '{dataset_key}' is not a directory: {path}"
                )
            dataset_path_overrides[dataset_key] = path

    preset_configs = load_presets(args.preset)

    target_key_map: Dict[str, str] = {}
    for entry in args.target_key:
        if ":" not in entry:
            raise ValueError(
                f"Invalid --target-key '{entry}'. Expected format LABEL:KEY (e.g., Predicted:pred_age)."
            )
        label, key = entry.split(":", 1)
        label = label.strip()
        key = key.strip()
        if not label or not key:
            raise ValueError(
                f"Invalid --target-key '{entry}'. Both label and key must be non-empty."
            )
        target_key_map[label] = key

    collect_chronological = not args.skip_chronological

    if not collect_chronological and not target_key_map:
        raise ValueError(
            "At least one target is required. Enable chronological correlations or provide --target-key."
        )

    age_sources = parse_age_sources(args.age_source)
    target_sources_map = parse_target_sources(args.target_source)

    for dataset, cfg in preset_configs.items():
        ig_dir_str = cfg.get("ig_dir")
        if ig_dir_str and dataset not in dataset_path_overrides:
            ig_path = Path(ig_dir_str).expanduser()
            dataset_path_overrides[dataset] = ig_path
        age_source_path = cfg.get("age_source")
        if age_source_path and dataset not in age_sources:
            age_sources[dataset] = load_age_source(Path(age_source_path).expanduser())
        for ts_cfg in cfg.get("target_sources", []) or []:
            label = ts_cfg.get("label")
            file_path = ts_cfg.get("file")
            id_column = ts_cfg.get("id_column", "subject_id")
            value_column = ts_cfg.get("value_column")
            if not label or not file_path or not value_column:
                raise ValueError(
                    f"Incomplete target source configuration for dataset '{dataset}' in preset."
                )
            ids, values = load_target_source(
                Path(file_path).expanduser(),
                id_column,
                value_column,
            )
            target_sources_map[dataset].append((label, ids, values))
        for label, key in (cfg.get("target_keys") or {}).items():
            if label in target_key_map and target_key_map[label] != key:
                raise ValueError(
                    f"Conflicting target key definitions for '{label}'."
                )
            target_key_map[label] = key

    dataset_order: List[str] = []
    dataset_order.extend(args.datasets)
    for dataset in preset_configs.keys():
        if dataset not in dataset_order:
            dataset_order.append(dataset)
    for dataset in age_sources.keys():
        if dataset not in dataset_order:
            dataset_order.append(dataset)
    for dataset in target_sources_map.keys():
        if dataset not in dataset_order:
            dataset_order.append(dataset)

    if not dataset_order:
        raise ValueError("No datasets specified. Provide --datasets or --preset.")

    args.datasets = dataset_order

    all_dataset_rows: List[pd.DataFrame] = []

    for dataset in args.datasets:
        override_dir = dataset_path_overrides.get(dataset)

        if override_dir is not None:
            ig_dir = override_dir
        else:
            ig_dir = discover_ig_directory(args.root_dir, dataset, prefer_td=args.prefer_td)

        if ig_dir is None:
            print(f"✗ Skipping {dataset}: no IG directory found under {args.root_dir}.")
            continue

        if not ig_dir.exists() or not ig_dir.is_dir():
            print(f"✗ Skipping {dataset}: override path {ig_dir} is not a directory.")
            continue

        ig_files = sorted(ig_dir.glob("*_ig*.npz"), key=ig_file_sort_key)

        if not ig_files:
            print(
                f"✗ Skipping {dataset}: directory {ig_dir} contains no files matching '*_ig*.npz'."
            )
            continue

        print(
            f"==============================================================================\n"
            f"DATASET: {dataset}\n"
            f"Using IG directory: {ig_dir}\n"
            f"Found {len(ig_files)} IG files\n"
            f"=============================================================================="
        )

        override_subjects, override_ages = age_sources.get(dataset, (None, None))

        try:
            (
                network_names,
                subjects,
                ages,
                network_matrix,
                target_arrays,
            ) = aggregate_subject_networks(
                ig_files,
                network_map,
                aggregation_method=args.aggregation_method,
                age_key=args.age_key,
                subject_key=args.subject_key,
                target_key_map=target_key_map,
                collect_chronological=collect_chronological,
                verbose=args.verbose,
                override_subjects=override_subjects,
                override_ages=override_ages,
            )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  ✗ Failed on {dataset}: {exc}")
            continue

        target_arrays_map: OrderedDict[str, np.ndarray] = OrderedDict()
        if collect_chronological:
            target_arrays_map[args.chronological_label] = ages
        for label, arr in target_arrays.items():
            target_arrays_map[label] = arr

        dataset_target_sources = target_sources_map.get(dataset, [])
        if dataset_target_sources:
            subject_index = {str(subj): idx for idx, subj in enumerate(subjects)}
            for label, ext_ids, ext_values in dataset_target_sources:
                aligned = np.full(len(subjects), np.nan, dtype=float)
                matched = 0
                for sid, value in zip(ext_ids, ext_values):
                    idx = subject_index.get(str(sid))
                    if idx is not None:
                        aligned[idx] = float(value)
                        matched += 1
                if matched == 0:
                    if len(ext_values) == len(subjects):
                        aligned = np.asarray(ext_values, dtype=float)
                        target_arrays_map[label] = aligned
                        print(
                            f"  ⚠︎ External target '{label}' for {dataset} supplied no matching IDs; "
                            "assuming provided order matches IG subjects."
                        )
                        continue
                    print(
                        f"  ⚠︎ External target '{label}' for {dataset} had no matching subject IDs; skipping."
                    )
                    continue
                target_arrays_map[label] = aligned

        network_index = {name: idx for idx, name in enumerate(network_names)}

        if args.scatter_plots and scatter_root is not None:
            dataset_scatter_dir = scatter_root / dataset
            dataset_scatter_dir.mkdir(parents=True, exist_ok=True)
        else:
            dataset_scatter_dir = None

        target_sequences: List[Tuple[str, np.ndarray]] = list(target_arrays_map.items())

        if not target_sequences:
            print(f"  ✗ No targets available for {dataset}; skipping.")
            continue

        per_dataset_frames: List[pd.DataFrame] = []

        for label, values in target_sequences:
            if not np.isfinite(values).any():
                if args.verbose:
                    print(
                        f"  ⚠︎ Skipping target '{label}' for {dataset}: all values are NaN or missing."
                    )
                continue

            correlations_df = compute_correlations(
                network_names,
                values,
                network_matrix,
                target_label=label,
                apply_fdr=args.apply_fdr,
            )

            if correlations_df.empty:
                if args.verbose:
                    print(
                        f"  ⚠︎ Target '{label}' for {dataset} produced fewer than 3 valid subjects per network."
                    )
                continue

            correlations_df.insert(0, "Dataset", dataset)
            correlations_df.insert(1, "Parcellation", args.parcellation)
            correlations_df.insert(2, "Aggregation", args.aggregation_method)

            if args.scatter_plots and dataset_scatter_dir is not None:
                target_scatter_dir = dataset_scatter_dir / sanitize_filename(label)
                target_scatter_dir.mkdir(parents=True, exist_ok=True)
                target_vector = target_arrays_map[label]

                for _, row in correlations_df.iterrows():
                    network_name = row["Network"]
                    spearman_p = float(row["Spearman_p"])
                    spearman_rho = float(row["Spearman_rho"])
                    n_subjects = int(row["N_Subjects"])

                    if not np.isfinite(spearman_p) or not np.isfinite(spearman_rho):
                        continue
                    if spearman_p > args.scatter_alpha:
                        continue
                    if abs(spearman_rho) < args.scatter_min_abs_rho:
                        continue
                    if n_subjects < args.scatter_min_n:
                        continue

                    idx = network_index.get(network_name)
                    if idx is None:
                        continue

                    network_values = network_matrix[:, idx]
                    mask = np.isfinite(target_vector) & np.isfinite(network_values)
                    if mask.sum() < args.scatter_min_n:
                        continue

                    create_scatter_figure(
                        dataset_key=dataset,
                        target_label=label,
                        network_name=network_name,
                        x_values=target_vector[mask],
                        y_values=network_values[mask],
                        rho=spearman_rho,
                        p_value=spearman_p,
                        output_dir=target_scatter_dir,
                        chronological_label=args.chronological_label,
                        title_suffix=network_name,
                    )

            per_dataset_frames.append(correlations_df)

            target_slug = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_") or "target"
            dataset_csv = output_dir / f"{dataset}_{target_slug}_network_correlations.csv"
            correlations_df.to_csv(dataset_csv, index=False)
            print(f"  ✓ Saved dataset summary ({label}) → {dataset_csv}")

        if not per_dataset_frames:
            print(f"  ✗ No valid correlations computed for {dataset} (all targets skipped).")
            continue

        for frame in per_dataset_frames:
            all_dataset_rows.append(frame)

        if args.save_subject_level:
            subject_df = pd.DataFrame(network_matrix, columns=network_names)
            subject_df.insert(0, "Subject", subjects)
            insert_position = 1
            if collect_chronological:
                subject_df.insert(insert_position, args.chronological_label, ages)
                insert_position += 1
            for label, arr in target_arrays.items():
                # Avoid clobbering chronological column if label duplicates it
                if collect_chronological and label == args.chronological_label:
                    continue
                subject_df.insert(insert_position, label, arr)
                insert_position += 1
            subject_csv = output_dir / f"{dataset}_subject_network_values.csv"
            subject_df.to_csv(subject_csv, index=False)
            print(f"  ✓ Saved subject-level matrix → {subject_csv}")

    if all_dataset_rows:
        combined = pd.concat(all_dataset_rows, ignore_index=True)
        combined_path = (
            output_dir
            / f"network_correlations_{args.parcellation}_{args.aggregation_method}.csv"
        )
        combined.to_csv(combined_path, index=False)
        print(f"\n✓ Combined summary saved → {combined_path}")
    else:
        print("\nNo datasets processed successfully.")


if __name__ == "__main__":
    main()

