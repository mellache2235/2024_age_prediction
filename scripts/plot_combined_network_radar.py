#!/usr/bin/env python3
"""Create publication-ready radar plots comparing TD, ADHD, and ASD network attributions.

- Default run: generates the count-based overlap radar (three panels: TD, ADHD, ASD).
- When mean-IG summaries are supplied, produces cohort-specific grids:
  * TD → 2×2 grid (HCP-Development, NKI-RS TD, CMI-HBN TD, ADHD-200 TD)
  * ADHD → 1×2 grid
  * ASD → 1×2 grid
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.backends.backend_pdf as pdf
import matplotlib.colors as mcolors
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import OrderedDict
import warnings

# -----------------------------------------------------------------------------
# GLOBAL STYLING CONSTANTS
# -----------------------------------------------------------------------------

FONT_PATH = \
    "/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/" \
    "clustering_analysis/arial.ttf"

# Load Arial when available so Illustrator/AI exports match the rest of the figures
if os.path.exists(FONT_PATH):
    font_manager.fontManager.addfont(FONT_PATH)
    prop = font_manager.FontProperties(fname=FONT_PATH)
    plt.rcParams["font.family"] = prop.get_name()
else:
    plt.rcParams["font.family"] = "Arial"

# Consistent order for Yeo-17 networks (matches grant figure)
DEFAULT_NETWORK_ORDER: Tuple[str, ...] = (
    "VisCent",
    "VisPeri",
    "SomMotA",
    "SomMotB",
    "DorsAttnA",
    "DorsAttnB",
    "SalVentAttnA",
    "SalVentAttnB",
    "LimbicA",
    "LimbicB",
    "FPA",
    "FPB",
    "FPC",
    "DefaultA",
    "DefaultB",
    "DefaultC",
    "TempPar",
    "AmyHip",
    "Thalamus",
    "Striatum",
)

# Pastel rainbow palette for the radial bars (len(order) colours)
def pastel_spectral(n: int) -> List[str]:
    cmap = plt.cm.get_cmap("Spectral")
    colors = [mcolors.to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]
    # Lighten colours slightly to match example aesthetic
    lightened = []
    for hex_color in colors:
        r, g, b = mcolors.hex2color(hex_color)
        r = min(1.0, r * 0.85 + 0.15)
        g = min(1.0, g * 0.85 + 0.15)
        b = min(1.0, b * 0.85 + 0.15)
        lightened.append(mcolors.to_hex((r, g, b)))
    return lightened


def load_network_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Network CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Network CSV is empty: {csv_path}")
    return df


def extract_network_series(
    df: pd.DataFrame,
    network_order: Sequence[str],
    prefer_total: bool = True,
    value_overrides: Optional[Sequence[str]] = None,
) -> pd.Series:
    """Return a Series indexed by network_order with attribution values."""

    if "network" in df.columns:
        network_col = "network"
    elif "Network" in df.columns:
        network_col = "Network"
    else:
        raise KeyError("CSV must include 'network' or 'Network' column")

    default_candidates = [
        "Effect_Size_Pct",
        "Effect_Size",
        "total_attribution",
        "Count",
        "mean_attribution",
        "Mean_IG",
        "abs_mean",
        "mean",
    ]

    value_column_candidates = []
    if value_overrides:
        value_column_candidates.extend(value_overrides)
    if prefer_total:
        value_column_candidates.extend(["total_attribution", "Count"])
    value_column_candidates.extend(default_candidates)

    value_col = None
    for candidate in value_column_candidates:
        if candidate in df.columns:
            value_col = candidate
            break

    if value_col is None:
        raise KeyError(
            "CSV must contain one of 'total_attribution', 'mean_attribution', or 'Count'."
        )

    series = df[[network_col, value_col]].copy()
    series.rename(columns={network_col: "network", value_col: "value"}, inplace=True)
    series["network"] = series["network"].astype(str)

    # Aggregate duplicates if any
    agg = series.groupby("network")["value"].sum()

    # Align to desired order, fill missing with 0
    aligned = pd.Series(index=list(network_order), dtype=float)
    aligned.loc[:] = 0.0
    for network, val in agg.items():
        if network in aligned.index:
            aligned[network] = val
    return aligned


def normalize_series(series_dict: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
    """Normalize all series to [0, 1] using shared max for consistent scaling."""

    all_values = np.concatenate([s.values for s in series_dict.values()])
    max_val = np.nanmax(all_values) if np.any(np.isfinite(all_values)) else 1.0
    if max_val == 0:
        max_val = 1.0
    normalized = {key: series / max_val for key, series in series_dict.items()}
    return normalized


def load_effect_size_series(
    csv_path: Path,
    network_order: Sequence[str],
    target_label: str,
    value_column: str,
    absolute_values: bool = True,
    aggregation: str = "mean",
) -> pd.Series:
    """Extract effect-size style network series from correlation summaries."""

    df = load_network_csv(csv_path)

    available_targets = None
    if "Target" in df.columns:
        available_targets = sorted(df["Target"].astype(str).unique())
        filtered = df[df["Target"].astype(str) == target_label]
        if filtered.empty:
            raise ValueError(
                f"No rows in {csv_path} matched Target='{target_label}'. Available targets: {available_targets}"
            )
        df = filtered

    if "Network" not in df.columns:
        raise KeyError(f"Expected 'Network' column in {csv_path}.")

    if value_column not in df.columns:
        raise KeyError(
            f"Effect-size column '{value_column}' not present in {csv_path}. Columns: {list(df.columns)}"
        )

    values = df[["Network", value_column]].copy()
    values.rename(columns={"Network": "network", value_column: "value"}, inplace=True)
    values["network"] = values["network"].astype(str)

    if absolute_values:
        values["value"] = values["value"].abs()

    grouped = values.groupby("network")["value"]
    if aggregation == "mean":
        agg_values = grouped.mean()
    elif aggregation == "sum":
        agg_values = grouped.sum()
    elif aggregation == "median":
        agg_values = grouped.median()
    else:
        raise ValueError("aggregation must be one of 'mean', 'sum', or 'median'.")

    aligned = pd.Series(index=list(network_order), dtype=float)
    aligned.loc[:] = 0.0
    for network, val in agg_values.items():
        if network in aligned.index:
            aligned[network] = float(val)

    if not absolute_values:
        finite_vals = aligned[np.isfinite(aligned)]
        if not finite_vals.empty:
            min_val = float(finite_vals.min())
            if min_val < 0:
                aligned = aligned - min_val

    return aligned


def create_radar_panel(
    ax: plt.Axes,
    values: Sequence[float],
    labels: Sequence[str],
    title: str,
    colors: Sequence[str],
) -> None:
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    bar_width = (2 * np.pi) / n * 0.95

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_facecolor("white")

    bars = ax.bar(
        angles,
        values,
        width=bar_width,
        color=colors,
        edgecolor="white",
        linewidth=1.2,
        align="edge",
    )

    # Add subtle radial gridlines
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_yticks([])
    ax.set_ylim(0, 1.05)
    ax.grid(False)

    # Concentric reference rings
    radii = [0.25, 0.5, 0.75, 1.0]
    for r in radii:
        ax.plot(np.linspace(0, 2 * np.pi, 360), np.full(360, r), color="#DDDDDD", linewidth=0.6)

    ax.set_xticks(angles + bar_width / 2)
    ax.set_xticklabels(labels, fontsize=12, fontweight="bold", color="#111111")

    ax.set_title(title, fontsize=18, fontweight="bold", pad=30, color="#111111")

    # Lift bars slightly (zorder) for cleaner overlap
    for bar in bars:
        bar.set_zorder(5)


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    png_path = output_path.with_suffix(".png")
    tiff_path = output_path.with_suffix(".tiff")
    ai_path = output_path.with_suffix(".ai")

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    fig.savefig(
        tiff_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        format="tiff",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    pdf.FigureCanvas(fig).print_pdf(str(ai_path))


def parse_labeled_paths(
    entries: Optional[Sequence[str]],
    group_name: str,
    expected_count: Optional[int] = None,
) -> List[Tuple[str, Path]]:
    if not entries:
        return []

    parsed: List[Tuple[str, Path]] = []
    for raw in entries:
        if "=" not in raw:
            raise ValueError(
                f"Invalid {group_name} entry '{raw}'. Expected format LABEL=/absolute/path/to.csv"
            )
        label, path_str = raw.split("=", 1)
        label = label.strip()
        path = Path(path_str).expanduser()
        if not label:
            raise ValueError(f"{group_name} entry '{raw}' missing label before '='.")
        if not path.exists():
            raise FileNotFoundError(f"File not found for {group_name} entry '{raw}': {path}")
        parsed.append((label, path))

    if expected_count is not None and len(parsed) != expected_count:
        raise ValueError(
            f"Expected {expected_count} labeled CSVs for {group_name}, received {len(parsed)}."
        )

    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create three-panel network radar plot for TD, ADHD, and ASD cohorts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python plot_combined_network_radar.py \
    --td /oak/.../td_shared_network_analysis.csv \
    --adhd /oak/.../adhd_shared_network_analysis.csv \
    --asd /oak/.../asd_shared_network_analysis.csv \
    --output /oak/.../figures/network_radar_panels
""",
    )

    parser.add_argument(
        "--td",
        type=Path,
        help="Count-based CSV for TD cohorts (optional if only ADHD/ASD plots are desired)."
    )
    parser.add_argument(
        "--adhd",
        type=Path,
        help="Count-based CSV for ADHD cohorts (optional)."
    )
    parser.add_argument(
        "--asd",
        type=Path,
        help="Count-based CSV for ASD cohorts (optional)."
    )
    parser.add_argument("--output", required=True, type=Path, help="Output path (without extension)")
    parser.add_argument(
        "--network-order",
        nargs="+",
        help="Custom network ordering; defaults to Yeo-17 order matching grant figure.",
    )
    parser.add_argument(
        "--radius-label",
        type=str,
        default="Normalized Attribution",
        help="Label displayed beneath panels to denote radial magnitude.",
    )
    parser.add_argument(
        "--prefer-total",
        action="store_true",
        help="Prefer total_attribution/Count column when available (default: mean).",
    )
    parser.add_argument(
        "--td-ig",
        action="append",
        metavar="LABEL=PATH",
        help="Mean-IG CSVs for TD layout (provide four entries: HCP-Development, NKI-RS TD, CMI-HBN TD, ADHD-200 TD).",
    )
    parser.add_argument(
        "--adhd-ig",
        action="append",
        metavar="LABEL=PATH",
        help="Mean-IG CSVs for ADHD layout (provide two entries, e.g., ADHD-200 ADHD, CMI-HBN ADHD).",
    )
    parser.add_argument(
        "--asd-ig",
        action="append",
        metavar="LABEL=PATH",
        help="Mean-IG CSVs for ASD layout (provide two entries, e.g., Stanford ASD, ABIDE ASD).",
    )
    parser.add_argument(
        "--ig-target",
        type=str,
        default="Chronological_Age",
        help="Target label to filter within IG summary CSVs (matches 'Target' column).",
    )
    parser.add_argument(
        "--ig-column",
        type=str,
        default="Mean_IG",
        help="Column name to use for effect sizes in IG summary CSVs (default: Mean_IG).",
    )
    parser.add_argument(
        "--ig-aggregation",
        choices=["mean", "sum", "median"],
        default="mean",
        help="Aggregation applied when multiple rows per network are present (default: mean).",
    )
    parser.add_argument(
        "--no-ig-abs",
        action="store_true",
        help="Keep signed IG values instead of absolute magnitudes when plotting effect sizes.",
    )
    parser.add_argument(
        "--ig-radius-label",
        type=str,
        default="Normalized |Mean IG|",
        help="Label displayed beneath effect-size panels.",
    )

    return parser.parse_args()


def render_radar_grid(
    series_dict: "OrderedDict[str, pd.Series]",
    network_order: Sequence[str],
    colors: Sequence[str],
    layout: Tuple[int, int],
    title_prefix: str,
    radius_label: str,
    output_path: Path,
) -> None:
    rows, cols = layout
    normalized = normalize_series(series_dict)

    fig, axes = plt.subplots(
        rows,
        cols,
        subplot_kw={"projection": "polar"},
        figsize=(6 * cols, 6 * rows),
    )

    axes_iter = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    for ax, (label, series) in zip(axes_iter, normalized.items()):
        create_radar_panel(ax, series.values, network_order, label, colors)

    # Hide any unused axes
    for ax in axes_iter[len(series_dict) :]:
        ax.set_visible(False)

    fig.text(0.5, 0.02, radius_label, ha="center", fontsize=15, fontweight="bold")
    fig.subplots_adjust(wspace=0.35, hspace=0.35, bottom=0.12)

    save_figure(fig, output_path)
    print(f"✓ Saved {title_prefix} radar panels to {output_path}")


def main() -> None:
    args = parse_args()

    network_order = tuple(args.network_order) if args.network_order else DEFAULT_NETWORK_ORDER
    colors = pastel_spectral(len(network_order))

    count_datasets: "OrderedDict[str, pd.Series]" = OrderedDict()
    if args.td:
        count_datasets["TD"] = extract_network_series(
            load_network_csv(args.td),
            network_order,
            args.prefer_total,
        )
    if args.adhd:
        count_datasets["ADHD"] = extract_network_series(
            load_network_csv(args.adhd),
            network_order,
            args.prefer_total,
        )
    if args.asd:
        count_datasets["ASD"] = extract_network_series(
            load_network_csv(args.asd),
            network_order,
            args.prefer_total,
        )

    if count_datasets:
        normalized_counts = normalize_series(count_datasets)

        fig, axes = plt.subplots(
            1,
            len(count_datasets),
            subplot_kw={"projection": "polar"},
            figsize=(6 * len(count_datasets), 6.5),
        )
        if len(count_datasets) == 1:
            axes = [axes]

        for ax, (label, series) in zip(axes, normalized_counts.items()):
            create_radar_panel(ax, series.values, network_order, label, colors)

        if len(count_datasets) < len(axes):
            for ax in axes[len(count_datasets) :]:
                ax.set_visible(False)

        fig.text(0.5, 0.02, args.radius_label, ha="center", fontsize=15, fontweight="bold")
        fig.subplots_adjust(wspace=0.35, bottom=0.12)

        save_figure(fig, args.output)
        print(f"✓ Saved radar panels to {args.output}")

    td_effect_entries = parse_labeled_paths(args.td_ig, "TD", expected_count=len(args.td_ig) if args.td_ig else None)
    adhd_effect_entries = parse_labeled_paths(args.adhd_ig, "ADHD", expected_count=len(args.adhd_ig) if args.adhd_ig else None)
    asd_effect_entries = parse_labeled_paths(args.asd_ig, "ASD", expected_count=len(args.asd_ig) if args.asd_ig else None)

    if td_effect_entries or adhd_effect_entries or asd_effect_entries:
        target_slug = re.sub(r"[^A-Za-z0-9]+", "_", args.ig_target).strip("_") or "target"
        column_slug = re.sub(r"[^A-Za-z0-9]+", "_", args.ig_column).strip("_") or "metric"
        base_output = args.output.parent / args.output.name

        if td_effect_entries:
            if len(td_effect_entries) != 4:
                warnings.warn(
                    "TD effect-size radar expects four datasets; skipping TD layout."
                )
            else:
                td_series_dict: "OrderedDict[str, pd.Series]" = OrderedDict()
                for label, csv_path in td_effect_entries:
                    td_series_dict[label] = load_effect_size_series(
                        csv_path,
                        network_order,
                        target_label=args.ig_target,
                        value_column=args.ig_column,
                        absolute_values=not args.no_ig_abs,
                        aggregation=args.ig_aggregation,
                    )

                td_output = base_output.parent / f"{base_output.name}_td_effect_{target_slug}_{column_slug}"
                render_radar_grid(
                    td_series_dict,
                    network_order,
                    colors,
                    layout=(2, 2),
                    title_prefix="TD",
                    radius_label=args.ig_radius_label,
                    output_path=td_output,
                )

        if adhd_effect_entries:
            if len(adhd_effect_entries) != 2:
                warnings.warn(
                    "ADHD effect-size radar expects two datasets; skipping ADHD layout."
                )
            else:
                adhd_series_dict: "OrderedDict[str, pd.Series]" = OrderedDict()
                for label, csv_path in adhd_effect_entries:
                    adhd_series_dict[label] = load_effect_size_series(
                        csv_path,
                        network_order,
                        target_label=args.ig_target,
                        value_column=args.ig_column,
                        absolute_values=not args.no_ig_abs,
                        aggregation=args.ig_aggregation,
                    )

                adhd_output = base_output.parent / f"{base_output.name}_adhd_effect_{target_slug}_{column_slug}"
                render_radar_grid(
                    adhd_series_dict,
                    network_order,
                    colors,
                    layout=(1, 2),
                    title_prefix="ADHD",
                    radius_label=args.ig_radius_label,
                    output_path=adhd_output,
                )

        if asd_effect_entries:
            if len(asd_effect_entries) != 2:
                warnings.warn(
                    "ASD effect-size radar expects two datasets; skipping ASD layout."
                )
            else:
                asd_series_dict: "OrderedDict[str, pd.Series]" = OrderedDict()
                for label, csv_path in asd_effect_entries:
                    asd_series_dict[label] = load_effect_size_series(
                        csv_path,
                        network_order,
                        target_label=args.ig_target,
                        value_column=args.ig_column,
                        absolute_values=not args.no_ig_abs,
                        aggregation=args.ig_aggregation,
                    )

                asd_output = base_output.parent / f"{base_output.name}_asd_effect_{target_slug}_{column_slug}"
                render_radar_grid(
                    asd_series_dict,
                    network_order,
                    colors,
                    layout=(1, 2),
                    title_prefix="ASD",
                    radius_label=args.ig_radius_label,
                    output_path=asd_output,
                )


if __name__ == "__main__":
    main()

