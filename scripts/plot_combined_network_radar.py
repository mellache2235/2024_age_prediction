#!/usr/bin/env python3
"""Create three-panel radar plots comparing TD, ADHD, and ASD network attributions.

Generates the traditional count-based radar (default input CSVs) and, when
provided with network correlation summaries from `compute_network_age_correlations.py`,
produces an additional effect-size radar that visualizes mean IG magnitudes across
500 folds (or other supplied metrics).
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

    parser.add_argument("--td", required=True, type=Path, help="CSV for TD cohorts")
    parser.add_argument("--adhd", required=True, type=Path, help="CSV for ADHD cohorts")
    parser.add_argument("--asd", required=True, type=Path, help="CSV for ASD cohorts")
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
        type=Path,
        help="Optional network correlations CSV for TD cohort (Mean IG effect sizes).",
    )
    parser.add_argument(
        "--adhd-ig",
        type=Path,
        help="Optional network correlations CSV for ADHD cohort (Mean IG effect sizes).",
    )
    parser.add_argument(
        "--asd-ig",
        type=Path,
        help="Optional network correlations CSV for ASD cohort (Mean IG effect sizes).",
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


def main() -> None:
    args = parse_args()

    network_order = tuple(args.network_order) if args.network_order else DEFAULT_NETWORK_ORDER

    td_series = extract_network_series(load_network_csv(args.td), network_order, args.prefer_total)
    adhd_series = extract_network_series(load_network_csv(args.adhd), network_order, args.prefer_total)
    asd_series = extract_network_series(load_network_csv(args.asd), network_order, args.prefer_total)

    all_series = {"TD": td_series, "ADHD": adhd_series, "ASD": asd_series}
    normalized = normalize_series(all_series)

    colors = pastel_spectral(len(network_order))

    fig, axes = plt.subplots(
        1,
        3,
        subplot_kw={"projection": "polar"},
        figsize=(18, 6.5),
    )

    titles = ["(A) TD", "(B) ADHD", "(C) ASD"]
    for ax, key, title in zip(axes, normalized.keys(), titles):
        create_radar_panel(ax, normalized[key].values, network_order, title, colors)

    # Shared label underneath panels
    fig.text(0.5, 0.02, args.radius_label, ha="center", fontsize=15, fontweight="bold")

    fig.subplots_adjust(wspace=0.35, bottom=0.12)

    save_figure(fig, args.output)
    print(f"✓ Saved radar panels to {args.output}")

    effect_inputs = {
        "TD": args.td_ig,
        "ADHD": args.adhd_ig,
        "ASD": args.asd_ig,
    }

    provided_effect_paths = {k: v for k, v in effect_inputs.items() if v is not None}
    if provided_effect_paths:
        missing = [k for k, v in effect_inputs.items() if v is None]
        if missing:
            raise ValueError(
                "Effect-size radar requested but missing IG summary for cohorts: "
                + ", ".join(missing)
            )

        effect_series: Dict[str, pd.Series] = {}
        for cohort, csv_path in effect_inputs.items():
            effect_series[cohort] = load_effect_size_series(
                csv_path,
                network_order,
                target_label=args.ig_target,
                value_column=args.ig_column,
                absolute_values=not args.no_ig_abs,
                aggregation=args.ig_aggregation,
            )

        normalized_effect = normalize_series(effect_series)

        fig_effect, axes_effect = plt.subplots(
            1,
            3,
            subplot_kw={"projection": "polar"},
            figsize=(18, 6.5),
        )

        for ax, cohort, title in zip(axes_effect, normalized_effect.keys(), titles):
            create_radar_panel(ax, normalized_effect[cohort].values, network_order, title, colors)

        fig_effect.text(
            0.5,
            0.02,
            args.ig_radius_label,
            ha="center",
            fontsize=15,
            fontweight="bold",
        )

        fig_effect.subplots_adjust(wspace=0.35, bottom=0.12)

        target_slug = re.sub(r"[^A-Za-z0-9]+", "_", args.ig_target).strip("_") or "target"
        column_slug = re.sub(r"[^A-Za-z0-9]+", "_", args.ig_column).strip("_") or "metric"
        effect_output = args.output.parent / f"{args.output.name}_effect_{target_slug}_{column_slug}"

        save_figure(fig_effect, effect_output)
        print(f"✓ Saved effect-size radar panels to {effect_output}")


if __name__ == "__main__":
    main()

