#!/usr/bin/env python3
"""
Centralized plot styling for ALL scatter plots.

SINGLE SOURCE OF TRUTH for consistent formatting across:
- Brain-behavior scatter plots
- Brain age prediction scatter plots
- Combined subplot figures

This ensures plots can be placed side-by-side without visible style differences.

Author: Standardized Plotting
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.backends.backend_pdf as pdf
from pathlib import Path
from typing import Optional, Tuple

# ============================================================================
# GLOBAL CONSTANTS - SINGLE SOURCE OF TRUTH
# ============================================================================

# Font
FONT_PATH = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/clustering_analysis/arial.ttf'

# Colors - Publication-ready, NO post-processing needed
COLOR_DOTS = '#5A6FA8'           # Bluer dots (fill AND edge - same color)
COLOR_LINE = '#D32F2F'           # Red regression line (vibrant, clearly visible)
COLOR_BACKGROUND = '#FFFFFF'     # Pure white background
COLOR_SPINE = '#000000'          # Black spines (crisp, professional)
COLOR_TICK = '#000000'           # Black tick labels (high contrast)
COLOR_STATS_TEXT = '#000000'     # Black statistics text (clear, readable)

# Scatter properties - Optimized for Affinity Designer readiness
SCATTER_ALPHA = 0.7              # Slight transparency for overlapping points
SCATTER_SIZE = 100               # Larger for better visibility (was 80)
SCATTER_LINEWIDTH = 1.5          # Thicker edge for definition (was 0.8)
SCATTER_EDGECOLOR = '#5A6FA8'    # SAME as fill color (uniform appearance)

# Line properties - Bold and clearly visible
LINE_ALPHA = 1.0                 # Fully opaque (no transparency)
LINE_WIDTH = 3.0                 # Clear visibility without being too thick
LINE_STYLE = '-'                 # Solid line

# Spine properties - Professional and crisp (not too heavy)
SPINE_WIDTH = 1.5                # Balanced thickness
SPINE_COLOR = '#000000'          # Black for professional look

# Tick properties - Clear and readable
TICK_LENGTH_MAJOR = 6            # Standard length for visibility
TICK_WIDTH_MAJOR = 1.2           # Visible but not heavy
TICK_LABELSIZE = 14              # Larger for readability (was 11)
TICK_COLOR = '#000000'           # Black for maximum contrast
TICK_PAD = 6                     # Good spacing

# Font sizes - Publication-ready (no need to increase later)
FONTSIZE_LABEL = 16              # Large, clear labels (was 13)
FONTSIZE_TITLE = 18              # Prominent title (was 15)
FONTSIZE_STATS = 14              # Readable statistics (was 11)
FONTSIZE_STATS_SUBPLOT = 12      # Readable in subplots (was 10)

# Font weights
FONTWEIGHT_LABEL = 'normal'
FONTWEIGHT_TITLE = 'bold'

# Title padding
TITLE_PAD = 20                   # More space for breathing room
TITLE_PAD_SUBPLOT = 12

# Figure sizes (width, height in inches)
FIGSIZE_SINGLE = (8, 6)          # Single scatter plot
FIGSIZE_SUBPLOT = (6, 4.5)       # Individual subplot in multi-panel
DPI = 300                         # High resolution for publications

# Aesthetic enhancements
USE_TIGHT_LAYOUT = True
TIGHT_LAYOUT_PAD = 1.5           # Padding for tight_layout
FIGURE_FACECOLOR = 'white'       # Figure background
AXES_FACECOLOR = 'white'         # Axes background

# Dataset title mapping - Clear, publication-ready titles
DATASET_TITLES = {
    'nki': 'NKI-RS',
    'nki_rs': 'NKI-RS',
    'nki_rs_td': 'NKI-RS',
    'nki_td': 'NKI-RS',
    
    'adhd200': 'ADHD-200',
    'adhd200_td': 'ADHD-200 TD Subset (NYU)',
    'adhd200_adhd': 'ADHD-200 ADHD',
    
    'cmihbn': 'CMI-HBN',
    'cmihbn_td': 'CMI-HBN TD Subset',
    'cmihbn_adhd': 'CMI-HBN ADHD',
    
    'abide': 'ABIDE',
    'abide_asd': 'ABIDE ASD',
    'abide_td': 'ABIDE TD',
    
    'stanford': 'Stanford',
    'stanford_asd': 'Stanford ASD',
    
    'hcp_dev': 'HCP-Dev',
    'dev': 'HCP-Dev'
}


def get_dataset_title(dataset_key: str) -> str:
    """
    Get publication-ready title for dataset.
    
    Args:
        dataset_key: Internal dataset identifier (e.g., 'adhd200_td')
    
    Returns:
        Publication-ready title (e.g., 'ADHD-200 TD Subset (NYU)')
    """
    # Convert to lowercase for matching
    key_lower = dataset_key.lower().replace('_', '').replace('-', '')
    
    # Try exact match first
    if dataset_key in DATASET_TITLES:
        return DATASET_TITLES[dataset_key]
    
    # Try fuzzy matching
    for key, title in DATASET_TITLES.items():
        key_clean = key.lower().replace('_', '').replace('-', '')
        if key_clean == key_lower:
            return title
    
    # Fallback: format nicely
    return dataset_key.replace('_', '-').upper()


# ============================================================================
# FONT SETUP
# ============================================================================

def setup_arial_font():
    """Setup Arial font globally."""
    try:
        if Path(FONT_PATH).exists():
            font_manager.fontManager.addfont(FONT_PATH)
            prop = font_manager.FontProperties(fname=FONT_PATH)
            plt.rcParams['font.family'] = prop.get_name()
            return True
    except Exception as e:
        print(f"Warning: Could not load Arial font: {e}")
        return False


# ============================================================================
# CORE SCATTER PLOT FUNCTION
# ============================================================================

def create_standardized_scatter(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str = 'Observed',
    ylabel: str = 'Predicted',
    stats_text: Optional[str] = None,
    is_subplot: bool = False
) -> None:
    """
    Create a standardized scatter plot with EXACT consistent styling.
    
    This is the SINGLE SOURCE OF TRUTH for all scatter plot formatting.
    Optimized for publication quality with modern, clean aesthetics.
    
    Args:
        ax: Matplotlib axes object
        x: X-axis data (observed values)
        y: Y-axis data (predicted values)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        stats_text: Optional statistics text (e.g., "ρ = 0.75\np < 0.001")
        is_subplot: If True, uses smaller fonts for multi-panel figures
    """
    # Set background colors
    ax.set_facecolor(AXES_FACECOLOR)
    
    # Scatter plot with subtle edge
    ax.scatter(
        x, y,
        alpha=SCATTER_ALPHA,
        s=SCATTER_SIZE,
        color=COLOR_DOTS,
        edgecolors=SCATTER_EDGECOLOR,
        linewidth=SCATTER_LINEWIDTH
    )
    
    # Best fit line - smooth and prominent
    coeffs = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = np.polyval(coeffs, x_line)
    
    ax.plot(
        x_line, y_line,
        color=COLOR_LINE,
        linewidth=LINE_WIDTH,
        alpha=LINE_ALPHA,
        linestyle=LINE_STYLE,
        zorder=10  # Ensure line is on top
    )
    
    # Statistics text (if provided)
    if stats_text:
        fontsize = FONTSIZE_STATS_SUBPLOT if is_subplot else FONTSIZE_STATS
        ax.text(
            0.95, 0.05,
            stats_text,
            transform=ax.transAxes,
            fontsize=fontsize,
            verticalalignment='bottom',
            horizontalalignment='right',
            color=COLOR_STATS_TEXT,
            linespacing=1.5  # Better spacing between lines
            # NO bbox parameter - no bounding box
        )
    
    # Labels - clean and readable
    ax.set_xlabel(xlabel, fontsize=FONTSIZE_LABEL, fontweight=FONTWEIGHT_LABEL, 
                 color=COLOR_TICK, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABEL, fontweight=FONTWEIGHT_LABEL,
                 color=COLOR_TICK, labelpad=8)
    
    # Title - prominent but not overwhelming
    title_pad = TITLE_PAD_SUBPLOT if is_subplot else TITLE_PAD
    ax.set_title(title, fontsize=FONTSIZE_TITLE, fontweight=FONTWEIGHT_TITLE, 
                pad=title_pad, color=COLOR_TICK)
    
    # Clean minimalist style - NO top/right spines
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(SPINE_WIDTH)
    ax.spines['bottom'].set_linewidth(SPINE_WIDTH)
    ax.spines['left'].set_color(SPINE_COLOR)
    ax.spines['bottom'].set_color(SPINE_COLOR)
    
    # Ticks - MAJOR ONLY, clean and minimal
    ax.tick_params(
        axis='both',
        which='major',
        labelsize=TICK_LABELSIZE,
        direction='out',
        length=TICK_LENGTH_MAJOR,
        width=TICK_WIDTH_MAJOR,
        top=False,
        right=False,
        colors=TICK_COLOR,
        pad=TICK_PAD
    )
    
    # Subtle aesthetic touch - ensure no minor ticks
    ax.minorticks_off()


# ============================================================================
# HIGH-LEVEL FUNCTIONS
# ============================================================================

def create_single_scatter_plot(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    save_path: str,
    xlabel: str = 'Observed',
    ylabel: str = 'Predicted',
    stats_dict: Optional[dict] = None
) -> None:
    """
    Create and save a single scatter plot with standard styling.
    
    Args:
        x: X-axis data
        y: Y-axis data
        title: Plot title
        save_path: Output path (without extension)
        xlabel: X-axis label
        ylabel: Y-axis label
        stats_dict: Optional dict with 'rho', 'p_value', 'mae', 'r2', etc.
    """
    # Setup font
    setup_arial_font()
    
    # Create figure with aesthetic settings
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE, dpi=100)
    fig.patch.set_facecolor(FIGURE_FACECOLOR)
    
    # Format statistics
    stats_text = None
    if stats_dict:
        lines = []
        if 'rho' in stats_dict or 'spearman_rho' in stats_dict:
            rho = stats_dict.get('rho', stats_dict.get('spearman_rho', 0))
            lines.append(f"ρ = {rho:.3f}")
        if 'p_value' in stats_dict:
            lines.append(f"p = {stats_dict['p_value']:.4f}")
        if 'mae' in stats_dict:
            lines.append(f"MAE = {stats_dict['mae']:.2f}")
        if 'r2' in stats_dict:
            lines.append(f"R² = {stats_dict['r2']:.3f}")
        if lines:
            stats_text = '\n'.join(lines)
    
    # Create scatter
    create_standardized_scatter(
        ax, x, y, title, xlabel, ylabel, stats_text, is_subplot=False
    )
    
    # Apply tight layout with aesthetic padding
    if USE_TIGHT_LAYOUT:
        plt.tight_layout(pad=TIGHT_LAYOUT_PAD)
    
    # Save in multiple formats with optimal settings
    save_path = Path(save_path)
    png_path = save_path.with_suffix('.png')
    tiff_path = save_path.with_suffix('.tiff')
    ai_path = save_path.with_suffix('.ai')
    
    # PNG: High-res, crisp edges
    plt.savefig(png_path, dpi=DPI, bbox_inches='tight', 
               facecolor=FIGURE_FACECOLOR, edgecolor='none')
    
    # TIFF: High-res, lossless compression
    plt.savefig(tiff_path, dpi=DPI, bbox_inches='tight',
               facecolor=FIGURE_FACECOLOR, edgecolor='none',
               format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
    
    # AI: Vector format for Adobe Illustrator
    pdf.FigureCanvas(fig).print_pdf(str(ai_path))
    
    plt.close()
    
    print(f"  ✓ Saved: {png_path.name} + {tiff_path.name} + {ai_path.name}")


def create_multi_panel_scatter(
    data_list: list,
    save_path: str,
    suptitle: Optional[str] = None,
    nrows: int = 1,
    ncols: int = 3
) -> None:
    """
    Create multi-panel scatter plot with consistent subplot styling.
    
    Args:
        data_list: List of dicts, each with:
            - 'x': X data
            - 'y': Y data
            - 'title': Subplot title
            - 'stats': Optional stats dict
        save_path: Output path (without extension)
        suptitle: Optional super title for entire figure
        nrows: Number of rows
        ncols: Number of columns
    """
    # Setup font
    setup_arial_font()
    
    # Calculate figure size (scale by number of panels)
    fig_width = FIGSIZE_SUBPLOT[0] * ncols
    fig_height = FIGSIZE_SUBPLOT[1] * nrows
    
    # Create figure with aesthetic settings
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), dpi=100)
    fig.patch.set_facecolor(FIGURE_FACECOLOR)
    
    # Handle single subplot case
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    # Create each subplot
    for i, (ax, data) in enumerate(zip(axes_flat, data_list)):
        # Format statistics
        stats_text = None
        if 'stats' in data and data['stats']:
            stats = data['stats']
            lines = []
            if 'rho' in stats or 'spearman_rho' in stats:
                rho = stats.get('rho', stats.get('spearman_rho', 0))
                lines.append(f"ρ = {rho:.3f}")
            if 'p_value' in stats:
                lines.append(f"p = {stats['p_value']:.4f}")
            if 'mae' in stats:
                lines.append(f"MAE = {stats['mae']:.2f}")
            if 'r2' in stats:
                lines.append(f"R² = {stats['r2']:.3f}")
            if lines:
                stats_text = '\n'.join(lines)
        
        # Create subplot
        create_standardized_scatter(
            ax,
            data['x'],
            data['y'],
            data['title'],
            xlabel=data.get('xlabel', 'Observed'),
            ylabel=data.get('ylabel', 'Predicted'),
            stats_text=stats_text,
            is_subplot=True
        )
    
    # Hide extra subplots if any
    for i in range(len(data_list), len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    # Super title (if provided)
    if suptitle:
        fig.suptitle(suptitle, fontsize=18, fontweight='bold', 
                    y=0.98, color=COLOR_TICK)
    
    # Adjust spacing - minimal whitespace, clean look
    plt.tight_layout(pad=TIGHT_LAYOUT_PAD, w_pad=2.5, h_pad=2.0)
    
    # Save in multiple formats with optimal settings
    save_path = Path(save_path)
    png_path = save_path.with_suffix('.png')
    tiff_path = save_path.with_suffix('.tiff')
    ai_path = save_path.with_suffix('.ai')
    
    # PNG: High-res, crisp edges
    plt.savefig(png_path, dpi=DPI, bbox_inches='tight',
               facecolor=FIGURE_FACECOLOR, edgecolor='none')
    
    # TIFF: High-res, lossless compression
    plt.savefig(tiff_path, dpi=DPI, bbox_inches='tight',
               facecolor=FIGURE_FACECOLOR, edgecolor='none',
               format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
    
    # AI: Vector format for Adobe Illustrator
    pdf.FigureCanvas(fig).print_pdf(str(ai_path))
    
    plt.close()
    
    print(f"  ✓ Saved: {png_path.name} + {tiff_path.name} + {ai_path.name}")


# ============================================================================
# VALIDATION
# ============================================================================

def validate_consistency():
    """
    Validate that all styling parameters are consistent.
    Print summary for documentation.
    """
    print("\n" + "="*80)
    print("STANDARDIZED SCATTER PLOT STYLING")
    print("="*80)
    print(f"\nColors:")
    print(f"  Dots:              {COLOR_DOTS}")
    print(f"  Line:              {COLOR_LINE}")
    print(f"\nScatter Properties:")
    print(f"  Alpha:             {SCATTER_ALPHA}")
    print(f"  Size:              {SCATTER_SIZE}")
    print(f"  Edge width:        {SCATTER_LINEWIDTH}")
    print(f"\nLine Properties:")
    print(f"  Alpha:             {LINE_ALPHA}")
    print(f"  Width:             {LINE_WIDTH}")
    print(f"\nFont Sizes:")
    print(f"  Labels:            {FONTSIZE_LABEL}")
    print(f"  Title:             {FONTSIZE_TITLE}")
    print(f"  Stats (single):    {FONTSIZE_STATS}")
    print(f"  Stats (subplot):   {FONTSIZE_STATS_SUBPLOT}")
    print(f"  Ticks:             {TICK_LABELSIZE}")
    print(f"\nFigure Sizes:")
    print(f"  Single plot:       {FIGSIZE_SINGLE} inches")
    print(f"  Subplot:           {FIGSIZE_SUBPLOT} inches")
    print(f"  DPI:               {DPI}")
    print(f"\nSpines & Ticks:")
    print(f"  Spine width:       {SPINE_WIDTH}")
    print(f"  Tick length:       {TICK_LENGTH_MAJOR}")
    print(f"  Tick width:        {TICK_WIDTH_MAJOR}")
    print(f"  Top/right spines:  HIDDEN")
    print(f"  Minor ticks:       DISABLED")
    print(f"\nExport:")
    print(f"  Formats:           PNG + AI")
    print(f"  Font:              Arial")
    print(f"  Stats box:         NO BOUNDING BOX")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Validate and print styling
    validate_consistency()
    
    # Example usage
    print("Example usage:")
    print("""
    from plot_styles import create_single_scatter_plot
    
    # Single plot
    create_single_scatter_plot(
        x=observed,
        y=predicted,
        title='Dataset Name',
        save_path='output/plot',
        stats_dict={'rho': 0.75, 'p_value': 0.001, 'mae': 2.5}
    )
    
    # Multi-panel
    from plot_styles import create_multi_panel_scatter
    
    data_list = [
        {'x': x1, 'y': y1, 'title': 'Panel 1', 'stats': {'rho': 0.8, 'p_value': 0.001}},
        {'x': x2, 'y': y2, 'title': 'Panel 2', 'stats': {'rho': 0.7, 'p_value': 0.01}},
        {'x': x3, 'y': y3, 'title': 'Panel 3', 'stats': {'rho': 0.6, 'p_value': 0.05}}
    ]
    
    create_multi_panel_scatter(
        data_list=data_list,
        save_path='output/combined',
        nrows=1,
        ncols=3
    )
    """)

