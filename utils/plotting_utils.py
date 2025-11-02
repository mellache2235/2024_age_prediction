"""
Plotting and visualization utilities for age prediction analysis.

This module provides standardized plotting functions for creating publication-ready figures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import warnings

try:
    from scripts.plot_styles import (
        create_standardized_scatter,
        setup_arial_font,
        FIGSIZE_SINGLE,
        FIGURE_FACECOLOR,
        DPI
    )
except ImportError:
    # Allow plotting_utils to operate even if plot_styles is unavailable
    create_standardized_scatter = None
    setup_arial_font = None
    FIGSIZE_SINGLE = (6, 6)
    FIGURE_FACECOLOR = 'white'
    DPI = 300

warnings.filterwarnings("ignore", category=FutureWarning)

# Set up consistent plotting style
plt.style.use('default')
sns.set_style("white")

# Default color palette
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e', 
    'tertiary': '#2ca02c',
    'quaternary': '#d62728',
    'navy': '#1e3a8a',
    'red': '#dc2626',
    'gray': '#6b7280'
}

# Default figure parameters
FIGURE_PARAMS = {
    'dpi': 300,
    'figsize': (8, 6),
    'facecolor': 'white',
    'edgecolor': 'none'
}


def setup_fonts(font_path: Optional[str] = None) -> None:
    """
    Setup consistent fonts for plotting.
    
    Args:
        font_path (str, optional): Path to custom font file
    """
    # Prefer centralized Arial setup if available
    if setup_arial_font and setup_arial_font():
        return

    if font_path and Path(font_path).exists():
        font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = prop.get_name()
    else:
        # Use default system fonts
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']


def format_p_value(p_value: float) -> str:
    """
    Format p-value for display in plots.
    
    Args:
        p_value (float): P-value to format
        
    Returns:
        str: Formatted p-value string
    """
    if p_value < 0.001:
        return r"$\mathit{P} < 0.001$"
    else:
        return rf"$\mathit{{P}} = {p_value:.3f}$"


def plot_age_prediction(actual_ages: np.ndarray, predicted_ages: np.ndarray,
                       title: str = "Age Prediction", 
                       xlabel: str = "Chronological Age",
                       ylabel: str = "Brain Age",
                       save_path: Optional[str] = None,
                       show_identity_line: bool = True,
                       color: str = 'navy',
                       alpha: float = 0.6,
                       figsize: Tuple[int, int] = (6, 6)) -> plt.Figure:
    """
    Create a scatter plot of actual vs predicted ages with regression line.
    
    Args:
        actual_ages (np.ndarray): Actual age values
        predicted_ages (np.ndarray): Predicted age values
        title (str): Plot title
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
        save_path (str, optional): Path to save the figure
        show_identity_line (bool): Whether to show identity line
        color (str): Color for scatter points
        alpha (float): Transparency of scatter points
        figsize (Tuple[int, int]): Figure size
        
    Returns:
        plt.Figure: The created figure
    """
    # Calculate statistics
    r, p = stats.pearsonr(actual_ages, predicted_ages)
    mae = mean_absolute_error(actual_ages, predicted_ages)
    r_squared = r ** 2

    # Use centralized styling when available
    if create_standardized_scatter:
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE, dpi=DPI)
        fig.patch.set_facecolor(FIGURE_FACECOLOR)

        p_str = "< 0.001" if p < 0.001 else f"= {p:.3f}"
        stats_text = (
            f"R² = {r_squared:.3f}\n"
            f"MAE = {mae:.2f} years\n"
            f"P {p_str}"
        )

        create_standardized_scatter(
            ax,
            actual_ages,
            predicted_ages,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            stats_text=stats_text,
            is_subplot=False
        )

        # Apply consistent tick spacing (5-year increments)
        try:
            age_min = np.floor(min(actual_ages.min(), predicted_ages.min()) / 5) * 5
            age_max = np.ceil(max(actual_ages.max(), predicted_ages.max()) / 5) * 5
            ax.set_xlim(age_min, age_max)
            ax.set_ylim(age_min, age_max)
            locator = MultipleLocator(5)
            ax.xaxis.set_major_locator(locator)
            ax.yaxis.set_major_locator(locator)
        except Exception:
            pass

        plt.tight_layout(pad=1.5)

        if save_path:
            fig.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor=FIGURE_FACECOLOR, edgecolor='none')

        return fig

    # Fallback to original styling if centralized scatter unavailable
    fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_PARAMS['dpi'])
    sns.regplot(
        x=actual_ages,
        y=predicted_ages,
        ci=None,
        scatter_kws={'color': color, 'alpha': alpha, 's': 40, 'edgecolor': 'w', 'linewidth': 0.5},
        line_kws={'color': 'red', 'linewidth': 2},
        ax=ax
    )

    age_min, age_max = actual_ages.min(), actual_ages.max()
    lims = [age_min - 1, age_max + 2]
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    if show_identity_line:
        ax.plot(lims, lims, linestyle='--', color='gray', linewidth=1.2, label='Identity line')

    p_text = format_p_value(p)
    ax.text(
        0.95,
        0.05,
        f"$\\mathit{{R}}^2$ = {r_squared:.3f}\n{p_text}\n$\\mathrm{{MAE}} = {mae:.2f}\\;\\mathrm{{years}}$",
        transform=ax.transAxes,
        horizontalalignment='right',
        verticalalignment='bottom',
        fontsize=11,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    ax.set_xlabel(xlabel, fontsize=15, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=15, labelpad=10)
    ax.set_title(title, fontsize=15, pad=10)
    ax.spines[['right', 'top']].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(1.5)

    ax.tick_params(axis='both', which='major', length=6, width=1)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.tight_layout(pad=1.2)

    if save_path:
        plt.savefig(save_path, dpi=FIGURE_PARAMS['dpi'], bbox_inches='tight')

    return fig


def plot_brain_age_gap(actual_ages: np.ndarray, predicted_ages: np.ndarray,
                      title: str = "Brain Age Gap",
                      xlabel: str = "Chronological Age",
                      ylabel: str = "Brain Age Gap (years)",
                      save_path: Optional[str] = None,
                      color: str = 'navy',
                      alpha: float = 0.6,
                      figsize: Tuple[int, int] = (6, 6)) -> plt.Figure:
    """
    Create a plot of brain age gap (predicted - actual) vs chronological age.
    
    Args:
        actual_ages (np.ndarray): Actual age values
        predicted_ages (np.ndarray): Predicted age values
        title (str): Plot title
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
        save_path (str, optional): Path to save the figure
        color (str): Color for scatter points
        alpha (float): Transparency of scatter points
        figsize (Tuple[int, int]): Figure size
        
    Returns:
        plt.Figure: The created figure
    """
    # Calculate brain age gap
    brain_age_gap = predicted_ages - actual_ages
    
    # Calculate statistics
    r, p = stats.pearsonr(actual_ages, brain_age_gap)
    mean_gap = np.mean(brain_age_gap)
    std_gap = np.std(brain_age_gap)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_PARAMS['dpi'])
    
    # Create scatter plot
    ax.scatter(actual_ages, brain_age_gap, color=color, alpha=alpha, s=40,
              edgecolor='w', linewidth=0.5)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.2)
    
    # Add statistics text
    p_text = format_p_value(p)
    ax.text(0.95, 0.95,
            f"$\mathit{{r}} = {r:.3f}$\n"
            f"{p_text}\n"
            f"$\mathrm{{Mean\;Gap}} = {mean_gap:.2f} \pm {std_gap:.2f}\;\mathrm{{years}}$",
            transform=ax.transAxes,
            horizontalalignment='right',
            verticalalignment='top',
            fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Formatting
    ax.set_xlabel(xlabel, fontsize=15, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=15, labelpad=10)
    ax.set_title(title, fontsize=15, pad=10)
    
    # Remove top and right spines
    ax.spines[['right', 'top']].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(1.5)
    
    # Set tick parameters
    ax.tick_params(axis='both', which='major', length=6, width=1)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    plt.tight_layout(pad=1.2)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_PARAMS['dpi'], bbox_inches='tight')
    
    return fig


def plot_feature_importance(features: np.ndarray, feature_names: List[str],
                          title: str = "Feature Importance",
                          top_n: int = 20,
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Create a horizontal bar plot of feature importance.
    
    Args:
        features (np.ndarray): Feature importance values
        feature_names (List[str]): Names of features
        title (str): Plot title
        top_n (int): Number of top features to display
        save_path (str, optional): Path to save the figure
        figsize (Tuple[int, int]): Figure size
        
    Returns:
        plt.Figure: The created figure
    """
    # Sort features by importance
    sorted_indices = np.argsort(features)[::-1]
    top_features = features[sorted_indices[:top_n]]
    top_names = [feature_names[i] for i in sorted_indices[:top_n]]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_PARAMS['dpi'])
    
    # Create horizontal bar plot
    y_pos = np.arange(len(top_features))
    bars = ax.barh(y_pos, top_features, color=COLORS['primary'], alpha=0.7)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, top_features)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{value:.3f}', ha='left', va='center', fontsize=10)
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, fontsize=10)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)
    
    # Remove top and right spines
    ax.spines[['right', 'top']].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(1.5)
    
    # Invert y-axis to show highest importance at top
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_PARAMS['dpi'], bbox_inches='tight')
    
    return fig


def plot_network_analysis(network_scores: Dict[str, float],
                         title: str = "Network-level Analysis",
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Create a bar plot of network-level analysis results.
    
    Args:
        network_scores (Dict[str, float]): Dictionary mapping network names to scores
        title (str): Plot title
        save_path (str, optional): Path to save the figure
        figsize (Tuple[int, int]): Figure size
        
    Returns:
        plt.Figure: The created figure
    """
    # Sort networks by score
    sorted_networks = sorted(network_scores.items(), key=lambda x: x[1], reverse=True)
    network_names, scores = zip(*sorted_networks)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_PARAMS['dpi'])
    
    # Create bar plot
    bars = ax.bar(range(len(network_names)), scores, 
                 color=COLORS['primary'], alpha=0.7)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Formatting
    ax.set_xticks(range(len(network_names)))
    ax.set_xticklabels(network_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)
    
    # Remove top and right spines
    ax.spines[['right', 'top']].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(1.5)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_PARAMS['dpi'], bbox_inches='tight')
    
    return fig


def plot_correlation_matrix(correlation_matrix: np.ndarray,
                          feature_names: List[str],
                          title: str = "Correlation Matrix",
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Create a heatmap of correlation matrix.
    
    Args:
        correlation_matrix (np.ndarray): Correlation matrix
        feature_names (List[str]): Names of features
        title (str): Plot title
        save_path (str, optional): Path to save the figure
        figsize (Tuple[int, int]): Figure size
        
    Returns:
        plt.Figure: The created figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_PARAMS['dpi'])
    
    # Create heatmap
    im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation', fontsize=12)
    
    # Set ticks and labels
    ax.set_xticks(range(len(feature_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(feature_names, fontsize=10)
    
    # Add correlation values as text
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title(title, fontsize=14, pad=20)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_PARAMS['dpi'], bbox_inches='tight')
    
    return fig


def create_multi_panel_figure(plots: List[plt.Figure], 
                            layout: Tuple[int, int] = (2, 2),
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a multi-panel figure from multiple subplots.
    
    Args:
        plots (List[plt.Figure]): List of figure objects
        layout (Tuple[int, int]): Layout (rows, columns)
        save_path (str, optional): Path to save the figure
        
    Returns:
        plt.Figure: Combined figure
    """
    rows, cols = layout
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5), 
                            dpi=FIGURE_PARAMS['dpi'])
    
    # Flatten axes for easier indexing
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Add plots to subplots
    for i, plot in enumerate(plots[:len(axes)]):
        # This is a simplified version - in practice, you'd need to extract
        # the plot elements and recreate them in the subplot
        axes[i].text(0.5, 0.5, f'Plot {i+1}', ha='center', va='center',
                    transform=axes[i].transAxes, fontsize=16)
    
    # Hide unused subplots
    for i in range(len(plots), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_PARAMS['dpi'], bbox_inches='tight')
    
    return fig


def save_figure(fig: plt.Figure, save_path: str, 
               formats: List[str] = ['png']) -> None:
    """
    Save figure in multiple formats.
    
    Args:
        fig (plt.Figure): Figure to save
        save_path (str): Base path for saving (without extension)
        formats (List[str]): List of formats to save in
    """
    for fmt in formats:
        full_path = f"{save_path}.{fmt}"
        fig.savefig(full_path, dpi=FIGURE_PARAMS['dpi'], bbox_inches='tight')
        print(f"Figure saved to: {full_path}")


# Initialize fonts on import
setup_fonts()
