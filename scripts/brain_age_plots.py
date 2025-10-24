#!/usr/bin/env python3
"""
Create brain age prediction plots similar to the provided examples.

This script generates scatter plots showing brain age prediction accuracy
with statistical metrics and regression lines.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / 'utils'))

from plotting_utils import (
    plot_age_prediction,
    plot_brain_age_gap,
    setup_fonts,
    save_figure
)
from model_utils import evaluate_model_performance


def create_brain_age_scatter_plot(actual_ages: np.ndarray,
                                predicted_ages: np.ndarray,
                                title: str = "Brain Age Prediction Accuracy",
                                xlabel: str = "Chronological Age",
                                ylabel: str = "Brain Age",
                                dataset_name: str = "Dataset",
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a brain age prediction scatter plot with statistical metrics.
    
    Args:
        actual_ages (np.ndarray): Actual chronological ages
        predicted_ages (np.ndarray): Predicted brain ages
        title (str): Plot title
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
        dataset_name (str): Name of the dataset
        save_path (str, optional): Path to save the figure
        
    Returns:
        plt.Figure: The created scatter plot
    """
    # Calculate performance metrics
    metrics = evaluate_model_performance(actual_ages, predicted_ages)
    
    # Create the plot using the standardized function
    fig = plot_age_prediction(
        actual_ages=actual_ages,
        predicted_ages=predicted_ages,
        title=f"{dataset_name} Brain Age Accuracy",
        xlabel=xlabel,
        ylabel=ylabel,
        save_path=save_path,
        show_identity_line=True,
        color='navy',
        alpha=0.6,
        figsize=(8, 8)
    )
    
    return fig


def create_brain_behavior_scatter_plot(observed_values: np.ndarray,
                                     predicted_values: np.ndarray,
                                     behavior_name: str = "Behavior",
                                     title: str = "Brain-Behavior Analysis",
                                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a brain-behavior correlation scatter plot.
    
    Args:
        observed_values (np.ndarray): Observed behavioral values
        predicted_values (np.ndarray): Predicted behavioral values
        behavior_name (str): Name of the behavioral measure
        title (str): Plot title
        save_path (str, optional): Path to save the figure
        
    Returns:
        plt.Figure: The created scatter plot
    """
    # Calculate correlation
    from scipy.stats import pearsonr
    r, p = pearsonr(observed_values, predicted_values)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    
    # Create scatter plot with regression line
    import seaborn as sns
    sns.regplot(x=observed_values, y=predicted_values, ci=None,
               scatter_kws={'color': 'navy', 'alpha': 0.6, 's': 40, 
                          'edgecolor': 'w', 'linewidth': 0.5},
               line_kws={'color': 'red', 'linewidth': 2}, ax=ax)
    
    # Set axis limits
    obs_min, obs_max = observed_values.min(), observed_values.max()
    pred_min, pred_max = predicted_values.min(), predicted_values.max()
    
    ax.set_xlim(obs_min - 2, obs_max + 2)
    ax.set_ylim(pred_min - 2, pred_max + 2)
    
    # Add statistics text
    if p < 0.001:
        p_text = r"$\mathit{P} < 0.001$"
    else:
        p_text = rf"$\mathit{{P}} = {p:.3f}$"
    
    ax.text(0.95, 0.05,
            f"$\mathit{{R}} = {r:.3f}$\n"
            f"{p_text}",
            transform=ax.transAxes,
            horizontalalignment='right',
            verticalalignment='bottom',
            fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Formatting
    ax.set_xlabel(f"Observed {behavior_name}", fontsize=14, labelpad=10)
    ax.set_ylabel(f"Predicted {behavior_name}", fontsize=14, labelpad=10)
    ax.set_title(f"{title}: {behavior_name}", fontsize=16, pad=20)
    
    # Remove top and right spines
    ax.spines[['right', 'top']].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(1.5)
    
    # Set tick parameters
    ax.tick_params(axis='both', which='major', length=6, width=1)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_figure(fig, save_path, formats=['png', 'pdf'])
    
    return fig


def create_multi_panel_brain_age_plot(actual_ages: np.ndarray,
                                    predicted_ages: np.ndarray,
                                    dataset_names: List[str],
                                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a multi-panel brain age prediction plot.
    
    Args:
        actual_ages (np.ndarray): Actual ages (can be 2D for multiple datasets)
        predicted_ages (np.ndarray): Predicted ages (can be 2D for multiple datasets)
        dataset_names (List[str]): Names of datasets
        save_path (str, optional): Path to save the figure
        
    Returns:
        plt.Figure: Multi-panel figure
    """
    n_datasets = len(dataset_names)
    
    # Determine subplot layout
    if n_datasets == 1:
        fig, axes = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
        axes = [axes]
    elif n_datasets == 2:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=300)
    elif n_datasets <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(16, 16), dpi=300)
        axes = axes.flatten()
    else:
        # For more than 4 datasets, create a grid
        n_cols = 3
        n_rows = (n_datasets + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows), dpi=300)
        axes = axes.flatten()
    
    # Create plots for each dataset
    for i, dataset_name in enumerate(dataset_names):
        if i < len(axes):
            # Extract data for this dataset
            if actual_ages.ndim > 1:
                actual = actual_ages[:, i]
                predicted = predicted_ages[:, i]
            else:
                actual = actual_ages
                predicted = predicted_ages
            
            # Create scatter plot
            ax = axes[i]
            
            # Calculate metrics
            from scipy.stats import pearsonr
            r, p = pearsonr(actual, predicted)
            mae = np.mean(np.abs(predicted - actual))
            r_squared = r ** 2
            
            # Create scatter plot with regression line
            import seaborn as sns
            sns.regplot(x=actual, y=predicted, ci=None,
                       scatter_kws={'color': 'navy', 'alpha': 0.6, 's': 40, 
                                  'edgecolor': 'w', 'linewidth': 0.5},
                       line_kws={'color': 'red', 'linewidth': 2}, ax=ax)
            
            # Add identity line
            lims = [actual.min() - 1, actual.max() + 2]
            ax.plot(lims, lims, linestyle='--', color='gray', 
                   linewidth=1.2, label='Identity line')
            
            # Add statistics text
            if p < 0.001:
                p_text = r"$\mathit{P} < 0.001$"
            else:
                p_text = rf"$\mathit{{P}} = {p:.3f}$"
            
            ax.text(0.95, 0.05,
                    f"$\mathit{{R}}^2$ = {r_squared:.3f}\n"
                    f"{p_text}\n"
                    f"$\mathrm{{MAE}} = {mae:.2f}\;\mathrm{{years}}$",
                    transform=ax.transAxes,
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Formatting
            ax.set_xlabel("Chronological Age", fontsize=12, labelpad=10)
            ax.set_ylabel("Brain Age", fontsize=12, labelpad=10)
            ax.set_title(dataset_name, fontsize=14, pad=15)
            
            # Remove top and right spines
            ax.spines[['right', 'top']].set_visible(False)
            for spine in ['bottom', 'left']:
                ax.spines[spine].set_linewidth(1.5)
            
            # Set tick parameters
            ax.tick_params(axis='both', which='major', length=6, width=1)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
    
    # Hide unused subplots
    for i in range(len(dataset_names), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_figure(fig, save_path, formats=['png', 'pdf'])
    
    return fig


def main():
    """Main function for creating brain age plots."""
    parser = argparse.ArgumentParser(
        description="Create brain age prediction plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create brain age scatter plot
  python brain_age_plots.py \\
    --actual_ages data/actual_ages.npy \\
    --predicted_ages data/predicted_ages.npy \\
    --dataset_name "ABIDE ASD" \\
    --output_dir results/figures

  # Create brain-behavior plot
  python brain_age_plots.py \\
    --observed_values data/hyperactivity_observed.npy \\
    --predicted_values data/hyperactivity_predicted.npy \\
    --behavior_name "Hyperactivity" \\
    --output_dir results/figures
        """
    )
    
    parser.add_argument("--actual_ages", type=str,
                       help="Path to actual ages numpy file")
    parser.add_argument("--predicted_ages", type=str,
                       help="Path to predicted ages numpy file")
    parser.add_argument("--observed_values", type=str,
                       help="Path to observed behavioral values numpy file")
    parser.add_argument("--predicted_values", type=str,
                       help="Path to predicted behavioral values numpy file")
    parser.add_argument("--dataset_name", type=str, default="Dataset",
                       help="Name of the dataset")
    parser.add_argument("--behavior_name", type=str, default="Behavior",
                       help="Name of the behavioral measure")
    parser.add_argument("--output_dir", type=str, default="results/figures",
                       help="Output directory for figures")
    parser.add_argument("--plot_type", type=str, choices=['brain_age', 'brain_behavior', 'both'],
                       default='both', help="Type of plot to create")
    
    args = parser.parse_args()
    
    # Setup fonts
    setup_fonts()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.plot_type in ['brain_age', 'both']:
        if args.actual_ages and args.predicted_ages:
            # Load data
            actual_ages = np.load(args.actual_ages)
            predicted_ages = np.load(args.predicted_ages)
            
            # Create brain age scatter plot
            fig = create_brain_age_scatter_plot(
                actual_ages=actual_ages,
                predicted_ages=predicted_ages,
                dataset_name=args.dataset_name,
                save_path=os.path.join(args.output_dir, f"{args.dataset_name.replace(' ', '_')}_brain_age.png")
            )
            
            print(f"Brain age plot created and saved to: {args.output_dir}")
        else:
            print("Warning: --actual_ages and --predicted_ages required for brain age plots")
    
    if args.plot_type in ['brain_behavior', 'both']:
        if args.observed_values and args.predicted_values:
            # Load data
            observed_values = np.load(args.observed_values)
            predicted_values = np.load(args.predicted_values)
            
            # Create brain-behavior scatter plot
            fig = create_brain_behavior_scatter_plot(
                observed_values=observed_values,
                predicted_values=predicted_values,
                behavior_name=args.behavior_name,
                save_path=os.path.join(args.output_dir, f"{args.behavior_name.replace(' ', '_')}_brain_behavior.png")
            )
            
            print(f"Brain-behavior plot created and saved to: {args.output_dir}")
        else:
            print("Warning: --observed_values and --predicted_values required for brain-behavior plots")


if __name__ == "__main__":
    main()
