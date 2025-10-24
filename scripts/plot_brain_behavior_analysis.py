#!/usr/bin/env python3
"""
Plot brain-behavior correlation analysis results.

This script creates visualizations for brain-behavior correlation analysis
including correlation plots, PCA plots, and FDR correction results.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / 'utils'))

from plotting_utils import setup_fonts, save_figure

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def plot_correlation_matrix(correlation_data: pd.DataFrame,
                           output_path: str,
                           title: str = "Brain-Behavior Correlations",
                           figsize: tuple = (12, 8)) -> None:
    """
    Create correlation matrix heatmap.
    
    Args:
        correlation_data (pd.DataFrame): DataFrame with correlation results
        output_path (str): Output path for the plot
        title (str): Plot title
        figsize (tuple): Figure size
    """
    setup_fonts()
    
    # Prepare correlation matrix
    behaviors = correlation_data['behavior'].unique()
    pca_components = correlation_data['pca_component'].unique()
    
    # Create matrix
    corr_matrix = np.zeros((len(behaviors), len(pca_components)))
    pval_matrix = np.ones((len(behaviors), len(pca_components)))
    
    for _, row in correlation_data.iterrows():
        beh_idx = list(behaviors).index(row['behavior'])
        comp_idx = list(pca_components).index(row['pca_component'])
        corr_matrix[beh_idx, comp_idx] = row['correlation']
        pval_matrix[beh_idx, comp_idx] = row['p_value']
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Correlation heatmap
    sns.heatmap(corr_matrix, 
                xticklabels=[f'PC{i+1}' for i in range(len(pca_components))],
                yticklabels=behaviors,
                annot=True, 
                fmt='.3f',
                cmap='RdBu_r',
                center=0,
                ax=ax1,
                cbar_kws={'label': 'Correlation (r)'})
    ax1.set_title('Correlation Coefficients')
    ax1.set_xlabel('PCA Components')
    ax1.set_ylabel('Behavioral Measures')
    
    # P-value heatmap
    sns.heatmap(pval_matrix, 
                xticklabels=[f'PC{i+1}' for i in range(len(pca_components))],
                yticklabels=behaviors,
                annot=True, 
                fmt='.3f',
                cmap='viridis_r',
                ax=ax2,
                cbar_kws={'label': 'P-value'})
    ax2.set_title('P-values')
    ax2.set_xlabel('PCA Components')
    ax2.set_ylabel('Behavioral Measures')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_figure(plt, output_path)
    plt.close()


def plot_pca_explained_variance(pca_results: Dict,
                               output_path: str,
                               title: str = "PCA Explained Variance",
                               n_components: int = 10) -> None:
    """
    Plot PCA explained variance.
    
    Args:
        pca_results (Dict): PCA results dictionary
        output_path (str): Output path for the plot
        title (str): Plot title
        n_components (int): Number of components to plot
    """
    setup_fonts()
    
    explained_variance_ratio = pca_results.get('explained_variance_ratio', [])
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    plt.figure(figsize=(10, 6))
    
    # Plot explained variance
    plt.subplot(1, 2, 1)
    plt.bar(range(1, min(n_components + 1, len(explained_variance_ratio) + 1)), 
            explained_variance_ratio[:n_components])
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Individual Component Variance')
    plt.grid(True, alpha=0.3)
    
    # Plot cumulative variance
    plt.subplot(1, 2, 2)
    plt.plot(range(1, min(n_components + 1, len(cumulative_variance) + 1)), 
             cumulative_variance[:n_components], 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Variance')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_figure(plt, output_path)
    plt.close()


def plot_fdr_correction_results(correlation_data: pd.DataFrame,
                               output_path: str,
                               title: str = "FDR Correction Results") -> None:
    """
    Plot FDR correction results.
    
    Args:
        correlation_data (pd.DataFrame): DataFrame with FDR correction results
        output_path (str): Output path for the plot
        title (str): Plot title
    """
    setup_fonts()
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Before vs After FDR correction
    ax1.scatter(correlation_data['p_value'], correlation_data['fdr_corrected_p'], 
               alpha=0.6, s=50)
    ax1.plot([0, 1], [0, 1], 'r--', alpha=0.8)
    ax1.set_xlabel('Original P-value')
    ax1.set_ylabel('FDR Corrected P-value')
    ax1.set_title('P-values: Before vs After FDR Correction')
    ax1.grid(True, alpha=0.3)
    
    # Significance comparison
    significant_before = (correlation_data['p_value'] < 0.05).sum()
    significant_after = (correlation_data['fdr_corrected_p'] < 0.05).sum()
    
    categories = ['Before FDR', 'After FDR']
    counts = [significant_before, significant_after]
    colors = ['lightcoral', 'lightblue']
    
    ax2.bar(categories, counts, color=colors, alpha=0.7)
    ax2.set_ylabel('Number of Significant Correlations')
    ax2.set_title('Significant Correlations (p < 0.05)')
    ax2.grid(True, alpha=0.3)
    
    # Add count labels on bars
    for i, count in enumerate(counts):
        ax2.text(i, count + 0.5, str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_figure(plt, output_path)
    plt.close()


def plot_behavioral_distributions(behavioral_data: pd.DataFrame,
                                 output_path: str,
                                 title: str = "Behavioral Measures Distribution") -> None:
    """
    Plot distributions of behavioral measures.
    
    Args:
        behavioral_data (pd.DataFrame): DataFrame with behavioral data
        output_path (str): Output path for the plot
        title (str): Plot title
    """
    setup_fonts()
    
    # Get behavioral columns (exclude metadata)
    behavioral_cols = [col for col in behavioral_data.columns 
                      if col not in ['subject_id', 'site', 'age', 'sex', 'iq']]
    
    n_cols = min(3, len(behavioral_cols))
    n_rows = (len(behavioral_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(behavioral_cols):
        if i < len(axes):
            axes[i].hist(behavioral_data[col].dropna(), bins=20, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{col}', fontweight='bold')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(behavioral_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_figure(plt, output_path)
    plt.close()


def create_all_brain_behavior_plots(results_file: str, output_dir: str) -> None:
    """
    Create all brain-behavior analysis plots from results file.
    
    Args:
        results_file (str): Path to brain-behavior analysis results JSON file
        output_dir (str): Output directory for plots
    """
    import json
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Create plots for each dataset
    for dataset_name, dataset_results in results.items():
        if 'correlation_results' in dataset_results:
            corr_data = pd.DataFrame(dataset_results['correlation_results'])
            
            # Correlation matrix
            corr_path = os.path.join(output_dir, f"{dataset_name}_correlation_matrix.png")
            plot_correlation_matrix(corr_data, corr_path, f"Brain-Behavior Correlations - {dataset_name}")
            
            # FDR correction results
            if 'fdr_corrected_p' in corr_data.columns:
                fdr_path = os.path.join(output_dir, f"{dataset_name}_fdr_correction.png")
                plot_fdr_correction_results(corr_data, fdr_path, f"FDR Correction - {dataset_name}")
        
        # PCA results
        if 'pca_results' in dataset_results:
            pca_path = os.path.join(output_dir, f"{dataset_name}_pca_variance.png")
            plot_pca_explained_variance(dataset_results['pca_results'], pca_path, 
                                      f"PCA Explained Variance - {dataset_name}")
        
        # Behavioral distributions
        if 'behavioral_data' in dataset_results:
            # Convert to DataFrame
            beh_data = pd.DataFrame(dataset_results['behavioral_data'])
            beh_path = os.path.join(output_dir, f"{dataset_name}_behavioral_distributions.png")
            plot_behavioral_distributions(beh_data, beh_path, f"Behavioral Distributions - {dataset_name}")
    
    logging.info(f"Created brain-behavior analysis plots in {output_dir}")


def main():
    """Main function for brain-behavior analysis plotting."""
    parser = argparse.ArgumentParser(
        description="Create brain-behavior correlation analysis plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create brain-behavior analysis plots
  python plot_brain_behavior_analysis.py \\
    --results_file results/brain_behavior_results.json \\
    --output_dir results/figures/brain_behavior_analysis
  
  # Create plots in custom directory
  python plot_brain_behavior_analysis.py \\
    --results_file results/brain_behavior_results.json \\
    --output_dir custom_behavior_plots/
        """
    )
    
    parser.add_argument("--results_file", type=str, required=True,
                       help="Path to brain-behavior analysis results JSON file (generated by comprehensive_brain_behavior_analysis.py)")
    parser.add_argument("--output_dir", type=str, default="results/figures/brain_behavior_analysis",
                       help="Output directory for plots (default: results/figures/brain_behavior_analysis)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        logging.error(f"Results file not found: {args.results_file}")
        return
    
    create_all_brain_behavior_plots(args.results_file, args.output_dir)
    logging.info("Brain-behavior analysis plotting completed!")


if __name__ == "__main__":
    main()
