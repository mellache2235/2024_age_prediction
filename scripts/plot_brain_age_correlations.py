#!/usr/bin/env python3
"""
Plot brain age prediction correlations and scatter plots.

This script creates visualizations for brain age prediction results including
scatter plots, correlation plots, and brain age gap distributions.
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


def plot_age_prediction_scatter(true_ages: np.ndarray, 
                               predicted_ages: np.ndarray,
                               dataset_name: str,
                               output_path: str,
                               title: str = None) -> None:
    """
    Create scatter plot of true vs predicted ages.
    
    Args:
        true_ages (np.ndarray): True ages
        predicted_ages (np.ndarray): Predicted ages
        dataset_name (str): Name of the dataset
        output_path (str): Output path for the plot
        title (str): Plot title
    """
    setup_fonts()
    
    # Calculate metrics
    from scipy.stats import pearsonr
    r, p = pearsonr(true_ages, predicted_ages)
    r_squared = r ** 2
    mae = np.mean(np.abs(true_ages - predicted_ages))
    
    # Format p-value
    if p < 0.001:
        p_text = "P < 0.001"
    else:
        p_text = f"P = {p:.3f}"
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.scatter(true_ages, predicted_ages, alpha=0.6, s=50)
    
    # Add diagonal line
    min_age = min(true_ages.min(), predicted_ages.min())
    max_age = max(true_ages.max(), predicted_ages.max())
    plt.plot([min_age, max_age], [min_age, max_age], 'r--', alpha=0.8, linewidth=2)
    
    # Add metrics text
    plt.text(0.05, 0.95, f'R² = {r_squared:.3f}\n{p_text}\nMAE = {mae:.2f} years', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xlabel('True Age (years)')
    plt.ylabel('Predicted Age (years)')
    plt.title(title or f'Brain Age Prediction - {dataset_name}')
    plt.grid(True, alpha=0.3)
    
    save_figure(plt, output_path)
    plt.close()


def plot_brain_age_gap_distribution(bag_values: np.ndarray,
                                   dataset_name: str,
                                   output_path: str,
                                   title: str = None) -> None:
    """
    Create distribution plot of brain age gaps.
    
    Args:
        bag_values (np.ndarray): Brain age gap values
        dataset_name (str): Name of the dataset
        output_path (str): Output path for the plot
        title (str): Plot title
    """
    setup_fonts()
    
    plt.figure(figsize=(10, 6))
    
    # Create histogram
    plt.hist(bag_values, bins=30, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add vertical line at mean
    mean_bag = np.mean(bag_values)
    plt.axvline(mean_bag, color='red', linestyle='--', linewidth=2, 
                label=f'Mean BAG = {mean_bag:.2f} years')
    
    # Add statistics text
    std_bag = np.std(bag_values)
    median_bag = np.median(bag_values)
    plt.text(0.05, 0.95, f'Mean: {mean_bag:.2f} ± {std_bag:.2f} years\n'
                         f'Median: {median_bag:.2f} years\n'
                         f'N: {len(bag_values)}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xlabel('Brain Age Gap (years)')
    plt.ylabel('Frequency')
    plt.title(title or f'Brain Age Gap Distribution - {dataset_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_figure(plt, output_path)
    plt.close()


def plot_group_comparison_bag(group1_bag: np.ndarray,
                             group2_bag: np.ndarray,
                             group1_name: str,
                             group2_name: str,
                             output_path: str,
                             title: str = None) -> None:
    """
    Create box plot comparing brain age gaps between groups.
    
    Args:
        group1_bag (np.ndarray): Brain age gaps for group 1
        group2_bag (np.ndarray): Brain age gaps for group 2
        group1_name (str): Name of group 1
        group2_name (str): Name of group 2
        output_path (str): Output path for the plot
        title (str): Plot title
    """
    setup_fonts()
    
    # Prepare data for plotting
    data = []
    labels = []
    
    data.extend(group1_bag)
    labels.extend([group1_name] * len(group1_bag))
    
    data.extend(group2_bag)
    labels.extend([group2_name] * len(group2_bag))
    
    df = pd.DataFrame({'BAG': data, 'Group': labels})
    
    # Create plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='Group', y='BAG')
    sns.stripplot(data=df, x='Group', y='BAG', size=4, alpha=0.6, color='red')
    
    # Add statistics
    from scipy.stats import ttest_ind
    t_stat, p_val = ttest_ind(group1_bag, group2_bag)
    
    plt.text(0.5, 0.95, f'T-test: t={t_stat:.3f}, p={p_val:.4f}', 
             transform=plt.gca().transAxes, horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.ylabel('Brain Age Gap (years)')
    plt.title(title or f'Brain Age Gap Comparison: {group1_name} vs {group2_name}')
    plt.grid(True, alpha=0.3)
    
    save_figure(plt, output_path)
    plt.close()


def create_all_brain_age_plots(results_file: str, output_dir: str) -> None:
    """
    Create all brain age prediction plots from results file.
    
    Args:
        results_file (str): Path to results JSON file
        output_dir (str): Output directory for plots
    """
    import json
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Get external testing results
    external_results = results.get('external_testing', {})
    comprehensive_analysis = results.get('comprehensive_analysis', {})
    
    # Create individual dataset plots
    for dataset_name, dataset_results in external_results.items():
        if 'true_ages' in dataset_results and 'predictions' in dataset_results:
            true_ages = np.array(dataset_results['true_ages'])
            predictions = np.array(dataset_results['predictions'])
            bag_values = predictions - true_ages
            
            # Scatter plot
            scatter_path = os.path.join(output_dir, f"{dataset_name}_age_scatter.png")
            plot_age_prediction_scatter(true_ages, predictions, dataset_name, scatter_path)
            
            # BAG distribution
            bag_path = os.path.join(output_dir, f"{dataset_name}_bag_distribution.png")
            plot_brain_age_gap_distribution(bag_values, dataset_name, bag_path)
    
    # Create group comparison plots
    group_comparisons = comprehensive_analysis.get('group_comparisons', {})
    individual_metrics = comprehensive_analysis.get('individual_metrics', {})
    
    for comparison_name, comparison in group_comparisons.items():
        group1_name = comparison['group1_name']
        group2_name = comparison['group2_name']
        
        # Get BAG values for each group
        group1_bags = []
        group2_bags = []
        
        # Find datasets for each group
        for dataset_name, metrics in individual_metrics.items():
            if group1_name.lower() in dataset_name.lower():
                # Get BAG values from external results
                if dataset_name in external_results:
                    true_ages = np.array(external_results[dataset_name]['true_ages'])
                    predictions = np.array(external_results[dataset_name]['predictions'])
                    bag_values = predictions - true_ages
                    group1_bags.extend(bag_values)
            elif group2_name.lower() in dataset_name.lower():
                if dataset_name in external_results:
                    true_ages = np.array(external_results[dataset_name]['true_ages'])
                    predictions = np.array(external_results[dataset_name]['predictions'])
                    bag_values = predictions - true_ages
                    group2_bags.extend(bag_values)
        
        if group1_bags and group2_bags:
            comparison_path = os.path.join(output_dir, f"{comparison_name}_bag_comparison.png")
            plot_group_comparison_bag(np.array(group1_bags), np.array(group2_bags),
                                    group1_name, group2_name, comparison_path)
    
    logging.info(f"Created brain age correlation plots in {output_dir}")


def main():
    """Main function for brain age correlation plotting."""
    parser = argparse.ArgumentParser(
        description="Create brain age correlation plots and statistical comparisons",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create all brain age correlation plots
  python plot_brain_age_correlations.py \\
    --results_file results/brain_age_prediction_results.json \\
    --output_dir results/figures/brain_age_correlations
  
  # Create plots in custom directory
  python plot_brain_age_correlations.py \\
    --results_file results/brain_age_prediction_results.json \\
    --output_dir custom_plots/
        """
    )
    
    parser.add_argument("--results_file", type=str, required=True,
                       help="Path to brain age prediction results JSON file (generated by brain_age_prediction.py)")
    parser.add_argument("--output_dir", type=str, default="results/figures/brain_age_correlations",
                       help="Output directory for plots (default: results/figures/brain_age_correlations)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        logging.error(f"Results file not found: {args.results_file}")
        return
    
    create_all_brain_age_plots(args.results_file, args.output_dir)
    logging.info("Brain age correlation plotting completed!")


if __name__ == "__main__":
    main()
