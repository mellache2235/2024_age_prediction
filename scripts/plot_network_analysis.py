#!/usr/bin/env python3
"""
Plot network analysis results using Yeo atlas grouping.

This script creates polar bar plots and network-level visualizations
for brain network analysis results.
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


def create_polar_bar_plot(network_data: pd.DataFrame,
                         output_path: str,
                         title: str = "Network Analysis",
                         max_value: Optional[float] = None,
                         show_values: bool = True) -> None:
    """
    Create polar bar plot for network analysis results.
    
    Args:
        network_data (pd.DataFrame): DataFrame with columns ['Network', 'Count']
        output_path (str): Output path for the plot
        title (str): Plot title
        max_value (float, optional): Maximum value for y-axis scaling
    """
    setup_fonts()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # Prepare data - handle different column name formats
    if 'Network' in network_data.columns and 'Count' in network_data.columns:
        networks = network_data['Network'].values
        counts = network_data['Count'].values
    elif 'network' in network_data.columns and 'mean_attribution' in network_data.columns:
        networks = network_data['network'].values
        counts = network_data['mean_attribution'].values
    elif 'network' in network_data.columns and 'total_attribution' in network_data.columns:
        networks = network_data['network'].values
        counts = network_data['total_attribution'].values
    else:
        raise ValueError(f"Expected columns 'Network'/'network' and 'Count'/'mean_attribution'/'total_attribution' not found. Available columns: {list(network_data.columns)}")
    
    # Set up angles
    angles = np.linspace(0, 2 * np.pi, len(networks), endpoint=False)
    
    # Create bars with distinct colors for each network
    colors = plt.cm.Set3(np.linspace(0, 1, len(networks)))
    bars = ax.bar(angles, counts, width=0.8, alpha=0.7, 
                  color=colors, edgecolor='white', linewidth=1)
    
    # Customize plot
    ax.set_xticks(angles)
    ax.set_xticklabels(networks, fontsize=10, fontweight='bold', color='black')
    ax.set_ylim(0, max_value or counts.max() * 1.1)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Remove y-axis labels and ticks
    ax.set_yticklabels([])
    ax.tick_params(axis='y', which='major', length=0)
    
    # Add subtle grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add value labels on bars (optional, can be removed for cleaner look)
    if show_values:
        for angle, count, bar in zip(angles, counts, bars):
            if count > 0:
                ax.text(angle, count + max_value * 0.02, f'{count:.1f}', 
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Set background color
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Save plot
    save_figure(plt, output_path)
    plt.close()


def create_network_comparison_plot(network_data_dict: Dict[str, pd.DataFrame],
                                  output_path: str,
                                  title: str = "Network Comparison") -> None:
    """
    Create comparison plot for multiple datasets' network analysis.
    
    Args:
        network_data_dict (Dict[str, pd.DataFrame]): Dictionary of network data
        output_path (str): Output path for the plot
        title (str): Plot title
    """
    setup_fonts()
    
    # Prepare data for comparison - handle different column name formats
    all_networks = set()
    for data in network_data_dict.values():
        if 'Network' in data.columns:
            all_networks.update(data['Network'].values)
        elif 'network' in data.columns:
            all_networks.update(data['network'].values)
        else:
            raise ValueError(f"Expected 'Network' or 'network' column not found. Available columns: {list(data.columns)}")
    
    all_networks = sorted(list(all_networks))
    
    # Create comparison DataFrame
    comparison_data = []
    for dataset_name, data in network_data_dict.items():
        for network in all_networks:
            # Handle different column name formats
            if 'Network' in data.columns and 'Count' in data.columns:
                count = data[data['Network'] == network]['Count'].values
            elif 'network' in data.columns and 'mean_attribution' in data.columns:
                count = data[data['network'] == network]['mean_attribution'].values
            elif 'network' in data.columns and 'total_attribution' in data.columns:
                count = data[data['network'] == network]['total_attribution'].values
            else:
                count = [0]  # Default to 0 if no matching data
            
            count = count[0] if len(count) > 0 else 0
            comparison_data.append({
                'Dataset': dataset_name,
                'Network': network,
                'Count': count
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create plot
    plt.figure(figsize=(14, 8))
    
    # Create grouped bar plot
    x = np.arange(len(all_networks))
    width = 0.8 / len(network_data_dict)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(network_data_dict)))
    
    for i, (dataset_name, color) in enumerate(zip(network_data_dict.keys(), colors)):
        dataset_counts = [comparison_df[(comparison_df['Dataset'] == dataset_name) & 
                                      (comparison_df['Network'] == network)]['Count'].values[0] 
                         for network in all_networks]
        
        plt.bar(x + i * width, dataset_counts, width, label=dataset_name, 
                color=color, alpha=0.8)
    
    plt.xlabel('Brain Networks', fontsize=12, fontweight='bold')
    plt.ylabel('Feature Count', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(x + width * (len(network_data_dict) - 1) / 2, all_networks, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_figure(plt, output_path)
    plt.close()


def create_network_heatmap(network_data_dict: Dict[str, pd.DataFrame],
                          output_path: str,
                          title: str = "Network Analysis Heatmap") -> None:
    """
    Create heatmap for network analysis results.
    
    Args:
        network_data_dict (Dict[str, pd.DataFrame]): Dictionary of network data
        output_path (str): Output path for the plot
        title (str): Plot title
    """
    setup_fonts()
    
    # Prepare data for heatmap - handle different column name formats
    all_networks = set()
    for data in network_data_dict.values():
        if 'Network' in data.columns:
            all_networks.update(data['Network'].values)
        elif 'network' in data.columns:
            all_networks.update(data['network'].values)
        else:
            raise ValueError(f"Expected 'Network' or 'network' column not found. Available columns: {list(data.columns)}")
    
    all_networks = sorted(list(all_networks))
    
    # Create matrix
    matrix_data = []
    for dataset_name, data in network_data_dict.items():
        row = []
        for network in all_networks:
            # Handle different column name formats
            if 'Network' in data.columns and 'Count' in data.columns:
                count = data[data['Network'] == network]['Count'].values
            elif 'network' in data.columns and 'mean_attribution' in data.columns:
                count = data[data['network'] == network]['mean_attribution'].values
            elif 'network' in data.columns and 'total_attribution' in data.columns:
                count = data[data['network'] == network]['total_attribution'].values
            else:
                count = [0]  # Default to 0 if no matching data
            
            count = count[0] if len(count) > 0 else 0
            row.append(count)
        matrix_data.append(row)
    
    matrix = np.array(matrix_data)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix, 
                xticklabels=all_networks,
                yticklabels=list(network_data_dict.keys()),
                annot=True, 
                fmt='.1f',
                cmap='viridis',
                cbar_kws={'label': 'Feature Count'})
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Brain Networks', fontsize=12, fontweight='bold')
    plt.ylabel('Datasets', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    save_figure(plt, output_path)
    plt.close()


def create_all_network_plots(config: Dict, output_dir: str = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/figures/network_analysis") -> None:
    """
    Create all network analysis plots.
    
    Args:
        config (Dict): Configuration dictionary
        output_dir (str): Output directory for plots
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get network analysis results (assuming they exist from network_analysis_yeo.py)
    network_results_dir = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_analysis"
    
    if not os.path.exists(network_results_dir):
        logging.warning(f"Network analysis results directory not found: {network_results_dir}")
        logging.info("Please run network_analysis_yeo.py first to generate network analysis results")
        return
    
    # Load network analysis results for each dataset
    network_data_dict = {}
    count_data_config = config.get('network_analysis', {}).get('count_data', {})
    
    for dataset_name in count_data_config.keys():
        network_file = os.path.join(network_results_dir, f"{dataset_name}_network_analysis.csv")
        if os.path.exists(network_file):
            network_data = pd.read_csv(network_file)
            network_data_dict[dataset_name] = network_data
        else:
            logging.warning(f"Network analysis file not found for {dataset_name}: {network_file}")
    
    if not network_data_dict:
        logging.error("No network analysis data found. Please run network_analysis_yeo.py first.")
        return
    
    # Create individual polar bar plots
    for dataset_name, network_data in network_data_dict.items():
        polar_path = os.path.join(output_dir, f"{dataset_name}_network_polar.png")
        create_polar_bar_plot(network_data, polar_path, f"Network Analysis - {dataset_name}")
    
    # Create comparison plots
    if len(network_data_dict) > 1:
        # Group by condition
        td_datasets = {k: v for k, v in network_data_dict.items() if 'td' in k.lower()}
        adhd_datasets = {k: v for k, v in network_data_dict.items() if 'adhd' in k.lower() and 'td' not in k.lower()}
        asd_datasets = {k: v for k, v in network_data_dict.items() if 'asd' in k.lower() and 'td' not in k.lower()}
        
        # TD comparison
        if len(td_datasets) > 1:
            td_path = os.path.join(output_dir, "td_networks_comparison.png")
            create_network_comparison_plot(td_datasets, td_path, "TD Networks Comparison")
        
        # ADHD comparison
        if len(adhd_datasets) > 1:
            adhd_path = os.path.join(output_dir, "adhd_networks_comparison.png")
            create_network_comparison_plot(adhd_datasets, adhd_path, "ADHD Networks Comparison")
        
        # ASD comparison
        if len(asd_datasets) > 1:
            asd_path = os.path.join(output_dir, "asd_networks_comparison.png")
            create_network_comparison_plot(asd_datasets, asd_path, "ASD Networks Comparison")
        
        # Overall heatmap
        heatmap_path = os.path.join(output_dir, "network_analysis_heatmap.png")
        create_network_heatmap(network_data_dict, heatmap_path, "Network Analysis Heatmap")
    
    logging.info(f"Created network analysis plots in {output_dir}")


def main():
    """Main function for network analysis plotting."""
    parser = argparse.ArgumentParser(
        description="Create network analysis plots using Yeo atlas grouping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create network analysis plots for all datasets
  python plot_network_analysis.py --config config.yaml
  
  # Create plots in custom directory
  python plot_network_analysis.py \\
    --config config.yaml \\
    --output_dir custom_network_plots/
        """
    )
    
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to configuration file (default: config.yaml)")
    parser.add_argument("--output_dir", type=str, default="/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/figures/network_analysis",
                       help="Output directory for plots (default: results/figures/network_analysis)")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    create_all_network_plots(config, args.output_dir)
    logging.info("Network analysis plotting completed!")


if __name__ == "__main__":
    main()
