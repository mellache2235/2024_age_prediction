#!/usr/bin/env python3
"""
Create polar area plots for network analysis similar to the attached image.

This script creates filled polar area plots (radar charts) that show network
activity across different brain regions/networks, with filled areas and
connected data points.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / 'utils'))

from plotting_utils import setup_fonts, save_figure

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_polar_area_plot(network_data: pd.DataFrame,
                          output_path: str,
                          title: str = "Network Analysis",
                          max_value: Optional[float] = None,
                          fill_color: str = 'lightblue',
                          line_color: str = 'darkblue',
                          line_width: float = 2.0,
                          alpha: float = 0.3,
                          figsize: Tuple[int, int] = (10, 10)) -> None:
    """
    Create polar area plot (radar chart) for network analysis results.
    
    This creates a filled polar plot similar to the attached image, with:
    - Filled area between data points and center
    - Connected data points with lines
    - Radial axes for each network/region
    - Light grid background
    
    Args:
        network_data (pd.DataFrame): DataFrame with columns ['Network', 'Count'] or ['network', 'mean_attribution']
        output_path (str): Output path for the plot (without extension)
        title (str): Plot title
        max_value (float, optional): Maximum value for y-axis scaling
        fill_color (str): Color for filled area
        line_color (str): Color for connecting lines and data points
        line_width (float): Width of connecting lines
        alpha (float): Transparency of filled area
        figsize (Tuple[int, int]): Figure size
    """
    setup_fonts()
    
    # Create figure with polar projection
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    
    # Prepare data - handle different column name formats
    if 'Network' in network_data.columns and 'Count' in network_data.columns:
        networks = network_data['Network'].values
        values = network_data['Count'].values
    elif 'network' in network_data.columns and 'mean_attribution' in network_data.columns:
        networks = network_data['network'].values
        values = network_data['mean_attribution'].values
    elif 'network' in network_data.columns and 'total_attribution' in network_data.columns:
        networks = network_data['network'].values
        values = network_data['total_attribution'].values
    else:
        raise ValueError(f"Expected columns 'Network'/'network' and 'Count'/'mean_attribution'/'total_attribution' not found. Available columns: {list(network_data.columns)}")
    
    # Check if we have any data
    if len(networks) == 0 or len(values) == 0:
        logging.warning(f"No network data found for {title}. Skipping plot creation.")
        return
    
    # Set up angles for each network (distribute evenly around the circle)
    angles = np.linspace(0, 2 * np.pi, len(networks), endpoint=False)
    
    # Calculate max value for scaling
    if max_value is None:
        if len(values) > 0 and np.any(values > 0):
            max_value = values.max()
        else:
            max_value = 1.0
            logging.warning(f"No positive values found in network data for {title}. Using default max_value=1.0")
    
    # Normalize values to 0-1 range for better visualization
    normalized_values = values / max_value
    
    # Close the polygon by repeating the first point at the end
    angles_closed = np.concatenate((angles, [angles[0]]))
    values_closed = np.concatenate((normalized_values, [normalized_values[0]]))
    
    # Create the filled polar area plot
    ax.fill(angles_closed, values_closed, color=fill_color, alpha=alpha, linewidth=0)
    
    # Add connecting lines
    ax.plot(angles_closed, values_closed, color=line_color, linewidth=line_width, marker='o', 
            markersize=6, markerfacecolor=line_color, markeredgecolor='white', markeredgewidth=1)
    
    # Customize the plot
    ax.set_ylim(0, 1.0)  # Normalized scale
    
    # Set network labels
    ax.set_xticks(angles)
    ax.set_xticklabels(networks, fontsize=10, fontweight='bold', color='black')
    
    # Remove grid
    ax.grid(False)
    ax.set_theta_zero_location('N')  # Start from top (North)
    ax.set_theta_direction(-1)  # Clockwise direction
    
    # Remove radial tick labels and ticks for cleaner look
    ax.set_yticks([])
    ax.set_yticklabels([])
    
    # Add title
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    save_figure(fig, output_path)
    logging.info(f"Polar area plot saved to: {output_path}")


def create_comparison_polar_plots(network_data_dict: Dict[str, pd.DataFrame],
                                 output_dir: str,
                                 base_title: str = "Network Analysis Comparison",
                                 max_value: Optional[float] = None,
                                 colors: Optional[List[str]] = None) -> None:
    """
    Create side-by-side polar area plots for comparing different datasets/conditions.
    
    Args:
        network_data_dict (Dict[str, pd.DataFrame]): Dictionary mapping dataset names to network data
        output_dir (str): Output directory for plots
        base_title (str): Base title for plots
        max_value (float, optional): Maximum value for consistent scaling across plots
        colors (List[str], optional): List of colors for each dataset
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default colors if not provided
    if colors is None:
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink', 'lightgray']
    
    # Calculate consistent max value across all datasets
    if max_value is None:
        all_values = []
        for data in network_data_dict.values():
            if 'Count' in data.columns:
                all_values.extend(data['Count'].values)
            elif 'mean_attribution' in data.columns:
                all_values.extend(data['mean_attribution'].values)
            elif 'total_attribution' in data.columns:
                all_values.extend(data['total_attribution'].values)
        
        if all_values:
            max_value = max(all_values)
        else:
            max_value = 1.0
    
    # Create individual plots for each dataset
    for i, (dataset_name, network_data) in enumerate(network_data_dict.items()):
        color = colors[i % len(colors)]
        title = f"{dataset_name}"
        output_path = output_dir / f"{dataset_name}_polar_area_plot"
        
        create_polar_area_plot(
            network_data=network_data,
            output_path=str(output_path),
            title=title,
            max_value=max_value,
            fill_color=color,
            line_color='darkblue',
            line_width=2.0,
            alpha=0.3
        )
    
    # Create combined comparison plot
    create_combined_polar_plot(network_data_dict, str(output_dir / "combined_polar_comparison"), 
                              base_title, max_value, colors)


def create_combined_polar_plot(network_data_dict: Dict[str, pd.DataFrame],
                              output_path: str,
                              title: str = "Network Analysis Comparison",
                              max_value: Optional[float] = None,
                              colors: Optional[List[str]] = None) -> None:
    """
    Create a single polar plot with multiple datasets overlaid.
    
    Args:
        network_data_dict (Dict[str, pd.DataFrame]): Dictionary mapping dataset names to network data
        output_path (str): Output path for the plot
        title (str): Plot title
        max_value (float, optional): Maximum value for consistent scaling
        colors (List[str], optional): List of colors for each dataset
    """
    setup_fonts()
    
    # Create figure with polar projection
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # Default colors if not provided
    if colors is None:
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink', 'lightgray']
    
    # Calculate consistent max value across all datasets
    if max_value is None:
        all_values = []
        for data in network_data_dict.values():
            if 'Count' in data.columns:
                all_values.extend(data['Count'].values)
            elif 'mean_attribution' in data.columns:
                all_values.extend(data['mean_attribution'].values)
            elif 'total_attribution' in data.columns:
                all_values.extend(data['total_attribution'].values)
        
        if all_values:
            max_value = max(all_values)
        else:
            max_value = 1.0
    
    # Get all unique networks across datasets
    all_networks = set()
    for data in network_data_dict.values():
        if 'Network' in data.columns:
            all_networks.update(data['Network'].values)
        elif 'network' in data.columns:
            all_networks.update(data['network'].values)
    
    all_networks = sorted(list(all_networks))
    angles = np.linspace(0, 2 * np.pi, len(all_networks), endpoint=False)
    
    # Plot each dataset
    for i, (dataset_name, network_data) in enumerate(network_data_dict.items()):
        # Prepare data for this dataset
        if 'Network' in network_data.columns and 'Count' in network_data.columns:
            networks = network_data['Network'].values
            values = network_data['Count'].values
        elif 'network' in network_data.columns and 'mean_attribution' in network_data.columns:
            networks = network_data['network'].values
            values = network_data['mean_attribution'].values
        elif 'network' in network_data.columns and 'total_attribution' in network_data.columns:
            networks = network_data['network'].values
            values = network_data['total_attribution'].values
        else:
            continue
        
        # Create mapping from network to value
        network_to_value = dict(zip(networks, values))
        
        # Get values for all networks (fill missing with 0)
        dataset_values = [network_to_value.get(network, 0) for network in all_networks]
        normalized_values = np.array(dataset_values) / max_value
        
        # Close the polygon
        angles_closed = np.concatenate((angles, [angles[0]]))
        values_closed = np.concatenate((normalized_values, [normalized_values[0]]))
        
        # Create the filled polar area plot
        color = colors[i % len(colors)]
        ax.fill(angles_closed, values_closed, color=color, alpha=0.3, linewidth=0, label=dataset_name)
        ax.plot(angles_closed, values_closed, color=color, linewidth=2, marker='o', 
                markersize=4, markerfacecolor=color, markeredgecolor='white', markeredgewidth=1)
    
    # Customize the plot
    ax.set_ylim(0, 1.0)
    ax.set_xticks(angles)
    ax.set_xticklabels(all_networks, fontsize=10, fontweight='bold', color='black')
    ax.grid(False)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_yticks([])
    ax.set_yticklabels([])
    
    # Add title and legend
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    save_figure(fig, output_path)
    logging.info(f"Combined polar area plot saved to: {output_path}")


def main():
    """Main function for creating polar network plots."""
    parser = argparse.ArgumentParser(
        description="Create polar area plots for network analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create polar plots for network analysis results
  python create_polar_network_plots.py --network_csv results/network_analysis/nki_network_analysis.csv --output_dir results/polar_plots
  
  # Create comparison plots for multiple datasets
  python create_polar_network_plots.py --config config.yaml --output_dir results/polar_plots --comparison
        """
    )
    
    parser.add_argument('--network_csv', type=str, help='Path to network analysis CSV file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for plots')
    parser.add_argument('--title', type=str, default='Network Analysis', help='Plot title')
    parser.add_argument('--config', type=str, help='Path to config file for multiple datasets')
    parser.add_argument('--comparison', action='store_true', help='Create comparison plots for multiple datasets')
    parser.add_argument('--max_value', type=float, help='Maximum value for scaling')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.comparison and args.config:
        # Load configuration and create comparison plots
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get network analysis results directory
        network_results_dir = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_analysis_yeo"
        
        # Find all network analysis CSV files
        network_data_dict = {}
        for dataset_name in ['nki', 'adhd200_adhd', 'cmihbn_adhd', 'abide_asd', 'stanford_asd']:
            csv_path = Path(network_results_dir) / f"{dataset_name}_network_analysis.csv"
            if csv_path.exists():
                try:
                    network_data = pd.read_csv(csv_path)
                    network_data_dict[dataset_name] = network_data
                    logging.info(f"Loaded network data for {dataset_name}: {len(network_data)} networks")
                except Exception as e:
                    logging.warning(f"Could not load network data for {dataset_name}: {e}")
        
        if network_data_dict:
            create_comparison_polar_plots(network_data_dict, str(output_dir), args.title, args.max_value)
        else:
            logging.error("No network analysis data found for comparison plots")
    
    elif args.network_csv:
        # Create single polar plot
        if not os.path.exists(args.network_csv):
            logging.error(f"Network CSV file not found: {args.network_csv}")
            return
        
        network_data = pd.read_csv(args.network_csv)
        output_path = output_dir / "network_polar_area_plot"
        
        create_polar_area_plot(
            network_data=network_data,
            output_path=str(output_path),
            title=args.title,
            max_value=args.max_value
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
