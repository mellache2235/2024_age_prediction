#!/usr/bin/env python3
"""
Network-level analysis using Yeo atlas to group ROIs by brain networks.

This script groups ROIs by Yeo 17-network atlas and creates network-level
attribution plots and analysis.

Usage:
    python scripts/network_analysis_yeo.py --count_csv <count_data.csv> --yeo_atlas <yeo_atlas.csv>
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / 'utils'))

from count_data_utils import load_count_data, create_region_mapping
from plotting_utils import setup_fonts, save_figure

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_yeo_atlas(yeo_atlas_path: str, network_names: List[str] = None) -> pd.DataFrame:
    """
    Load Yeo 17-network atlas.
    
    Args:
        yeo_atlas_path (str): Path to Yeo atlas CSV file
        network_names (List[str], optional): List of network names for mapping
        
    Returns:
        pd.DataFrame: Yeo atlas data with ROI indices and network assignments
    """
    if not os.path.exists(yeo_atlas_path):
        raise FileNotFoundError(f"Yeo atlas file not found: {yeo_atlas_path}")
    
    yeo_atlas = pd.read_csv(yeo_atlas_path)
    logging.info(f"Loaded Yeo atlas with {len(yeo_atlas)} ROIs")
    
    # Check if we have the expected columns
    if 'Yeo_17network' not in yeo_atlas.columns:
        logging.warning("Yeo_17network column not found, checking for alternative column names")
        # Try common alternative names
        for col in yeo_atlas.columns:
            if 'network' in col.lower() or 'yeo' in col.lower():
                yeo_atlas = yeo_atlas.rename(columns={col: 'Yeo_17network'})
                break
    
    # Map network indices to network names if provided
    if network_names is not None:
        # Create mapping from network index to network name
        network_mapping = {i: name for i, name in enumerate(network_names)}
        yeo_atlas['Network_Name'] = yeo_atlas['Yeo_17network'].map(network_mapping)
        logging.info(f"Mapped network indices to names: {len(network_names)} networks")
    
    return yeo_atlas

def map_rois_to_networks(count_data: pd.DataFrame, yeo_atlas: pd.DataFrame) -> pd.DataFrame:
    """
    Map ROI regions to Yeo networks using ROI index.
    
    Args:
        count_data (pd.DataFrame): Count data with region and count columns
        yeo_atlas (pd.DataFrame): Yeo atlas data with ROI indices
        
    Returns:
        pd.DataFrame: Count data with network assignments
    """
    # Create a mapping from ROI index to network assignments
    network_mapping = {}
    
    # Map ROI indices to network assignments
    for idx, row in yeo_atlas.iterrows():
        # ROI index is typically 1-based in the atlas
        roi_index = idx + 1
        
        # Use Network_Name if available, otherwise use Yeo_17network
        if 'Network_Name' in yeo_atlas.columns and pd.notna(row['Network_Name']):
            network_mapping[roi_index] = row['Network_Name']
        else:
            network_mapping[roi_index] = row['Yeo_17network']
    
    # Extract ROI index from region names in count data
    # Assuming region names are like "ROI_1", "ROI_2", etc. or just "1", "2", etc.
    def extract_roi_index(region_name):
        try:
            # Try to extract number from region name
            if isinstance(region_name, str):
                # Handle formats like "ROI_1", "Region_1", or just "1"
                import re
                numbers = re.findall(r'\d+', region_name)
                if numbers:
                    return int(numbers[0])
            elif isinstance(region_name, (int, float)):
                return int(region_name)
        except:
            pass
        return None
    
    # Map regions to networks using ROI index
    count_data['roi_index'] = count_data['region'].apply(extract_roi_index)
    count_data['network'] = count_data['roi_index'].map(network_mapping)
    
    # Handle unmapped regions
    unmapped = count_data['network'].isna().sum()
    if unmapped > 0:
        logging.warning(f"{unmapped} regions could not be mapped to networks")
        logging.info(f"Sample unmapped regions: {count_data[count_data['network'].isna()]['region'].head().tolist()}")
        # Fill unmapped regions with 'Unknown'
        count_data['network'] = count_data['network'].fillna('Unknown')
    
    # Drop the temporary roi_index column
    count_data = count_data.drop(columns=['roi_index'])
    
    return count_data

def aggregate_by_networks(count_data: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate attribution scores by network.
    
    Args:
        count_data (pd.DataFrame): Count data with network assignments
        
    Returns:
        pd.DataFrame: Network-level aggregated data
    """
    # Group by network and aggregate
    network_agg = count_data.groupby('network').agg({
        'attribution': ['mean', 'std', 'count', 'sum']
    }).round(4)
    
    # Flatten column names
    network_agg.columns = ['_'.join(col).strip() for col in network_agg.columns]
    network_agg = network_agg.reset_index()
    
    # Rename columns for clarity
    network_agg = network_agg.rename(columns={
        'attribution_mean': 'mean_attribution',
        'attribution_std': 'std_attribution',
        'attribution_count': 'n_regions',
        'attribution_sum': 'total_attribution'
    })
    
    # Sort by mean attribution
    network_agg = network_agg.sort_values('mean_attribution', ascending=False)
    
    return network_agg

def create_network_polar_plot(network_data: pd.DataFrame, output_file: str = None, 
                            title: str = "Network Attribution Plot"):
    """
    Create polar bar plot for network-level data.
    
    Args:
        network_data (pd.DataFrame): Network-level aggregated data
        output_file (str): Output file path for the plot
        title (str): Plot title
    """
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # Get data
    networks = network_data['network']
    attributions = network_data['mean_attribution']
    
    # Create angles for each network
    angles = np.linspace(0, 2 * np.pi, len(networks), endpoint=False)
    
    # Create bars with different colors for each network
    colors = plt.cm.Set3(np.linspace(0, 1, len(networks)))
    bars = ax.bar(angles, attributions, width=0.8, alpha=0.8, color=colors)
    
    # Customize the plot
    max_attr = attributions.max()
    ax.set_ylim(0, max_attr * 1.1)
    ax.set_ylabel('Mean Attribution', labelpad=20)
    
    # Set network labels
    ax.set_xticks(angles)
    ax.set_xticklabels(networks, fontsize=10, fontweight='bold', color='black')
    
    # Remove y-axis ticks and labels (match R theme)
    ax.set_yticks([])
    ax.set_yticklabels([])
    
    # Remove grid
    ax.grid(False)
    
    # Remove spines/borders
    ax.spines['polar'].set_visible(False)
    
    # Set title
    ax.set_title(title, pad=20, fontsize=14, fontweight='bold')
    
    # Add legend with network names and values
    legend_labels = [f"{net}: {attr:.3f}" for net, attr in zip(networks, attributions)]
    ax.legend(bars, legend_labels, loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logging.info(f"Network plot saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()

def create_network_bar_plot(network_data: pd.DataFrame, output_file: str = None,
                          title: str = "Network Attribution Bar Plot"):
    """
    Create regular bar plot for network-level data.
    
    Args:
        network_data (pd.DataFrame): Network-level aggregated data
        output_file (str): Output file path for the plot
        title (str): Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bar plot
    networks = network_data['network']
    attributions = network_data['mean_attribution']
    errors = network_data['std_attribution']
    
    bars = ax.bar(networks, attributions, yerr=errors, capsize=5, alpha=0.8,
                  color=plt.cm.Set3(np.linspace(0, 1, len(networks))))
    
    # Customize plot
    ax.set_xlabel('Brain Networks', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Attribution', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, attr, n_regions in zip(bars, attributions, network_data['n_regions']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{attr:.3f}\n(n={n_regions})', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save or show plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logging.info(f"Network bar plot saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()

def analyze_network_patterns(network_data: pd.DataFrame) -> Dict:
    """
    Analyze network-level patterns.
    
    Args:
        network_data (pd.DataFrame): Network-level aggregated data
        
    Returns:
        Dict: Analysis results
    """
    # Calculate statistics
    total_networks = len(network_data)
    total_regions = network_data['n_regions'].sum()
    mean_attribution = network_data['mean_attribution'].mean()
    max_attribution = network_data['mean_attribution'].max()
    
    # Top networks
    top_3_networks = network_data.head(3)
    
    # Network with most regions
    most_regions_network = network_data.loc[network_data['n_regions'].idxmax()]
    
    results = {
        'total_networks': total_networks,
        'total_regions': total_regions,
        'mean_attribution_across_networks': mean_attribution,
        'max_attribution': max_attribution,
        'top_3_networks': top_3_networks.to_dict('records'),
        'network_with_most_regions': most_regions_network.to_dict(),
        'network_summary': network_data.to_dict('records')
    }
    
    return results

def process_single_dataset(count_csv_path: str, yeo_atlas_path: str, 
                          output_dir: str, title: str, network_names: List[str] = None) -> None:
    """Process a single dataset for network analysis."""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load count data
    if not os.path.exists(count_csv_path):
        logging.error(f"Count data file not found: {count_csv_path}")
        return
    
    # Read file based on extension
    if count_csv_path.endswith('.xlsx') or count_csv_path.endswith('.xls'):
        count_data = pd.read_excel(count_csv_path)
    else:
        count_data = pd.read_csv(count_csv_path)
    
    logging.info(f"Loaded count data with {len(count_data)} regions")
    logging.info(f"Columns in count data: {list(count_data.columns)}")
    
    # Standardize column names
    if 'Region ID' in count_data.columns and 'Count' in count_data.columns:
        count_data = count_data.rename(columns={'Region ID': 'region', 'Count': 'attribution'})
        logging.info("Renamed 'Region ID' to 'region' and 'Count' to 'attribution'")
    elif 'region' not in count_data.columns:
        logging.error(f"Expected 'region' or 'Region ID' column not found. Available columns: {list(count_data.columns)}")
        return
    
    # Load Yeo atlas with network names
    yeo_atlas = load_yeo_atlas(yeo_atlas_path, network_names)
    
    # Map ROIs to networks
    count_data_with_networks = map_rois_to_networks(count_data, yeo_atlas)
    
    # Aggregate by networks
    network_data = aggregate_by_networks(count_data_with_networks)
    
    # Save network data
    network_csv = output_dir / "network_aggregated_data.csv"
    network_data.to_csv(network_csv, index=False)
    logging.info(f"Network data saved to: {network_csv}")
    
    # Create visualizations
    polar_plot = output_dir / "network_polar_plot.png"
    create_network_polar_plot(network_data, str(polar_plot), f"{args.title} - Polar Plot")
    
    bar_plot = output_dir / "network_bar_plot.png"
    create_network_bar_plot(network_data, str(bar_plot), f"{args.title} - Bar Plot")
    
    # Analyze patterns
    analysis_results = analyze_network_patterns(network_data)
    
    # Save analysis results
    import json
    results_file = output_dir / "network_analysis_results.json"
    with open(results_file, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    logging.info("Network analysis completed!")
    logging.info(f"Results saved to: {output_dir}")
    logging.info(f"Top 3 networks: {[net['network'] for net in analysis_results['top_3_networks']]}")


def main():
    """Main function to process all datasets from config."""
    parser = argparse.ArgumentParser(description="Network analysis using Yeo atlas")
    parser.add_argument("--count_csv", type=str, 
                       help="Path to count data CSV file (generated by generate_count_data.py)")
    parser.add_argument("--yeo_atlas", type=str, 
                       help="Path to Yeo atlas CSV file")
    parser.add_argument("--output_dir", type=str, default="results/network_analysis_yeo",
                       help="Output directory for results")
    parser.add_argument("--title", type=str, default="Network Attribution Analysis",
                       help="Title for plots")
    parser.add_argument("--process_all", action="store_true",
                       help="Process all datasets from config")
    
    args = parser.parse_args()
    
    # Load configuration
    with open('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    if args.process_all:
        # Process all datasets from config
        count_data_config = config.get('network_analysis', {}).get('count_data', {})
        yeo_atlas_path = config.get('network_analysis', {}).get('yeo_atlas_path', '/path/to/yeo_atlas.csv')
        network_names = config.get('network_analysis', {}).get('network_names', None)
        
        for dataset_name, excel_path in count_data_config.items():
            # Convert Excel to CSV if needed
            csv_path = f"results/count_data/{dataset_name}_count_data.csv"
            if not os.path.exists(csv_path):
                logging.info(f"Converting {excel_path} to CSV...")
                from count_data_utils import convert_excel_to_csv
                convert_excel_to_csv(excel_path, csv_path)
            
            # Process dataset
            output_dir = f"results/network_analysis_yeo/{dataset_name}"
            title = f"Network Analysis - {dataset_name.replace('_', ' ').title()}"
            
            logging.info(f"Processing {dataset_name}...")
            process_single_dataset(csv_path, yeo_atlas_path, output_dir, title, network_names)
        
        logging.info("All datasets processed!")
        
    else:
        # Process single dataset
        if not args.count_csv or not args.yeo_atlas:
            logging.error("Both --count_csv and --yeo_atlas are required for single dataset processing")
            return
        
        network_names = config.get('network_analysis', {}).get('network_names', None)
        process_single_dataset(args.count_csv, args.yeo_atlas, args.output_dir, args.title, network_names)

if __name__ == "__main__":
    main()
