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
    # The region names are likely the ROI indices themselves (1, 2, 3, ..., 246)
    def extract_roi_index(region_name):
        try:
            if isinstance(region_name, str):
                # Try direct conversion first (region names might be "1", "2", etc.)
                if region_name.isdigit():
                    return int(region_name)
                
                # Try to extract number from region name
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
    
    # Debug: Check ROI index extraction
    logging.info(f"Sample region names: {count_data['region'].head().tolist()}")
    logging.info(f"Sample extracted ROI indices: {count_data['roi_index'].head().tolist()}")
    logging.info(f"Valid ROI indices: {count_data['roi_index'].notna().sum()}/{len(count_data)}")
    
    count_data['network'] = count_data['roi_index'].map(network_mapping)
    
    # Handle unmapped regions
    unmapped = count_data['network'].isna().sum()
    if unmapped > 0:
        logging.warning(f"{unmapped} regions could not be mapped to networks")
        logging.info(f"Sample unmapped regions: {count_data[count_data['network'].isna()]['region'].head().tolist()}")
        logging.info(f"Sample unmapped ROI indices: {count_data[count_data['network'].isna()]['roi_index'].head().tolist()}")
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
    # Debug: Check data before aggregation
    logging.info(f"Columns in count_data: {list(count_data.columns)}")
    logging.info(f"Sample attribution values: {count_data['attribution'].head().tolist()}")
    logging.info(f"Sample network assignments: {count_data['network'].head().tolist()}")
    logging.info(f"Unique networks: {count_data['network'].unique()}")
    
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
    
    # Debug: Check final aggregated data
    logging.info(f"Final network aggregation shape: {network_agg.shape}")
    logging.info(f"Final network aggregation columns: {list(network_agg.columns)}")
    logging.info(f"Sample mean_attribution values: {network_agg['mean_attribution'].head().tolist()}")
    logging.info(f"Max mean_attribution: {network_agg['mean_attribution'].max()}")
    logging.info(f"Min mean_attribution: {network_agg['mean_attribution'].min()}")
    
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
                          output_dir: str, title: str, network_names: List[str] = None, 
                          dataset_name: str = None) -> None:
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
    
    # Save network data with dataset name in filename
    # Use provided dataset_name or extract from title
    if dataset_name is None:
        dataset_name = title.split(" - ")[-1].lower().replace(" ", "_")
    network_csv = output_dir / f"{dataset_name}_network_analysis.csv"
    network_data.to_csv(network_csv, index=False)
    logging.info(f"Network data saved to: {network_csv}")
    
    # Create visualizations
    from plot_network_analysis import create_polar_bar_plot, create_network_comparison_plot
    
    # Create polar plot
    polar_plot = output_dir / f"{dataset_name}_network_polar_plot"
    create_polar_bar_plot(network_data, str(polar_plot), f"{title} - Polar Plot", show_values=True)
    
    # Analyze patterns
    analysis_results = analyze_network_patterns(network_data)
    
    # Save analysis results
    import json
    results_file = output_dir / f"{dataset_name}_network_analysis_results.json"
    with open(results_file, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    logging.info("Network analysis completed!")
    logging.info(f"Results saved to: {output_dir}")
    logging.info(f"Top 3 networks: {[net['network'] for net in analysis_results['top_3_networks']]}")


def create_shared_network_analysis(dataset_excel_paths: List[str], 
                                  yeo_atlas_path: str,
                                  output_dir: str,
                                  title: str,
                                  network_names: List[str] = None,
                                  min_datasets: int = 2) -> None:
    """
    Create network analysis for shared regions across multiple datasets.
    Uses minimum count approach for shared regions.
    
    Args:
        dataset_excel_paths (List[str]): List of paths to count data Excel files
        yeo_atlas_path (str): Path to Yeo atlas CSV file
        output_dir (str): Output directory for results
        title (str): Title for plots
        network_names (List[str], optional): List of network names
        min_datasets (int): Minimum number of datasets a region must appear in
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Yeo atlas
    yeo_atlas = load_yeo_atlas(yeo_atlas_path, network_names)
    
    # Collect shared regions using minimum count approach
    all_regions = {}
    
    for excel_path in dataset_excel_paths:
        if not os.path.exists(excel_path):
            logging.warning(f"Count data file not found: {excel_path}")
            continue
            
        dataset_name = Path(excel_path).stem.replace('_count_data', '').replace('top_50_consensus_features_', '').replace('_aging', '')
        
        try:
            count_data = pd.read_excel(excel_path)
        except Exception as e:
            logging.error(f"Error reading Excel file {excel_path}: {e}")
            continue
        
        # Check for required columns
        if 'Count' not in count_data.columns:
            logging.error(f"No Count column found in {excel_path}. Available columns: {list(count_data.columns)}")
            continue
        
        # Get top 50 regions
        top_regions = count_data.nlargest(50, 'Count')
        
        for _, row in top_regions.iterrows():
            # Use Region ID as the key for matching across datasets
            region_id = row.get('Region ID', row.get('(ID) Region Label', 'Unknown'))
            
            if region_id not in all_regions:
                all_regions[region_id] = {
                    'counts': [],
                    'datasets': []
                }
            
            all_regions[region_id]['counts'].append(row['Count'])
            all_regions[region_id]['datasets'].append(dataset_name)
    
    # Filter regions that appear in at least min_datasets
    shared_regions = {region_id: data for region_id, data in all_regions.items() 
                     if len(data['datasets']) >= min_datasets}
    
    if not shared_regions:
        logging.warning(f"No shared regions found across {min_datasets} datasets")
        return
    
    # Create shared count data using minimum count approach
    shared_count_data = []
    for region_id, data in shared_regions.items():
        min_count = np.min(data['counts'])
        shared_count_data.append({
            'region': region_id,
            'Count': int(min_count)
        })
    
    # Convert to DataFrame
    shared_df = pd.DataFrame(shared_count_data)
    
    # Map ROIs to networks
    network_data = map_rois_to_networks(shared_df, yeo_atlas)
    
    # Aggregate by networks
    aggregated_data = aggregate_by_networks(network_data)
    
    # Save network analysis results
    output_file = os.path.join(output_dir, f"shared_network_analysis.csv")
    aggregated_data.to_csv(output_file, index=False)
    logging.info(f"Shared network analysis saved to: {output_file}")
    
    # Create plots
    from plot_network_analysis import create_polar_bar_plot
    
    # Polar bar plot
    polar_plot_path = os.path.join(output_dir, f"shared_network_polar")
    create_polar_bar_plot(aggregated_data, polar_plot_path, title, show_values=True)
    
    # Analyze network patterns
    network_patterns = analyze_network_patterns(aggregated_data)
    
    # Save analysis results
    import json
    results_file = os.path.join(output_dir, f"shared_network_analysis_results.json")
    with open(results_file, 'w') as f:
        json.dump(network_patterns, f, indent=2, default=str)
    
    logging.info(f"Shared network analysis completed for {len(shared_regions)} shared regions")


def main():
    """Main function to process all datasets from config."""
    parser = argparse.ArgumentParser(description="Network analysis using Yeo atlas")
    parser.add_argument("--count_csv", type=str, 
                       help="Path to count data CSV file (generated by generate_count_data.py)")
    parser.add_argument("--yeo_atlas", type=str, 
                       help="Path to Yeo atlas CSV file")
    parser.add_argument("--output_dir", type=str, default="/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_analysis_yeo",
                       help="Output directory for results")
    parser.add_argument("--title", type=str, default="Network Attribution Analysis",
                       help="Title for plots")
    parser.add_argument("--process_all", action="store_true",
                       help="Process all datasets from config")
    parser.add_argument("--process_shared", action="store_true",
                       help="Process shared network analysis across cohorts")
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(__file__).parent.parent / 'config.yaml'
    if not config_path.exists():
        logging.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.process_all:
        # Process all datasets from config
        count_data_config = config.get('network_analysis', {}).get('count_data', {})
        yeo_atlas_path = config.get('network_analysis', {}).get('yeo_atlas_path', '/path/to/yeo_atlas.csv')
        network_names = config.get('network_analysis', {}).get('network_names', None)
        
        for dataset_name, excel_path in count_data_config.items():
            # Convert Excel to CSV if needed
            csv_path = f"/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/count_data/{dataset_name}_count_data.csv"
            if not os.path.exists(csv_path):
                logging.info(f"Converting {excel_path} to CSV...")
                from count_data_utils import convert_excel_to_csv
                convert_excel_to_csv(excel_path, csv_path)
            
            # Process dataset
            output_dir = f"/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_analysis_yeo/{dataset_name}"
            title = f"Network Analysis - {dataset_name.replace('_', ' ').title()}"
            
            logging.info(f"Processing {dataset_name}...")
            process_single_dataset(csv_path, yeo_atlas_path, output_dir, title, network_names, dataset_name)
        
        logging.info("All datasets processed!")
    
    elif args.process_shared:
        # Process shared network analysis across cohorts
        count_data_config = config.get('network_analysis', {}).get('count_data', {})
        yeo_atlas_path = config.get('network_analysis', {}).get('yeo_atlas_path', '/path/to/yeo_atlas.csv')
        network_names = config.get('network_analysis', {}).get('network_names', None)
        
        # Shared among TD cohorts
        td_datasets = ['dev', 'nki', 'adhd200_td', 'cmihbn_td']
        td_paths = [count_data_config.get(d) for d in td_datasets if count_data_config.get(d) and os.path.exists(count_data_config.get(d))]
        
        if len(td_paths) >= 2:
            output_dir = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_analysis_yeo/shared_TD"
            title = "Shared Network Analysis - TD Cohorts"
            logging.info("Processing shared TD network analysis...")
            create_shared_network_analysis(td_paths, yeo_atlas_path, output_dir, title, network_names, min_datasets=2)
        
        # Shared among ADHD cohorts
        adhd_datasets = ['adhd200_adhd', 'cmihbn_adhd']
        adhd_paths = [count_data_config.get(d) for d in adhd_datasets if count_data_config.get(d) and os.path.exists(count_data_config.get(d))]
        
        if len(adhd_paths) >= 2:
            output_dir = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_analysis_yeo/shared_ADHD"
            title = "Shared Network Analysis - ADHD Cohorts"
            logging.info("Processing shared ADHD network analysis...")
            create_shared_network_analysis(adhd_paths, yeo_atlas_path, output_dir, title, network_names, min_datasets=2)
        
        # Shared among ASD cohorts
        asd_datasets = ['abide_asd', 'stanford_asd']
        asd_paths = [count_data_config.get(d) for d in asd_datasets if count_data_config.get(d) and os.path.exists(count_data_config.get(d))]
        
        if len(asd_paths) >= 2:
            output_dir = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_analysis_yeo/shared_ASD"
            title = "Shared Network Analysis - ASD Cohorts"
            logging.info("Processing shared ASD network analysis...")
            create_shared_network_analysis(asd_paths, yeo_atlas_path, output_dir, title, network_names, min_datasets=2)
        
        # Shared among all cohorts
        all_paths = [path for path in count_data_config.values() if path and os.path.exists(path)]
        if len(all_paths) >= 3:
            output_dir = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_analysis_yeo/shared_all"
            title = "Shared Network Analysis - All Cohorts"
            logging.info("Processing shared all cohorts network analysis...")
            create_shared_network_analysis(all_paths, yeo_atlas_path, output_dir, title, network_names, min_datasets=3)
        
        logging.info("All shared network analyses completed!")
        
    else:
        # Process single dataset
        if not args.count_csv or not args.yeo_atlas:
            logging.error("Both --count_csv and --yeo_atlas are required for single dataset processing")
            return
        
        network_names = config.get('network_analysis', {}).get('network_names', None)
        # Extract dataset name from title for single dataset processing
        single_dataset_name = args.title.split(" - ")[-1].lower().replace(" ", "_") if " - " in args.title else "dataset"
        process_single_dataset(args.count_csv, args.yeo_atlas, args.output_dir, args.title, network_names, single_dataset_name)

if __name__ == "__main__":
    main()
