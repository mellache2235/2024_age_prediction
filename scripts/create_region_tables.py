#!/usr/bin/env python3
"""
Create tables for regions of importance based on count data from Excel files.

This script generates comprehensive tables summarizing important brain regions
across different datasets and cohorts using the count data Excel files.

Format: Brain Regions, Subdivision, (ID) Region Label, Count
For shared regions: Count = minimum count across datasets
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import logging

# Try to import openpyxl for Excel support
try:
    import openpyxl
except ImportError:
    logging.warning("openpyxl not found. Install with: pip install openpyxl")

def load_roi_labels_mapping(roi_labels_path: str) -> Dict[int, str]:
    """
    Load ROI labels mapping from the Brainnetome ROI labels file.
    
    Args:
        roi_labels_path (str): Path to the ROI labels file
        
    Returns:
        Dict[int, str]: Mapping from ROI index to brain region name
    """
    roi_mapping = {}
    try:
        with open(roi_labels_path, 'r') as f:
            for i, line in enumerate(f, 1):  # ROI indices start from 1
                roi_mapping[i] = line.strip()
        logging.info(f"Loaded {len(roi_mapping)} ROI labels from {roi_labels_path}")
    except Exception as e:
        logging.warning(f"Could not load ROI labels from {roi_labels_path}: {e}")
    return roi_mapping

def get_accurate_brain_region(row: pd.Series, roi_mapping: Dict[int, str]) -> str:
    """
    Get brain region name from Gyrus column (not ROI mapping).
    
    Args:
        row (pd.Series): Row from count data
        roi_mapping (Dict[int, str]): ROI index to brain region mapping (not used)
        
    Returns:
        str: Brain region name from Gyrus column
    """
    # Use Gyrus column for brain regions (as requested)
    if 'Gyrus' in row and pd.notna(row['Gyrus']) and str(row['Gyrus']).strip() != 'Unknown':
        return str(row['Gyrus']).strip()
    
    # Fallback to Description if Gyrus is not available
    if 'Description' in row and pd.notna(row['Description']) and str(row['Description']).strip() != 'Unknown':
        return str(row['Description']).strip()
    
    return 'Unknown'

def get_subdivision_name(row: pd.Series) -> str:
    """
    Get subdivision name from Region Alias column (contains a9l, a8m, a8dl, etc.).
    
    Args:
        row (pd.Series): Row from count data
        
    Returns:
        str: Subdivision name from Region Alias column
    """
    # Use Region Alias column for subdivision (contains a9l, a8m, a8dl, etc.)
    if 'Region Alias' in row and pd.notna(row['Region Alias']) and str(row['Region Alias']).strip() != 'Unknown':
        return str(row['Region Alias']).strip()
    
    # Fallback to other columns if Region Alias is not available
    for col in ['Gyrus', 'Description']:
        if col in row and pd.notna(row[col]) and str(row[col]).strip() != 'Unknown':
            return str(row[col]).strip()
    
    return 'Unknown'

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_dataset_region_table_from_excel(excel_path: str, 
                                          output_path: str,
                                          roi_labels_path: str,
                                          top_n: int = 50) -> pd.DataFrame:
    """
    Create a region table for a single dataset from Excel file.
    
    Format: Brain Regions, Subdivision, (ID) Region Label, Count
    
    Args:
        excel_path (str): Path to count data Excel file
        output_path (str): Output path for the table
        roi_labels_path (str): Path to ROI labels file for accurate mapping
        top_n (int): Number of top regions to include
        
    Returns:
        pd.DataFrame: Region table
    """
    # Load count data from Excel
    try:
        count_data = pd.read_excel(excel_path)
    except Exception as e:
        logging.error(f"Error reading Excel file {excel_path}: {e}")
        return pd.DataFrame()
    
    # Check for required columns
    required_cols = ['Count']
    if not all(col in count_data.columns for col in required_cols):
        logging.error(f"Missing required columns in {excel_path}. Available: {list(count_data.columns)}")
        return pd.DataFrame()
    
    # Load ROI labels for accurate brain region mapping
    roi_mapping = load_roi_labels_mapping(roi_labels_path)
    
    # Sort by count and get top N
    top_regions = count_data.nlargest(top_n, 'Count')
    
    # Create final table with correct column mapping
    region_table = pd.DataFrame({
        'Brain Regions': top_regions.apply(lambda row: get_accurate_brain_region(row, roi_mapping), axis=1),
        'Subdivision': top_regions.apply(lambda row: get_subdivision_name(row), axis=1),
        '(ID) Region Label': top_regions.get('(ID) Region Label', top_regions.get('Region ID', 'Unknown')),
        'Count': top_regions['Count']
    })
    
    # Save table
    region_table.to_csv(output_path, index=False)
    logging.info(f"Created region table for dataset: {len(region_table)} regions saved to {output_path}")
    
    return region_table


def create_shared_region_table_from_excel(dataset_excel_paths: List[str],
                                         output_path: str,
                                         roi_labels_path: str,
                                         top_n: int = 50,
                                         min_datasets: int = 2) -> pd.DataFrame:
    """
    Create a table of regions shared across multiple datasets from Excel files.
    
    Format: Brain Regions, Subdivision, (ID) Region Label, Count
    For shared regions: Count = minimum count across datasets
    
    Args:
        dataset_excel_paths (List[str]): List of paths to count data Excel files
        output_path (str): Output path for the table
        roi_labels_path (str): Path to ROI labels file for accurate mapping
        top_n (int): Number of top regions to include per dataset
        min_datasets (int): Minimum number of datasets a region must appear in
        
    Returns:
        pd.DataFrame: Shared regions table
    """
    # Load ROI labels for accurate brain region mapping
    roi_mapping = load_roi_labels_mapping(roi_labels_path)
    
    # Collect top regions from each dataset
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
        
        # Get top N regions
        top_regions = count_data.nlargest(top_n, 'Count')
        
        for _, row in top_regions.iterrows():
            # Use Region ID as the key for matching across datasets
            region_id = row.get('Region ID', row.get('(ID) Region Label', 'Unknown'))
            
            if region_id not in all_regions:
                all_regions[region_id] = {
                    'brain_region': get_accurate_brain_region(row, roi_mapping),
                    'subdivision': get_subdivision_name(row),
                    'region_label': row.get('(ID) Region Label', region_id),
                    'counts': [],
                    'datasets': []
                }
            
            all_regions[region_id]['counts'].append(row['Count'])
            all_regions[region_id]['datasets'].append(dataset_name)
    
    # Filter regions that appear in at least min_datasets
    shared_regions = {region_id: data for region_id, data in all_regions.items() 
                     if len(data['datasets']) >= min_datasets}
    
    # Create table in the desired format
    # For shared regions, use the MINIMUM count across datasets (as requested)
    table_data = []
    for region_id, data in shared_regions.items():
        min_count = np.min(data['counts'])
        
        table_data.append({
            'Brain Regions': data['brain_region'],
            'Subdivision': data['subdivision'],
            '(ID) Region Label': data['region_label'],
            'Count': int(min_count)  # Use minimum count as requested
        })
    
    # Create and sort table
    if table_data:
        shared_table = pd.DataFrame(table_data)
        shared_table = shared_table.sort_values('Count', ascending=False)
        
        # Save table
        shared_table.to_csv(output_path, index=False)
        logging.info(f"Created shared region table: {len(shared_table)} regions shared across datasets, saved to {output_path}")
    else:
        # Create empty table with proper columns
        shared_table = pd.DataFrame(columns=['Brain Regions', 'Subdivision', '(ID) Region Label', 'Count'])
        shared_table.to_csv(output_path, index=False)
        logging.warning(f"No shared regions found. Created empty table at {output_path}")
    
    return shared_table


def create_all_region_tables(config: Dict, output_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Create all region tables (individual datasets and shared regions).
    
    Args:
        config (Dict): Configuration dictionary
        output_dir (str): Output directory for tables
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary of created tables
    """
    # Get ROI labels path from config
    roi_labels_path = config.get('network_analysis', {}).get('roi_labels_path', '')
    if not roi_labels_path:
        logging.warning("ROI labels path not found in config. Using fallback mapping.")
        roi_labels_path = ""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get count data paths from config
    count_data_paths = config.get('network_analysis', {}).get('count_data', {})
    
    if not count_data_paths:
        logging.error("No count data paths found in configuration")
        return {}
    
    tables = {}
    
    # Create individual dataset tables
    logging.info("Creating individual dataset region tables...")
    for dataset_name, excel_path in count_data_paths.items():
        if not os.path.exists(excel_path):
            logging.warning(f"Count data Excel file not found for {dataset_name}: {excel_path}")
            continue
        
        output_path = os.path.join(output_dir, f"{dataset_name}_region_table.csv")
        table = create_dataset_region_table_from_excel(excel_path, output_path, roi_labels_path)
        if not table.empty:
            tables[f"{dataset_name}_individual"] = table
    
    # Create shared region tables
    logging.info("Creating shared region tables...")
    
    # Shared among TD cohorts
    td_datasets = ['dev', 'nki', 'adhd200_td', 'cmihbn_td']
    td_paths = [count_data_paths.get(d) for d in td_datasets if count_data_paths.get(d) and os.path.exists(count_data_paths.get(d))]
    
    if len(td_paths) >= 2:
        output_path = os.path.join(output_dir, "shared_regions_TD.csv")
        table = create_shared_region_table_from_excel(td_paths, output_path, roi_labels_path, min_datasets=2)
        if not table.empty:
            tables["shared_TD"] = table
    
    # Shared among ADHD cohorts
    adhd_datasets = ['adhd200_adhd', 'cmihbn_adhd']
    adhd_paths = [count_data_paths.get(d) for d in adhd_datasets if count_data_paths.get(d) and os.path.exists(count_data_paths.get(d))]
    
    if len(adhd_paths) >= 2:
        output_path = os.path.join(output_dir, "shared_regions_ADHD.csv")
        table = create_shared_region_table_from_excel(adhd_paths, output_path, roi_labels_path, min_datasets=2)
        if not table.empty:
            tables["shared_ADHD"] = table
    
    # Shared among ASD cohorts
    asd_datasets = ['abide_asd', 'stanford_asd']
    asd_paths = [count_data_paths.get(d) for d in asd_datasets if count_data_paths.get(d) and os.path.exists(count_data_paths.get(d))]
    
    if len(asd_paths) >= 2:
        output_path = os.path.join(output_dir, "shared_regions_ASD.csv")
        table = create_shared_region_table_from_excel(asd_paths, output_path, roi_labels_path, min_datasets=2)
        if not table.empty:
            tables["shared_ASD"] = table
    
    # Shared among all cohorts
    all_paths = [path for path in count_data_paths.values() if path and os.path.exists(path)]
    if len(all_paths) >= 3:
        output_path = os.path.join(output_dir, "shared_regions_all.csv")
        table = create_shared_region_table_from_excel(all_paths, output_path, roi_labels_path, min_datasets=3)
        if not table.empty:
            tables["shared_all"] = table
    
    logging.info(f"Created {len(tables)} region tables in {output_dir}")
    return tables


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Create region tables from count data Excel files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create all region tables using config file
  python create_region_tables_fixed.py --config config.yaml --output_dir results/region_tables
  
  # Create individual dataset table
  python create_region_tables_fixed.py \\
    --excel_path /path/to/count_data.xlsx \\
    --output_path results/individual_table.csv
        """
    )
    
    parser.add_argument("--config", type=str, 
                       help="Path to configuration YAML file")
    parser.add_argument("--output_dir", type=str, default="results/region_tables",
                       help="Output directory for region tables")
    parser.add_argument("--excel_path", type=str,
                       help="Path to single Excel file (for individual table)")
    parser.add_argument("--output_path", type=str,
                       help="Output path for single table")
    parser.add_argument("--top_n", type=int, default=50,
                       help="Number of top regions to include (default: 50)")
    
    args = parser.parse_args()
    
    if args.excel_path and args.output_path:
        # Create single dataset table
        # Get ROI labels path from config for single dataset processing
        roi_labels_path = config.get('network_analysis', {}).get('roi_labels_path', '')
        table = create_dataset_region_table_from_excel(args.excel_path, args.output_path, roi_labels_path, args.top_n)
        if not table.empty:
            logging.info(f"Successfully created individual region table with {len(table)} regions")
        else:
            logging.error("Failed to create individual region table")
    
    elif args.config:
        # Load configuration and create all tables
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        tables = create_all_region_tables(config, args.output_dir)
        
        logging.info("=== REGION TABLES SUMMARY ===")
        for table_name, table in tables.items():
            logging.info(f"{table_name}: {len(table)} regions")
        logging.info("Region table creation completed!")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
