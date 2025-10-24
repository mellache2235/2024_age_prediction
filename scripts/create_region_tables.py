#!/usr/bin/env python3
"""
Create tables for regions of importance based on count data from CSV files.

This script generates comprehensive tables summarizing important brain regions
across different datasets and cohorts using the count data CSV files.
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

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / 'utils'))

from count_data_utils import load_count_data, create_region_mapping

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_dataset_region_table(count_csv_path: str, 
                               roi_labels_path: str,
                               output_path: str,
                               top_n: int = 50) -> pd.DataFrame:
    """
    Create a region table for a single dataset.
    
    Args:
        count_csv_path (str): Path to count data CSV file
        roi_labels_path (str): Path to ROI labels file
        output_path (str): Output path for the table
        top_n (int): Number of top regions to include
        
    Returns:
        pd.DataFrame: Region table
    """
    # Load count data
    count_data = pd.read_csv(count_csv_path)
    
    # Load ROI labels
    roi_mapping = create_region_mapping(roi_labels_path)
    
    # Extract ROI index from region names
    def extract_roi_index(region_name):
        try:
            if isinstance(region_name, str):
                import re
                numbers = re.findall(r'\d+', region_name)
                if numbers:
                    return int(numbers[0])
            elif isinstance(region_name, (int, float)):
                return int(region_name)
        except:
            pass
        return None
    
    count_data['roi_index'] = count_data['region'].apply(extract_roi_index)
    
    # Map ROI indices to region names
    count_data['region_name'] = count_data['roi_index'].map(roi_mapping)
    
    # Sort by count and get top N
    top_regions = count_data.nlargest(top_n, 'Count')
    
    # Create final table
    region_table = pd.DataFrame({
        'ROI_Index': top_regions['roi_index'],
        'Region_Name': top_regions['region_name'],
        'Count': top_regions['Count'],
        'Rank': range(1, len(top_regions) + 1)
    })
    
    # Save table
    region_table.to_csv(output_path, index=False)
    logging.info(f"Created region table for dataset: {len(region_table)} regions saved to {output_path}")
    
    return region_table


def create_shared_region_table(dataset_csv_paths: List[str],
                              roi_labels_path: str,
                              output_path: str,
                              top_n: int = 50,
                              min_datasets: int = 2) -> pd.DataFrame:
    """
    Create a table of regions shared across multiple datasets.
    
    Args:
        dataset_csv_paths (List[str]): List of paths to count data CSV files
        roi_labels_path (str): Path to ROI labels file
        output_path (str): Output path for the table
        top_n (int): Number of top regions to include per dataset
        min_datasets (int): Minimum number of datasets a region must appear in
        
    Returns:
        pd.DataFrame: Shared regions table
    """
    # Load ROI labels
    roi_mapping = create_region_mapping(roi_labels_path)
    
    # Collect top regions from each dataset
    all_regions = {}
    
    for csv_path in dataset_csv_paths:
        if not os.path.exists(csv_path):
            logging.warning(f"Count data file not found: {csv_path}")
            continue
            
        dataset_name = Path(csv_path).stem.replace('_count_data', '')
        count_data = pd.read_csv(csv_path)
        
        # Extract ROI index
        def extract_roi_index(region_name):
            try:
                if isinstance(region_name, str):
                    import re
                    numbers = re.findall(r'\d+', region_name)
                    if numbers:
                        return int(numbers[0])
                elif isinstance(region_name, (int, float)):
                    return int(region_name)
            except:
                pass
            return None
        
        count_data['roi_index'] = count_data['region'].apply(extract_roi_index)
        
        # Get top N regions
        top_regions = count_data.nlargest(top_n, 'Count')
        
        for _, row in top_regions.iterrows():
            roi_idx = row['roi_index']
            if roi_idx is not None:
                if roi_idx not in all_regions:
                    all_regions[roi_idx] = {
                        'region_name': roi_mapping.get(roi_idx, f'ROI_{roi_idx}'),
                        'datasets': [],
                        'counts': [],
                        'total_count': 0
                    }
                all_regions[roi_idx]['datasets'].append(dataset_name)
                all_regions[roi_idx]['counts'].append(row['Count'])
                all_regions[roi_idx]['total_count'] += row['Count']
    
    # Filter regions that appear in at least min_datasets
    shared_regions = {roi_idx: data for roi_idx, data in all_regions.items() 
                     if len(data['datasets']) >= min_datasets}
    
    # Create table
    table_data = []
    for roi_idx, data in shared_regions.items():
        table_data.append({
            'ROI_Index': roi_idx,
            'Region_Name': data['region_name'],
            'N_Datasets': len(data['datasets']),
            'Datasets': ', '.join(data['datasets']),
            'Total_Count': data['total_count'],
            'Mean_Count': np.mean(data['counts']),
            'Max_Count': np.max(data['counts'])
        })
    
    # Sort by total count
    shared_table = pd.DataFrame(table_data)
    shared_table = shared_table.sort_values('Total_Count', ascending=False)
    shared_table['Rank'] = range(1, len(shared_table) + 1)
    
    # Save table
    shared_table.to_csv(output_path, index=False)
    logging.info(f"Created shared region table: {len(shared_table)} regions shared across datasets, saved to {output_path}")
    
    return shared_table


def create_all_region_tables(config: Dict, output_dir: str = "results/region_tables") -> Dict[str, pd.DataFrame]:
    """
    Create all region tables for different dataset groups.
    
    Args:
        config (Dict): Configuration dictionary
        output_dir (str): Output directory for tables
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary of created tables
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get paths from config
    roi_labels_path = config.get('network_analysis', {}).get('roi_labels_path')
    count_data_config = config.get('network_analysis', {}).get('count_data', {})
    
    if not roi_labels_path or not count_data_config:
        logging.error("Missing required configuration: roi_labels_path or count_data")
        return {}
    
    tables = {}
    
    # Create individual dataset tables
    logging.info("Creating individual dataset region tables...")
    for dataset_name, excel_path in count_data_config.items():
        csv_path = f"results/count_data/{dataset_name}_count_data.csv"
        if os.path.exists(csv_path):
            output_path = os.path.join(output_dir, f"{dataset_name}_top_regions.csv")
            table = create_dataset_region_table(csv_path, roi_labels_path, output_path)
            tables[f"{dataset_name}_individual"] = table
        else:
            logging.warning(f"Count data CSV not found for {dataset_name}: {csv_path}")
    
    # Create shared TD regions table
    td_datasets = ['nki', 'adhd200_td', 'cmihbn_td']
    td_csv_paths = [f"results/count_data/{dataset}_count_data.csv" for dataset in td_datasets]
    td_csv_paths = [path for path in td_csv_paths if os.path.exists(path)]
    
    if td_csv_paths:
        logging.info("Creating shared TD regions table...")
        output_path = os.path.join(output_dir, "shared_td_regions.csv")
        td_table = create_shared_region_table(td_csv_paths, roi_labels_path, output_path)
        tables['shared_td'] = td_table
    
    # Create shared ADHD regions table
    adhd_datasets = ['adhd200_adhd', 'cmihbn_adhd']
    adhd_csv_paths = [f"results/count_data/{dataset}_count_data.csv" for dataset in adhd_datasets]
    adhd_csv_paths = [path for path in adhd_csv_paths if os.path.exists(path)]
    
    if adhd_csv_paths:
        logging.info("Creating shared ADHD regions table...")
        output_path = os.path.join(output_dir, "shared_adhd_regions.csv")
        adhd_table = create_shared_region_table(adhd_csv_paths, roi_labels_path, output_path)
        tables['shared_adhd'] = adhd_table
    
    # Create shared ASD regions table
    asd_datasets = ['abide_asd', 'stanford_asd']
    asd_csv_paths = [f"results/count_data/{dataset}_count_data.csv" for dataset in asd_datasets]
    asd_csv_paths = [path for path in asd_csv_paths if os.path.exists(path)]
    
    if asd_csv_paths:
        logging.info("Creating shared ASD regions table...")
        output_path = os.path.join(output_dir, "shared_asd_regions.csv")
        asd_table = create_shared_region_table(asd_csv_paths, roi_labels_path, output_path)
        tables['shared_asd'] = asd_table
    
    # Create overall shared regions table (across all datasets)
    all_csv_paths = [f"results/count_data/{dataset}_count_data.csv" for dataset in count_data_config.keys()]
    all_csv_paths = [path for path in all_csv_paths if os.path.exists(path)]
    
    if all_csv_paths:
        logging.info("Creating overall shared regions table...")
        output_path = os.path.join(output_dir, "shared_all_regions.csv")
        all_table = create_shared_region_table(all_csv_paths, roi_labels_path, output_path, min_datasets=3)
        tables['shared_all'] = all_table
    
    logging.info(f"Created {len(tables)} region tables in {output_dir}")
    return tables


def main():
    """Main function to create region tables."""
    parser = argparse.ArgumentParser(description="Create region tables from count data")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default="results/region_tables",
                       help="Output directory for tables")
    parser.add_argument("--top_n", type=int, default=50,
                       help="Number of top regions to include")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create all region tables
    tables = create_all_region_tables(config, args.output_dir)
    
    # Print summary
    logging.info("=== REGION TABLES SUMMARY ===")
    for table_name, table in tables.items():
        logging.info(f"{table_name}: {len(table)} regions")
    
    logging.info("Region table creation completed!")


if __name__ == "__main__":
    main()