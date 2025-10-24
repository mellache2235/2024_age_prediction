"""
Count data utilities for processing Excel files and generating network plots.

This module provides functions for converting Excel count data to CSV format
and processing count data for network analysis and visualization.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def convert_excel_to_csv(excel_path: str, csv_path: str, 
                        sheet_name: Optional[str] = None) -> None:
    """
    Convert Excel file to CSV format, preserving all original columns.
    
    Args:
        excel_path (str): Path to the Excel file
        csv_path (str): Path to save the CSV file
        sheet_name (str, optional): Sheet name to read (default: first sheet)
    """
    try:
        # Read Excel file
        if sheet_name:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
        else:
            df = pd.read_excel(excel_path)
        
        # Save as CSV with all original columns
        df.to_csv(csv_path, index=False)
        logger.info(f"Converted {excel_path} to {csv_path}")
        logger.info(f"CSV contains {len(df)} rows and {len(df.columns)} columns: {list(df.columns)}")
        
    except Exception as e:
        logger.error(f"Error converting {excel_path} to CSV: {e}")
        raise


def load_count_data(excel_path: str, 
                   use_adhd_asd_class: bool = False,
                   sheet_name: Optional[str] = None) -> pd.DataFrame:
    """
    Load count data from Excel file and process for network analysis.
    
    Args:
        excel_path (str): Path to the Excel file
        use_adhd_asd_class (bool): If True, use ADHD/ASD class counts for non-normative data
        sheet_name (str, optional): Sheet name to read (default: first sheet)
    
    Returns:
        pd.DataFrame: Processed count data with columns ['Region', 'Count']
    """
    try:
        # Read Excel file
        if sheet_name:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
        else:
            df = pd.read_excel(excel_path)
        
        # Expected columns: Region ID, Gyrus, Description, Region Alias, (ID) Region Label, Count
        expected_columns = ['Region ID', 'Gyrus', 'Description', 'Region Alias', '(ID) Region Label', 'Count']
        
        # Check if all expected columns exist
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing columns in {excel_path}: {missing_columns}")
            logger.info(f"Available columns: {list(df.columns)}")
        
        # Use the available columns
        if 'Count' in df.columns:
            count_col = 'Count'
        else:
            # Look for alternative count column names
            count_cols = [col for col in df.columns if 'count' in col.lower() or 'Count' in col]
            if count_cols:
                count_col = count_cols[0]
                logger.info(f"Using count column: {count_col}")
            else:
                raise ValueError(f"No count column found in {excel_path}")
        
        if 'Region ID' in df.columns:
            region_col = 'Region ID'
        elif 'Region' in df.columns:
            region_col = 'Region'
        elif '(ID) Region Label' in df.columns:
            region_col = '(ID) Region Label'
        else:
            # Look for alternative region column names
            region_cols = [col for col in df.columns if 'region' in col.lower() or 'Region' in col]
            if region_cols:
                region_col = region_cols[0]
                logger.info(f"Using region column: {region_col}")
            else:
                raise ValueError(f"No region column found in {excel_path}")
        
        # Create processed dataframe
        processed_df = pd.DataFrame({
            'Region': df[region_col],
            'Count': df[count_col]
        })
        
        # Remove any rows with missing values
        processed_df = processed_df.dropna()
        
        # Convert count to numeric if it's not already
        processed_df['Count'] = pd.to_numeric(processed_df['Count'], errors='coerce')
        processed_df = processed_df.dropna()
        
        # Sort by count in descending order
        processed_df = processed_df.sort_values('Count', ascending=False)
        
        logger.info(f"Loaded count data: {len(processed_df)} regions from {excel_path}")
        
        return processed_df
        
    except Exception as e:
        logger.error(f"Error loading count data from {excel_path}: {e}")
        raise


def process_all_count_data(config: Dict, output_dir: str = "results/count_data") -> Dict[str, pd.DataFrame]:
    """
    Process all count data files from configuration.
    
    Args:
        config (Dict): Configuration dictionary containing count data paths
        output_dir (str): Directory to save processed CSV files
    
    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping dataset names to processed count data
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    count_data = {}
    count_data_config = config.get('network_analysis', {}).get('count_data', {})
    
    for dataset_name, excel_path in count_data_config.items():
        try:
            logger.info(f"Converting {dataset_name} from {excel_path}")
            
            # Convert Excel to CSV (preserving all columns)
            csv_path = Path(output_dir) / f"{dataset_name}_count_data.csv"
            logger.info(f"Saving CSV to: {csv_path}")
            convert_excel_to_csv(excel_path, str(csv_path))
            
            # Load the converted CSV to verify and return
            df = pd.read_csv(csv_path)
            count_data[dataset_name] = df
            
            logger.info(f"✅ Converted count data for {dataset_name}: {len(df)} regions, {len(df.columns)} columns")
            logger.info(f"✅ CSV saved to: {csv_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to convert count data for {dataset_name}: {e}")
            continue
    
    return count_data


def get_top_regions_by_count(count_data: pd.DataFrame, 
                           top_n: int = 50,
                           min_count: Optional[int] = None) -> pd.DataFrame:
    """
    Get top N regions by count.
    
    Args:
        count_data (pd.DataFrame): Count data with 'Region' and 'Count' columns
        top_n (int): Number of top regions to return
        min_count (int, optional): Minimum count threshold
    
    Returns:
        pd.DataFrame: Top N regions by count
    """
    # Apply minimum count filter if specified
    if min_count is not None:
        filtered_data = count_data[count_data['Count'] >= min_count]
    else:
        filtered_data = count_data
    
    # Get top N regions
    top_regions = filtered_data.head(top_n)
    
    return top_regions


def normalize_counts(count_data: pd.DataFrame, 
                    method: str = 'max') -> pd.DataFrame:
    """
    Normalize count data.
    
    Args:
        count_data (pd.DataFrame): Count data with 'Region' and 'Count' columns
        method (str): Normalization method ('max', 'sum', 'zscore')
    
    Returns:
        pd.DataFrame: Normalized count data
    """
    df = count_data.copy()
    
    if method == 'max':
        df['Normalized_Count'] = df['Count'] / df['Count'].max()
    elif method == 'sum':
        df['Normalized_Count'] = df['Count'] / df['Count'].sum()
    elif method == 'zscore':
        df['Normalized_Count'] = (df['Count'] - df['Count'].mean()) / df['Count'].std()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return df


def create_region_mapping(roi_labels_path: str) -> Dict[int, str]:
    """
    Create mapping from ROI index to region name.
    
    Args:
        roi_labels_path (str): Path to ROI labels text file
    
    Returns:
        Dict[int, str]: Mapping from ROI index to region name
    """
    try:
        with open(roi_labels_path, 'r') as f:
            roi_labels = [line.strip() for line in f.readlines()]
        
        # Create mapping (assuming 1-based indexing)
        roi_mapping = {i+1: label for i, label in enumerate(roi_labels)}
        
        logger.info(f"Created ROI mapping: {len(roi_mapping)} regions")
        return roi_mapping
        
    except Exception as e:
        logger.error(f"Error creating ROI mapping from {roi_labels_path}: {e}")
        raise
