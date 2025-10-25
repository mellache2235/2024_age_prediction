#!/usr/bin/env python3
"""
Convert Excel count data files to CSV format for network analysis.

This script processes all count data Excel files specified in the configuration
and converts them to CSV format for use in network analysis and visualization.

Usage:
    python scripts/convert_count_data.py
"""

import os
import sys
import yaml
import logging
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / 'utils'))

from count_data_utils import process_all_count_data
from logging_utils import (print_section_header, print_step, print_success, 
                           print_warning, print_error, print_info, print_completion)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Convert all count data Excel files to CSV format."""
    print_section_header("CONVERT COUNT DATA - EXCEL TO CSV")
    
    # Load configuration
    config_path = Path(__file__).parent.parent / 'config.yaml'
    if not config_path.exists():
        print_error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    print_info(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    count_data_config = config.get('network_analysis', {}).get('count_data', {})
    print_info(f"Found {len(count_data_config)} datasets to convert")
    print()
    
    # Process all count data files
    try:
        print_step(1, "CONVERTING EXCEL FILES", "Processing all count data files")
        count_data = process_all_count_data(config)
        
        print()
        print_success(f"Successfully processed {len(count_data)} count data files:")
        for dataset_name, df in count_data.items():
            print_info(f"  {dataset_name}: {len(df)} regions", indent=2)
        
        output_files = [
            "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/count_data/*.csv"
        ]
        print_completion("Count Data Conversion", output_files)
        
    except Exception as e:
        print_error(f"Error processing count data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
