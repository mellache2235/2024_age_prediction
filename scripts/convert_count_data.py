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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Convert all count data Excel files to CSV format."""
    logging.info("Starting count data conversion...")
    
    # Load configuration
    config_path = Path(__file__).parent.parent / 'config.yaml'
    if not config_path.exists():
        logging.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Process all count data files
    try:
        count_data = process_all_count_data(config)
        
        logging.info(f"Successfully processed {len(count_data)} count data files:")
        for dataset_name, df in count_data.items():
            logging.info(f"  - {dataset_name}: {len(df)} regions")
        
        logging.info("Count data conversion completed successfully!")
        
    except Exception as e:
        logging.error(f"Error processing count data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
