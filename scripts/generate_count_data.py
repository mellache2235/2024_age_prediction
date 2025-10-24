#!/usr/bin/env python3
"""
Generate count data CSV from IG scores for network plots.

This script creates a CSV file similar to "yeao_attrib_collapsed_mean_abcd_any_use_y0_all_subjects_top20.csv"
by counting how many times each ROI's IG score magnitude is in the top percentile.

Usage:
    python scripts/generate_count_data.py --ig_csv results/integrated_gradients/nki_rs_td/nki_rs_td_features_IG_convnet_regressor_trained_on_hcp_dev_fold_0.csv --percentile 80 --output yeao_attrib_collapsed_mean_nki_rs_td_top20.csv
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_count_data(ig_csv_file: str, percentile: float = 50, output_file: str = None, 
                       region_names: list = None, use_absolute_values: bool = True):
    """
    Generate count data from IG scores CSV using top 50% of features.
    
    Args:
        ig_csv_file (str): Path to IG scores CSV file
        percentile (float): Percentile threshold for top features (default: 50)
        output_file (str): Output CSV file path
        region_names (list): List of region names
        use_absolute_values (bool): Whether to use absolute values of IG scores
    """
    # Load IG scores
    ig_data = pd.read_csv(ig_csv_file)
    
    # Get ROI columns (exclude metadata columns)
    roi_columns = [col for col in ig_data.columns if col.lower() not in 
                  ['subject_id', 'participant_id', 'subid', 'sub_id', 'id', 'unnamed: 0']]
    
    ig_scores = ig_data[roi_columns].values
    
    if use_absolute_values:
        ig_scores = np.abs(ig_scores)
    
    n_subjects, n_rois = ig_scores.shape
    logging.info(f"Processing {n_subjects} subjects and {n_rois} ROIs")
    
    # Calculate attribution (count of times each ROI is in top percentile)
    attributions = []
    
    for roi_idx in range(n_rois):
        roi_scores = ig_scores[:, roi_idx]
        threshold = np.percentile(roi_scores, percentile)
        count = np.sum(roi_scores >= threshold)
        attribution = count / n_subjects  # Normalize by number of subjects
        attributions.append(attribution)
    
    # Create region names
    if region_names is None:
        region_names = roi_columns  # Use actual column names from CSV
    elif len(region_names) != n_rois:
        logging.warning(f"Number of region names ({len(region_names)}) doesn't match number of ROIs ({n_rois})")
        region_names = roi_columns
    
    # Create DataFrame in the format expected by the R script
    df_plot = pd.DataFrame({
        'attribution': attributions,
        'region': region_names
    })
    
    # Save to CSV
    if output_file:
        df_plot.to_csv(output_file, index=False)
        logging.info(f"Count data saved to: {output_file}")
        logging.info(f"Attribution range: {min(attributions):.4f} - {max(attributions):.4f}")
        logging.info(f"Top 5 regions: {df_plot.nlargest(5, 'attribution')['region'].tolist()}")
    
    return df_plot

def load_yeo_atlas_regions(atlas_file: str = None) -> list:
    """
    Load Yeo atlas region names.
    
    Args:
        atlas_file (str): Path to Yeo atlas CSV file
        
    Returns:
        list: List of region names
    """
    if atlas_file and os.path.exists(atlas_file):
        try:
            atlas_df = pd.read_csv(atlas_file)
            if 'region_name' in atlas_df.columns:
                return atlas_df['region_name'].tolist()
            elif 'name' in atlas_df.columns:
                return atlas_df['name'].tolist()
            else:
                logging.warning(f"Could not find region name column in {atlas_file}")
        except Exception as e:
            logging.warning(f"Error loading atlas file {atlas_file}: {e}")
    
    # Return default ROI names
    return [f'ROI_{i}' for i in range(246)]

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate count data from Integrated Gradients scores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate count data from IG scores
  python generate_count_data.py \\
    --ig_csv /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/integrated_gradients/nki_rs_td/nki_rs_td_features_IG_convnet_regressor_trained_on_hcp_dev_fold_0.csv \\
    --output /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/count_data/nki_rs_td_count_data.csv \\
    --percentile 50
  
  # Generate count data with Yeo atlas region names
  python generate_count_data.py \\
    --ig_csv /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/integrated_gradients/adhd200_adhd/adhd200_adhd_features_IG_convnet_regressor_trained_on_hcp_dev_fold_0.csv \\
    --output /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/count_data/adhd200_adhd_count_data.csv \\
    --atlas_file /path/to/yeo_atlas.csv \\
    --percentile 80
        """
    )
    
    parser.add_argument("--ig_csv", type=str, required=True, 
                       help="Path to IG scores CSV file (generated by compute_integrated_gradients.py)")
    parser.add_argument("--percentile", type=float, default=50, 
                       help="Percentile threshold for top features (default: 50, range: 0-100)")
    parser.add_argument("--output", type=str, required=True, 
                       help="Output CSV file path for count data")
    parser.add_argument("--atlas_file", type=str, 
                       help="Path to Yeo atlas CSV file for region names (optional)")
    parser.add_argument("--no_absolute", action="store_true", 
                       help="Don't use absolute values of IG scores (default: use absolute values)")
    
    args = parser.parse_args()
    
    # Load region names from atlas if provided
    region_names = None
    if args.atlas_file:
        region_names = load_yeo_atlas_regions(args.atlas_file)
    
    # Generate count data
    df_plot = generate_count_data(
        ig_csv_file=args.ig_csv,
        percentile=args.percentile,
        output_file=args.output,
        region_names=region_names,
        use_absolute_values=not args.no_absolute
    )
    
    logging.info("Count data generation completed!")

if __name__ == "__main__":
    main()
