#!/usr/bin/env python3
"""
Plot brain visualization maps for feature importance.

This script creates 3D brain surface plots and NIfTI visualizations
for feature importance and brain age prediction results.
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
import logging

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / 'utils'))

from plotting_utils import setup_fonts, save_figure

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_brain_feature_map(feature_scores: np.ndarray,
                           roi_labels: List[str],
                           atlas_nifti_path: str,
                           output_path: str,
                           title: str = "Brain Feature Map",
                           percentile: float = 95.0) -> None:
    """
    Create NIfTI brain map from feature importance scores.
    
    Args:
        feature_scores (np.ndarray): Feature importance scores (246 ROIs)
        roi_labels (List[str]): ROI labels
        atlas_nifti_path (str): Path to atlas NIfTI file
        output_path (str): Output path for NIfTI file
        title (str): Title for the map
        percentile (float): Percentile threshold for visualization
    """
    try:
        from nilearn import image, plotting
        
        # Load atlas
        atlas_volume = image.load_img(atlas_nifti_path)
        img_data = atlas_volume.get_fdata()
        
        # Create feature map
        roi_nifti = image.math_img('img-img', img=atlas_volume)
        
        # Apply feature scores to atlas
        for i, score in enumerate(feature_scores):
            roi_idx = i + 1  # ROI indices start from 1
            roi_img = image.math_img(f'img * {score}', img=atlas_volume)
            roi_img = image.math_img(f'img * (img == {roi_idx})', img=roi_img)
            roi_nifti = image.math_img('img1+img2', img1=roi_nifti, img2=roi_img)
        
        # Save NIfTI file
        roi_nifti.to_filename(output_path)
        logging.info(f"Saved brain feature map to: {output_path}")
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot in three views
        display = plotting.plot_stat_map(roi_nifti, display_mode='ortho', 
                                       cut_coords=(0, 0, 0), colorbar=True, 
                                       cmap='inferno', axes=axes, figure=fig)
        
        plt.suptitle(title, fontsize=16)
        
        # Save plot
        plot_path = output_path.replace('.nii.gz', '_plot.png')
        save_figure(plt, plot_path)
        plt.close()
        
    except ImportError:
        logging.warning("nilearn not available, skipping brain visualization")
    except Exception as e:
        logging.error(f"Error creating brain feature map: {e}")


def create_top_features_brain_map(count_data_path: str,
                                roi_labels_path: str,
                                atlas_nifti_path: str,
                                output_dir: str,
                                dataset_name: str,
                                top_n: int = 50) -> None:
    """
    Create brain map for top features from count data.
    
    Args:
        count_data_path (str): Path to count data CSV
        roi_labels_path (str): Path to ROI labels
        atlas_nifti_path (str): Path to atlas NIfTI
        output_dir (str): Output directory
        dataset_name (str): Name of dataset
        top_n (int): Number of top features
    """
    # Load count data
    count_data = pd.read_csv(count_data_path)
    
    # Load ROI labels
    with open(roi_labels_path, 'r') as f:
        roi_labels = [line.strip() for line in f.readlines()]
    
    # Extract ROI indices and get top features
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
    top_features = count_data.nlargest(top_n, 'Count')
    
    # Create feature scores array
    feature_scores = np.zeros(246)  # 246 ROIs
    for _, row in top_features.iterrows():
        roi_idx = row['roi_index']
        if roi_idx is not None and 1 <= roi_idx <= 246:
            feature_scores[roi_idx - 1] = row['Count']  # Convert to 0-based indexing
    
    # Create brain map
    output_path = os.path.join(output_dir, f"{dataset_name}_top_features_brain_map.nii.gz")
    create_brain_feature_map(feature_scores, roi_labels, atlas_nifti_path, 
                           output_path, f"Top {top_n} Features - {dataset_name}")


def create_all_brain_visualizations(config: Dict, output_dir: str = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/figures/brain_visualization") -> None:
    """
    Create brain visualizations for all datasets.
    
    Args:
        config (Dict): Configuration dictionary
        output_dir (str): Output directory for visualizations
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get paths from config
    roi_labels_path = config.get('network_analysis', {}).get('roi_labels_path')
    atlas_nifti_path = config.get('network_analysis', {}).get('atlas_nifti_path')
    count_data_config = config.get('network_analysis', {}).get('count_data', {})
    
    if not all([roi_labels_path, atlas_nifti_path, count_data_config]):
        logging.error("Missing required configuration for brain visualization")
        return
    
    # Create brain maps for each dataset
    for dataset_name, excel_path in count_data_config.items():
        csv_path = f"/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/count_data/{dataset_name}_count_data.csv"
        if os.path.exists(csv_path):
            logging.info(f"Creating brain visualization for {dataset_name}...")
            create_top_features_brain_map(csv_path, roi_labels_path, atlas_nifti_path,
                                        output_dir, dataset_name)
        else:
            logging.warning(f"Count data CSV not found for {dataset_name}: {csv_path}")
    
    logging.info(f"Created brain visualizations in {output_dir}")


def main():
    """Main function for brain visualization plotting."""
    parser = argparse.ArgumentParser(
        description="Create 3D brain surface plots and NIfTI visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create brain visualizations for all datasets
  python plot_brain_visualization.py --config config.yaml
  
  # Create visualizations with custom parameters
  python plot_brain_visualization.py \\
    --config config.yaml \\
    --output_dir custom_brain_plots/ \\
    --top_n 100
        """
    )
    
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to configuration file (default: config.yaml)")
    parser.add_argument("--output_dir", type=str, default="/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/figures/brain_visualization",
                       help="Output directory for plots (default: results/figures/brain_visualization)")
    parser.add_argument("--top_n", type=int, default=50,
                       help="Number of top features to visualize (default: 50)")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    create_all_brain_visualizations(config, args.output_dir)
    logging.info("Brain visualization plotting completed!")


if __name__ == "__main__":
    main()
