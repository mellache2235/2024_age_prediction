#!/usr/bin/env python3
"""
Create heatmap showing PC loadings for top brain regions in brain-behavior analysis.

This script creates a heatmap similar to cognitive measure loadings, but for brain regions
loading on the first 3 PCs from the brain-behavior PCA analysis.

Usage:
    python plot_pc_loadings_heatmap.py --dataset nki_rs_td
    python plot_pc_loadings_heatmap.py --dataset adhd200_td
    python plot_pc_loadings_heatmap.py --dataset cmihbn_td
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import seaborn as sns
from sklearn.decomposition import PCA
import argparse

# Set Arial font
font_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/clustering_analysis/arial.ttf'
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()
else:
    print(f"Warning: Arial font not found at {font_path}, using default font")

# Add utils to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
from logging_utils import (print_section_header, print_step, print_success, 
                           print_warning, print_error, print_info, print_completion)

# ============================================================================
# DATASET CONFIGURATIONS
# ============================================================================

DATASET_CONFIGS = {
    'nki_rs_td': {
        'name': 'NKI-RS TD',
        'ig_csv': '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/integrated_gradients/nki_cog_dev_wIDS_features_IG_convnet_regressor_single_model_fold_0.csv',
        'output_dir': '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/nki_rs_td'
    },
    'adhd200_td': {
        'name': 'ADHD200 TD',
        'ig_csv': '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/integrated_gradients/adhd200_td_features_all_sites_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS_single_model_predictions.csv',
        'output_dir': '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/adhd200_td'
    },
    'cmihbn_td': {
        'name': 'CMI-HBN TD',
        'ig_csv': '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/integrated_gradients/cmihbn_td_features_all_sites_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS_single_model_predictions.csv',
        'output_dir': '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/cmihbn_td'
    }
}


def load_ig_scores(ig_csv_path):
    """Load IG scores from CSV file."""
    print_step("Loading IG scores", f"From {Path(ig_csv_path).name}")
    
    df = pd.read_csv(ig_csv_path)
    
    # Get subject IDs
    if 'subject_id' in df.columns:
        subject_ids = df['subject_id'].values
    elif 'Unnamed: 0' in df.columns:
        subject_ids = df['Unnamed: 0'].values
    else:
        subject_ids = df.index.values
    
    # Get ROI columns (exclude subject_id column)
    roi_cols = [col for col in df.columns if col not in ['subject_id', 'Unnamed: 0']]
    
    # Extract IG matrix
    ig_matrix = df[roi_cols].values
    
    print_info(f"IG subjects: {len(subject_ids)}")
    print_info(f"IG features (ROIs): {len(roi_cols)}")
    
    return ig_matrix, roi_cols, subject_ids


def perform_pca_and_get_loadings(ig_matrix, roi_names, n_components=3):
    """Perform PCA and extract loadings for top components."""
    print_step("Performing PCA", f"Extracting {n_components} components")
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_scores = pca.fit_transform(ig_matrix)
    
    # Get loadings (components)
    loadings = pca.components_  # Shape: (n_components, n_features)
    
    # Calculate variance explained
    var_explained = pca.explained_variance_ratio_
    
    print_info(f"Variance explained by PC1: {var_explained[0]*100:.2f}%")
    print_info(f"Variance explained by PC2: {var_explained[1]*100:.2f}%")
    print_info(f"Variance explained by PC3: {var_explained[2]*100:.2f}%")
    
    return loadings, var_explained


def select_top_regions(loadings, roi_names, n_top=15):
    """
    Select top N regions based on absolute loadings across all PCs.
    
    Args:
        loadings: Array of shape (n_components, n_features)
        roi_names: List of ROI names
        n_top: Number of top regions to select
    
    Returns:
        List of top region names and their indices
    """
    print_step("Selecting top regions", f"Based on absolute loadings across all PCs")
    
    # Calculate max absolute loading for each ROI across all PCs
    max_abs_loadings = np.max(np.abs(loadings), axis=0)
    
    # Get indices of top N regions
    top_indices = np.argsort(max_abs_loadings)[-n_top:][::-1]
    
    # Get top region names
    top_regions = [roi_names[i] for i in top_indices]
    
    print_info(f"Selected {len(top_regions)} regions with highest loadings")
    
    return top_regions, top_indices


def create_loading_heatmap(loadings, roi_names, top_indices, dataset_name, output_dir):
    """Create heatmap of PC loadings for top regions."""
    print_step("Creating PC loadings heatmap", f"For {dataset_name}")
    
    # Extract loadings for top regions
    top_loadings = loadings[:, top_indices].T  # Shape: (n_regions, n_components)
    
    # Get top region names
    top_regions = [roi_names[i] for i in top_indices]
    
    # Shorten region names if too long
    short_names = []
    for name in top_regions:
        # Remove common prefixes
        name = name.replace('IG_', '').replace('ROI_', '')
        # Truncate if too long
        if len(name) > 30:
            name = name[:27] + '...'
        short_names.append(name)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 10), dpi=300)
    
    # Create heatmap with red-blue colormap (like your example)
    sns.heatmap(top_loadings, 
                annot=True, 
                fmt='.3f',
                cmap='RdBu_r',  # Red for positive, blue for negative
                center=0,
                vmin=-1, vmax=1,
                cbar_kws={'label': 'Loading', 'shrink': 0.8},
                linewidths=0.5,
                linecolor='black',
                xticklabels=['PC1', 'PC2', 'PC3'],
                yticklabels=short_names,
                ax=ax)
    
    # Customize
    ax.set_xlabel('Principal Components', fontsize=14, fontweight='bold')
    ax.set_ylabel('Brain Regions', fontsize=14, fontweight='bold')
    ax.set_title(f'{dataset_name}\nTop Brain Region Loadings on First 3 PCs', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Rotate y-axis labels for better readability
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=12)
    
    # Adjust colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label('Loading', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir) / 'pc_loadings_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_success(f"Heatmap saved: {output_path}")
    
    # Also save loadings to CSV
    loadings_df = pd.DataFrame(
        top_loadings,
        index=short_names,
        columns=['PC1', 'PC2', 'PC3']
    )
    csv_path = Path(output_dir) / 'pc_loadings_top_regions.csv'
    loadings_df.to_csv(csv_path)
    print_success(f"Loadings CSV saved: {csv_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Create PC loadings heatmap for brain-behavior analysis')
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=['nki_rs_td', 'adhd200_td', 'cmihbn_td'],
                       help='Dataset to analyze')
    parser.add_argument('--n_top', type=int, default=15,
                       help='Number of top regions to display (default: 15)')
    
    args = parser.parse_args()
    
    # Get dataset config
    config = DATASET_CONFIGS[args.dataset]
    
    print_section_header(f"PC LOADINGS HEATMAP - {config['name']}")
    print_info(f"IG CSV: {config['ig_csv']}")
    print_info(f"Output: {config['output_dir']}")
    
    # Create output directory
    Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
    
    try:
        # Load IG scores
        ig_matrix, roi_names, subject_ids = load_ig_scores(config['ig_csv'])
        
        # Perform PCA and get loadings
        loadings, var_explained = perform_pca_and_get_loadings(ig_matrix, roi_names, n_components=3)
        
        # Select top regions
        top_regions, top_indices = select_top_regions(loadings, roi_names, n_top=args.n_top)
        
        # Create heatmap
        create_loading_heatmap(loadings, roi_names, top_indices, config['name'], config['output_dir'])
        
        print_completion(f"PC loadings heatmap created successfully for {config['name']}!")
        
    except Exception as e:
        print_error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

