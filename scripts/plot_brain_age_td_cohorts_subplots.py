#!/usr/bin/env python3
"""
Create brain age prediction scatter plots for TD cohorts with separate subplots.

This script creates a 2x3 subplot layout for all TD cohorts, with each dataset
in its own subplot for better visualization and comparison.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error
import logging

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / 'utils'))

from plotting_utils import setup_fonts, save_figure

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_dataset_data(npz_files_dir: str, dataset_info: Dict[str, str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load predicted and actual ages from .npz files for a dataset.
    
    Args:
        npz_files_dir (str): Directory containing .npz files
        dataset_info (Dict[str, str]): Dictionary with 'predicted' and 'actual' file paths
        
    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]: Predicted and actual ages
    """
    predicted_file = os.path.join(npz_files_dir, dataset_info['predicted'])
    actual_file = os.path.join(npz_files_dir, dataset_info['actual'])
    
    try:
        predicted_data = np.load(predicted_file)
        actual_data = np.load(actual_file)
        
        # Handle different .npz file structures
        if 'ages' in predicted_data:
            predicted_ages = predicted_data['ages']
        elif 'predicted_ages' in predicted_data:
            predicted_ages = predicted_data['predicted_ages']
        else:
            # Assume the array is the first (and only) array in the file
            predicted_ages = predicted_data[list(predicted_data.keys())[0]]
        
        if 'ages' in actual_data:
            actual_ages = actual_data['ages']
        elif 'actual_ages' in actual_data:
            actual_ages = actual_data['actual_ages']
        else:
            # Assume the array is the first (and only) array in the file
            actual_ages = actual_data[list(actual_data.keys())[0]]
        
        return predicted_ages, actual_ages
        
    except Exception as e:
        logging.warning(f"Could not load data for {dataset_info['predicted']}: {e}")
        return None, None


def plot_combined_td_cohorts(npz_files_dir: str, output_path: str, 
                           title: str = "TD Cohorts") -> None:
    """
    Create combined scatter plot for all TD cohorts with separate subplots.
    
    Args:
        npz_files_dir (str): Directory containing .npz files
        output_path (str): Output path for the plot
        title (str): Plot title
    """
    setup_fonts()
    
    # Define TD cohort datasets
    td_datasets = {
        'HCP-Dev': {
            'predicted': 'hcp_dev_nested_predicted_ages.npz',
            'actual': 'hcp_dev_nested_actual_ages.npz'
        },
        'ABIDE TD': {
            'predicted': 'predicted_abide_td_ages_most_updated.npz',
            'actual': 'actual_abide_td_ages_most_updated.npz'
        },
        'CMI-HBN TD': {
            'predicted': 'predicted_cmihbn_td_ages.npz',
            'actual': 'actual_cmihbn_td_ages.npz'
        },
        'ADHD200 TD': {
            'predicted': 'predicted_adhd200_td_ages.npz',
            'actual': 'actual_adhd200_td_ages.npz'
        },
        'NKI': {
            'predicted': 'predicted_nki_ages.npz',
            'actual': 'actual_nki_ages.npz'
        },
        'Stanford TD': {
            'predicted': 'predicted_stanford_td_ages_most_updated.npz',
            'actual': 'actual_stanford_td_ages_most_updated.npz'
        }
    }
    
    # Create 2x3 subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=300)
    axes = axes.flatten()
    
    # Collect all data for overall statistics
    all_actual = []
    all_predicted = []
    all_data = []
    
    # Plot each dataset in its own subplot
    for i, (dataset_name, dataset_info) in enumerate(td_datasets.items()):
        ax = axes[i]
        predicted_ages, actual_ages = load_dataset_data(npz_files_dir, dataset_info)
        
        if predicted_ages is not None and actual_ages is not None:
            # Calculate metrics for this dataset
            r_squared = r2_score(actual_ages, predicted_ages)
            mae = mean_absolute_error(actual_ages, predicted_ages)
            r, p = pearsonr(actual_ages, predicted_ages)
            
            # Plot scatter with blue color and blue edge
            ax.scatter(actual_ages, predicted_ages, 
                      color='#1f77b4', 
                      edgecolors='#1f77b4',
                      alpha=0.7, s=50, linewidth=0.5)
            
            # Add identity line
            lims = [min(min(actual_ages), min(predicted_ages)), max(max(actual_ages), max(predicted_ages))]
            ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
            
            # Add regression line
            z = np.polyfit(actual_ages, predicted_ages, 1)
            p_line = np.poly1d(z)
            ax.plot(lims, p_line(lims), 'k-', alpha=0.8, linewidth=2)
            
            # Format p-value
            if p < 0.001:
                p_text = "P < 0.001"
            else:
                p_text = f"P = {p:.3f}"
            
            # Add statistics text
            ax.text(0.05, 0.95,
                    f"$\mathit{{R}}^2 = {r_squared:.3f}$\n"
                    f"$\mathit{{MAE}} = {mae:.2f}$ years\n"
                    f"{p_text}\n"
                    f"N = {len(actual_ages)}",
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            # Customize subplot
            ax.set_xlabel('Chronological Age (years)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Predicted Brain Age (years)', fontsize=10, fontweight='bold')
            ax.set_title(dataset_name, fontsize=12, fontweight='bold', pad=10)
            
            # Remove grid and spines
            ax.grid(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=9)
            
            # Collect data for overall statistics
            all_actual.extend(actual_ages)
            all_predicted.extend(predicted_ages)
            all_data.append({
                'dataset': dataset_name,
                'actual': actual_ages,
                'predicted': predicted_ages,
                'r_squared': r_squared,
                'mae': mae,
                'r': r,
                'p': p,
                'n': len(actual_ages)
            })
            
            logging.info(f"{dataset_name}: R²={r_squared:.3f}, MAE={mae:.2f}, r={r:.3f}, p={p:.3f}, n={len(actual_ages)}")
        else:
            # Hide empty subplot
            ax.set_visible(False)
    
    # Hide the last empty subplot if needed
    if len(td_datasets) < 6:
        axes[-1].set_visible(False)
    
    if not all_actual:
        logging.error("No TD cohort data found!")
        return
    
    # Calculate overall statistics
    overall_r_squared = r2_score(all_actual, all_predicted)
    overall_mae = mean_absolute_error(all_actual, all_predicted)
    overall_r, overall_p = pearsonr(all_actual, all_predicted)
    
    # Add overall title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save the plot in multiple formats including .svg for Affinity Designer
    save_figure(fig, output_path, formats=['png', 'pdf', 'svg'])
    logging.info(f"Combined TD cohorts plot saved to: {output_path}")
    logging.info(f"Overall TD cohorts - R²: {overall_r_squared:.3f}, MAE: {overall_mae:.2f} years, r: {overall_r:.3f}, p: {overall_p:.3f}, N: {len(all_actual)}")


def main():
    """Main function for creating combined TD cohorts brain age prediction plot."""
    parser = argparse.ArgumentParser(
        description="Create combined brain age prediction scatter plot for TD cohorts with subplots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create combined TD cohorts plot
  python plot_brain_age_td_cohorts_subplots.py --npz_dir . --output_dir results/brain_age_plots
  
  # Create plot with custom title
  python plot_brain_age_td_cohorts_subplots.py --npz_dir . --output_dir results/brain_age_plots --title "TD Cohorts Brain Age Prediction"
        """
    )
    
    parser.add_argument('--npz_dir', type=str, default='.', help='Directory containing .npz files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for plots')
    parser.add_argument('--title', type=str, default='TD Cohorts', help='Plot title')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output path
    output_path = output_dir / "td_cohorts_combined_scatter"
    
    # Create the plot
    plot_combined_td_cohorts(args.npz_dir, str(output_path), args.title)


if __name__ == "__main__":
    main()
