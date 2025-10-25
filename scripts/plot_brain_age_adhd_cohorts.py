#!/usr/bin/env python3
"""
Create brain age prediction scatter plots for ADHD cohorts with separate subplots.

This script creates a 1x2 subplot layout for ADHD cohorts, with each dataset
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


def load_age_data_from_npz(predicted_file: str, actual_file: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load predicted and actual ages from .npz files.
    
    Args:
        predicted_file (str): Path to predicted ages .npz file
        actual_file (str): Path to actual ages .npz file
        
    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]: Predicted and actual ages
    """
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
        logging.warning(f"Could not load data from {predicted_file} or {actual_file}: {e}")
        return None, None


def plot_combined_adhd_cohorts(npz_files_dir: str, output_path: str, 
                             title: str = "ADHD Cohorts") -> None:
    """
    Create combined scatter plot for all ADHD cohorts with separate subplots.
    
    Args:
        npz_files_dir (str): Directory containing .npz files
        output_path (str): Output path for the plot
        title (str): Plot title
    """
    setup_fonts()
    
    # Define ADHD cohort datasets
    adhd_datasets = {
        'CMI-HBN ADHD': {
            'predicted': 'predicted_cmihbn_adhd_ages_most_updated.npz',
            'actual': 'actual_cmihbn_adhd_ages_most_updated.npz'
        },
        'ADHD200 ADHD': {
            'predicted': 'predicted_adhd200_ages_most_updated.npz',
            'actual': 'actual_adhd200_ages_most_updated.npz'
        }
    }
    
    # Create 1x2 subplot layout
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
    
    # Collect all data for overall statistics
    all_actual = []
    all_predicted = []
    all_data = []
    
    # Plot each dataset in its own subplot
    for i, (dataset_name, dataset_info) in enumerate(adhd_datasets.items()):
        ax = axes[i]
        predicted_file = os.path.join(npz_files_dir, dataset_info['predicted'])
        actual_file = os.path.join(npz_files_dir, dataset_info['actual'])
        
        try:
            predicted_ages, actual_ages = load_age_data_from_npz(predicted_file, actual_file)
            
            # Calculate metrics for this dataset
            r, p = pearsonr(actual_ages, predicted_ages)
            r_squared = r ** 2
            mae = mean_absolute_error(actual_ages, predicted_ages)
            
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
            
            # Format p-value (short form)
            if p < 0.001:
                p_text = "< 0.001"
            else:
                p_text = f"= {p:.3f}"
            
            # Add statistics text in bottom right corner
            ax.text(0.95, 0.05,
                    f"$\mathit{{R}}^2 = {r_squared:.3f}$\n"
                    f"$\mathit{{MAE}} = {mae:.2f}$ years\n"
                    f"$\mathit{{P}}$ {p_text}\n"
                    f"N = {len(actual_ages)}",
                    transform=ax.transAxes, fontsize=11, 
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            # Customize subplot
            ax.set_xlabel('Chronological Age (years)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Predicted Brain Age (years)', fontsize=11, fontweight='bold')
            ax.set_title(dataset_name, fontsize=13, fontweight='bold', pad=15)
            
            # Remove grid and spines
            ax.grid(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=10)
            
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
            
        except Exception as e:
            logging.warning(f"Could not load data for {dataset_name}: {e}")
            # Hide empty subplot
            ax.set_visible(False)
    
    if not all_actual:
        logging.error("No ADHD cohort data found!")
        return
    
    # Calculate overall statistics
    overall_r, overall_p = pearsonr(all_actual, all_predicted)
    overall_r_squared = overall_r ** 2
    overall_mae = mean_absolute_error(all_actual, all_predicted)
    
    # Add overall title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    # Save the plot in PNG format only
    save_figure(fig, output_path, formats=['png'])
    logging.info(f"Combined ADHD cohorts plot saved to: {output_path}")
    logging.info(f"Overall ADHD cohorts - R²: {overall_r_squared:.3f}, MAE: {overall_mae:.2f} years, r: {overall_r:.3f}, p: {overall_p:.3f}, N: {len(all_actual)}")


def main():
    """Main function for creating combined ADHD cohorts brain age prediction plot."""
    parser = argparse.ArgumentParser(
        description="Create combined brain age prediction scatter plot for ADHD cohorts with subplots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create combined ADHD cohorts plot
  python plot_brain_age_adhd_cohorts.py --npz_dir . --output_dir results/brain_age_plots
  
  # Create plot with custom title
  python plot_brain_age_adhd_cohorts.py --npz_dir . --output_dir results/brain_age_plots --title "ADHD Cohorts Brain Age Prediction"
        """
    )
    
    parser.add_argument('--npz_dir', type=str, default='../results/brain_age_predictions/npz_files', help='Directory containing .npz files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for plots')
    parser.add_argument('--title', type=str, default='ADHD Cohorts', help='Plot title')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output path
    output_path = output_dir / "adhd_cohorts_combined_scatter"
    
    # Create the plot
    plot_combined_adhd_cohorts(args.npz_dir, str(output_path), args.title)


if __name__ == "__main__":
    main()