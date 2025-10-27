#!/usr/bin/env python3
"""
Create brain age prediction scatter plots for TD cohorts with separate subplots.

This script creates a 2x2 subplot layout for core TD cohorts (HCP-Dev, NKI, CMI-HBN TD, ADHD200 TD).
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error
import logging

# Set Arial font
font_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/clustering_analysis/arial.ttf'
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()
else:
    print(f"Warning: Arial font not found at {font_path}, using default font")

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / 'utils'))

from plotting_utils import setup_fonts, save_figure
from logging_utils import (print_section_header, print_step, print_success, 
                           print_warning, print_error, print_info, print_completion)

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


def plot_combined_td_cohorts(npz_files_dir: str, output_path: str, 
                           title: str = "TD Cohorts") -> None:
    """
    Create combined scatter plot for core TD cohorts with 2x2 subplot layout.
    
    Args:
        npz_files_dir (str): Directory containing .npz files
        output_path (str): Output path for the plot
        title (str): Plot title
    """
    setup_fonts()
    
    # Define core TD cohort datasets (4 datasets only)
    # All use bias-corrected predictions
    td_datasets = {
        'HCP-Dev': {
            'predicted': 'predicted_hcp_dev_ages_most_updated.npz',
            'actual': 'actual_hcp_dev_ages_most_updated.npz'
        },
        'NKI': {
            'predicted': 'predicted_nki_ages_oct25.npz',
            'actual': 'actual_nki_ages_oct25.npz'
        },
        'CMI-HBN TD': {
            'predicted': 'predicted_cmihbn_td_ages_oct25.npz',
            'actual': 'actual_cmihbn_td_ages_oct25.npz'
        },
        'ADHD200 TD': {
            'predicted': 'predicted_adhd200_td_ages_oct25.npz',
            'actual': 'actual_adhd200_td_ages_oct25.npz'
        }
    }
    
    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=300)
    axes = axes.flatten()
    
    # Collect all data for overall statistics
    all_actual = []
    all_predicted = []
    all_data = []
    
    # Plot each dataset in its own subplot
    for i, (dataset_name, dataset_info) in enumerate(td_datasets.items()):
        ax = axes[i]
        predicted_file = os.path.join(npz_files_dir, dataset_info['predicted'])
        actual_file = os.path.join(npz_files_dir, dataset_info['actual'])
        
        try:
            predicted_ages, actual_ages = load_age_data_from_npz(predicted_file, actual_file)
            
            # Calculate metrics for this dataset
            r, p = pearsonr(actual_ages, predicted_ages)
            r_squared = r ** 2
            mae = mean_absolute_error(actual_ages, predicted_ages)
            
            # Plot scatter with darker blue/purple color (matching brain-behavior plots)
            ax.scatter(actual_ages, predicted_ages, 
                      color='#5A6FA8', 
                      edgecolors='#5A6FA8',
                      alpha=0.7, s=80, linewidth=1)
            
            # Set axis limits with padding to prevent dots from being cut off
            min_age = min(min(actual_ages), min(predicted_ages))
            max_age = max(max(actual_ages), max(predicted_ages))
            age_range = max_age - min_age
            padding = age_range * 0.05  # 5% padding on each side
            lims = [min_age - padding, max_age + padding]
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            
            # Add regression line (red, matching brain-behavior plots)
            z = np.polyfit(actual_ages, predicted_ages, 1)
            p_line = np.poly1d(z)
            ax.plot(lims, p_line(lims), color='#D32F2F', alpha=0.9, linewidth=2.5)
            
            # Format p-value (short form)
            if p < 0.001:
                p_text = "< 0.001"
            else:
                p_text = f"= {p:.3f}"
            
            # Add statistics text in bottom right corner (matching brain-behavior style)
            ax.text(0.95, 0.05,
                    f"$\mathit{{R}}^2 = {r_squared:.3f}$\n"
                    f"$\mathit{{MAE}} = {mae:.2f}$ years\n"
                    f"$\mathit{{P}}$ {p_text}\n"
                    f"N = {len(actual_ages)}",
                    transform=ax.transAxes, fontsize=14, 
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=1))
            
            # Customize subplot
            ax.set_xlabel('Chronological Age (years)', fontsize=14, fontweight='normal')
            ax.set_ylabel('Predicted Brain Age (years)', fontsize=14, fontweight='normal')
            ax.set_title(dataset_name, fontsize=16, fontweight='bold', pad=15)
            
            # Clean style with all spines visible
            ax.grid(False)
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['right'].set_linewidth(1.5)
            ax.spines['top'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)
            ax.tick_params(axis='both', which='major', labelsize=12, direction='out', length=6, width=1.5)
            
            # Add tick marks
            ax.tick_params(axis='both', which='major', labelsize=9, 
                          direction='out', length=4, width=1)
            
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
        logging.error("No TD cohort data found!")
        return
    
    # Calculate overall statistics
    overall_r, overall_p = pearsonr(all_actual, all_predicted)
    overall_r_squared = overall_r ** 2
    overall_mae = mean_absolute_error(all_actual, all_predicted)
    
    # Add overall title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save the plot in PNG format only
    save_figure(fig, output_path, formats=['png'])
    logging.info(f"Combined TD cohorts plot saved to: {output_path}")
    logging.info(f"Overall TD cohorts - R²: {overall_r_squared:.3f}, MAE: {overall_mae:.2f} years, r: {overall_r:.3f}, p: {overall_p:.3f}, N: {len(all_actual)}")


def main():
    """Main function for creating combined TD cohorts brain age prediction plot."""
    parser = argparse.ArgumentParser(
        description="Create combined brain age prediction scatter plot for TD cohorts with 2x2 subplots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create combined TD cohorts plot
  python plot_brain_age_td_cohorts.py --npz_dir . --output_dir results/brain_age_plots
  
  # Create plot with custom title
  python plot_brain_age_td_cohorts.py --npz_dir . --output_dir results/brain_age_plots --title "TD Cohorts Brain Age Prediction"
        """
    )
    
    parser.add_argument('--npz_dir', type=str, default='/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_age_predictions/npz_files', help='Directory containing .npz files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for plots')
    parser.add_argument('--title', type=str, default='TD Cohorts', help='Plot title')
    
    args = parser.parse_args()
    
    print_section_header("BRAIN AGE PREDICTION - TD COHORTS")
    
    print_info(f"NPZ files directory: {args.npz_dir}")
    print_info(f"Output directory: {args.output_dir}")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output path
    output_path = output_dir / "td_cohorts_combined_scatter"
    
    print_step(1, "CREATING TD COHORTS PLOT", "2x2 layout: HCP-Dev, NKI, CMI-HBN TD, ADHD200 TD")
    
    # Create the plot
    plot_combined_td_cohorts(args.npz_dir, str(output_path), args.title)
    
    print()
    print_completion("TD Cohorts Brain Age Plot", [f"{output_path}.png"])


if __name__ == "__main__":
    main()
