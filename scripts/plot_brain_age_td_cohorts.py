#!/usr/bin/env python3
"""
Create brain age prediction scatter plots for TD cohorts with separate subplots.

This script creates a 2x2 subplot layout for core TD cohorts (HCP-Development, NKI-RS, CMI-HBN, ADHD-200).
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.backends.backend_pdf as pdf
from matplotlib.ticker import MultipleLocator
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
    # HCP-Dev uses most_updated from project root
    hcp_dev_dir = npz_files_dir
    # TD cohorts use oct25 from generalization folders
    nki_dir = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/nki_updated'
    cmihbn_dir = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/cmihbn_updated'
    adhd200_dir = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/adhd200_updated'
    
    td_datasets = {
        'HCP-Development': {
            'predicted': os.path.join(hcp_dev_dir, 'predicted_hcp_dev_ages_most_updated.npz'),
            'actual': os.path.join(hcp_dev_dir, 'actual_hcp_dev_ages_most_updated.npz')
        },
        'NKI-RS': {
            'predicted': os.path.join(nki_dir, 'predicted_nki_ages_oct25.npz'),
            'actual': os.path.join(nki_dir, 'actual_nki_ages_oct25.npz')
        },
        'CMI-HBN': {
            'predicted': os.path.join(cmihbn_dir, 'predicted_cmihbn_td_ages_oct25.npz'),
            'actual': os.path.join(cmihbn_dir, 'actual_cmihbn_td_ages_oct25.npz')
        },
        'ADHD-200': {
            'predicted': os.path.join(adhd200_dir, 'predicted_adhd200_td_ages_oct25.npz'),
            'actual': os.path.join(adhd200_dir, 'actual_adhd200_td_ages_oct25.npz')
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
        predicted_file = dataset_info['predicted']
        actual_file = dataset_info['actual']
        
        try:
            predicted_ages, actual_ages = load_age_data_from_npz(predicted_file, actual_file)
            
            # Calculate metrics for this dataset
            r, p = pearsonr(actual_ages, predicted_ages)
            r_squared = r ** 2
            mae = mean_absolute_error(actual_ages, predicted_ages)
            
            # Plot scatter with darker blue/purple color (matching brain-behavior plots)
            ax.scatter(actual_ages, predicted_ages,
                      color='#0A1281',
                      edgecolors='#0A1281',
                      alpha=0.7, s=100, linewidth=1.2)
            
            # Set axis limits with padding to prevent dots from being cut off
            min_age = min(min(actual_ages), min(predicted_ages))
            max_age = max(max(actual_ages), max(predicted_ages))
            age_range = max_age - min_age
            padding = age_range * 0.05  # 5% padding on each side
            lims = [min_age - padding, max_age + padding]
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            
            # Add regression line (thin indigo to match points)
            z = np.polyfit(actual_ages, predicted_ages, 1)
            p_line = np.poly1d(z)
            ax.plot(lims, p_line(lims), color='#0A1281', alpha=0.9, linewidth=1.6)
            
            # Format p-value (short form)
            if p < 0.001:
                p_text = "< 0.001"
            else:
                p_text = f"= {p:.3f}"
            
            # Add statistics text in bottom right corner (NO bounding box)
            ax.text(0.95, 0.05,
                    f"$R^2$ = {r_squared:.3f}\n"
                    f"MAE = {mae:.2f} years\n"
                    f"P {p_text}",
                    transform=ax.transAxes, fontsize=16,
                    verticalalignment='bottom', horizontalalignment='right')
            
            # Customize subplot
            ax.set_xlabel('Chronological Age (years)', fontsize=16, fontweight='normal')
            ax.set_ylabel('Brain Age (years)', fontsize=16, fontweight='normal')
            ax.set_title(dataset_name, fontsize=18, fontweight='bold', pad=15)
            
            # Clean style - NO top/right spines
            ax.grid(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.0)
            ax.spines['bottom'].set_linewidth(1.0)
            
            # Ensure all ticks are present on both axes
            ax.tick_params(axis='both', which='major', labelsize=16, direction='out',
                          length=6, width=1.0, top=False, right=False)
            ax.minorticks_off()

            # Regular tick marks every 5 years for readability
            locator = MultipleLocator(5)
            ax.xaxis.set_major_locator(locator)
            ax.yaxis.set_major_locator(locator)
            
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
    
    # Save PNG + TIFF + AI
    png_path = output_path
    tiff_path = output_path.replace('.png', '.tiff')
    ai_path = output_path.replace('.png', '.ai')
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(tiff_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
    pdf.FigureCanvas(fig).print_pdf(ai_path)
    
    logging.info(f"Combined TD cohorts plot saved: {png_path} + {tiff_path} + {ai_path}")
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
    
    print_step(1, "CREATING TD COHORTS PLOT", "2x2 layout: HCP-Development, NKI-RS, CMI-HBN, ADHD-200")
    
    # Create the plot
    plot_combined_td_cohorts(args.npz_dir, str(output_path), args.title)
    
    print()
    print_completion("TD Cohorts Brain Age Plot", [f"{output_path}.png"])


if __name__ == "__main__":
    main()
