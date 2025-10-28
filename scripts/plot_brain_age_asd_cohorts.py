#!/usr/bin/env python3
"""
Create brain age prediction scatter plots for ASD cohorts with separate subplots.

This script creates a 1x2 subplot layout for ASD cohorts, with each dataset
in its own subplot for better visualization and comparison.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.backends.backend_pdf as pdf
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

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / 'utils'))

from plotting_utils import setup_fonts, save_figure
from logging_utils import (print_section_header, print_step, print_success, 
                           print_info, print_completion)

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


def plot_combined_asd_cohorts(npz_files_dir: str, output_path: str, 
                             title: str = "ASD Cohorts") -> None:
    """
    Create combined scatter plot for all ASD cohorts with separate subplots.
    
    Args:
        npz_files_dir (str): Directory containing .npz files
        output_path (str): Output path for the plot
        title (str): Plot title
    """
    setup_fonts()
    
    # Define ASD cohort datasets
    asd_datasets = {
        'ABIDE ASD': {
            'predicted': 'predicted_abide_asd_ages_most_updated.npz',
            'actual': 'actual_abide_asd_ages_most_updated.npz'
        },
        'Stanford ASD': {
            'predicted': 'predicted_stanford_asd_ages_most_updated.npz',
            'actual': 'actual_stanford_asd_ages_most_updated.npz'
        }
    }
    
    # Create 1x2 subplot layout
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
    
    # Collect all data for overall statistics
    all_actual = []
    all_predicted = []
    all_data = []
    
    # Plot each dataset in its own subplot
    for i, (dataset_name, dataset_info) in enumerate(asd_datasets.items()):
        ax = axes[i]
        predicted_file = os.path.join(npz_files_dir, dataset_info['predicted'])
        actual_file = os.path.join(npz_files_dir, dataset_info['actual'])
        
        try:
            predicted_ages, actual_ages = load_age_data_from_npz(predicted_file, actual_file)
            
            # Calculate metrics for this dataset
            r, p = pearsonr(actual_ages, predicted_ages)
            r_squared = r ** 2
            mae = mean_absolute_error(actual_ages, predicted_ages)
            
            # Plot scatter with specific blue color (#5A6FA8)
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
            
            # Add statistics text in bottom right corner (NO bounding box)
            ax.text(0.95, 0.05,
                    f"$R^2$ = {r_squared:.3f}\n"
                    f"MAE = {mae:.2f} years\n"
                    f"P {p_text}\n"
                    f"N = {len(actual_ages)}",
                    transform=ax.transAxes, fontsize=14, 
                    verticalalignment='bottom', horizontalalignment='right')
            
            # Customize subplot
            ax.set_xlabel('Chronological Age (years)', fontsize=14, fontweight='normal')
            ax.set_ylabel('Predicted Brain Age (years)', fontsize=14, fontweight='normal')
            ax.set_title(dataset_name, fontsize=16, fontweight='bold', pad=15)
            
            # Clean style - NO top/right spines
            ax.grid(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)
            
            # Tick styling - major ticks only (no minor ticks)
            ax.tick_params(axis='both', which='major', labelsize=12, direction='out', 
                          length=6, width=1.5, top=False, right=False)
            
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
        logging.error("No ASD cohort data found!")
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
    
    # Save PNG + TIFF + AI
    png_path = output_path
    tiff_path = output_path.replace('.png', '.tiff')
    ai_path = output_path.replace('.png', '.ai')
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(tiff_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
    pdf.FigureCanvas(fig).print_pdf(ai_path)
    
    logging.info(f"Combined ASD cohorts plot saved: {png_path} + {tiff_path} + {ai_path}")
    logging.info(f"Overall ASD cohorts - R²: {overall_r_squared:.3f}, MAE: {overall_mae:.2f} years, r: {overall_r:.3f}, p: {overall_p:.3f}, N: {len(all_actual)}")


def main():
    """Main function for creating combined ASD cohorts brain age prediction plot."""
    parser = argparse.ArgumentParser(
        description="Create combined brain age prediction scatter plot for ASD cohorts with subplots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create combined ASD cohorts plot
  python plot_brain_age_asd_cohorts.py --npz_dir . --output_dir results/brain_age_plots
  
  # Create plot with custom title
  python plot_brain_age_asd_cohorts.py --npz_dir . --output_dir results/brain_age_plots --title "ASD Cohorts Brain Age Prediction"
        """
    )
    
    parser.add_argument('--npz_dir', type=str, default='/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test', help='Directory containing .npz files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for plots')
    parser.add_argument('--title', type=str, default='ASD Cohorts', help='Plot title')
    
    args = parser.parse_args()
    
    print_section_header("BRAIN AGE PREDICTION - ASD COHORTS")
    
    print_info(f"NPZ files directory: {args.npz_dir}")
    print_info(f"Output directory: {args.output_dir}")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output path
    output_path = output_dir / "asd_cohorts_combined_scatter"
    
    print_step(1, "CREATING ASD COHORTS PLOT", "1x2 layout: ABIDE ASD, Stanford ASD")
    
    # Create the plot
    plot_combined_asd_cohorts(args.npz_dir, str(output_path), args.title)
    
    print()
    print_completion("ASD Cohorts Brain Age Plot", [f"{output_path}.png"])


if __name__ == "__main__":
    main()