#!/usr/bin/env python3
"""
Create combined brain age prediction scatter plots for ADHD cohorts.

This script creates a single scatter plot panel combining all ADHD cohorts
(CMI-HBN ADHD, ADHD200 ADHD) with different colors for each dataset.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / 'utils'))

from plotting_utils import setup_fonts, save_figure

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_age_data_from_npz(predicted_file: str, actual_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load predicted and actual ages from .npz files.
    
    Args:
        predicted_file (str): Path to .npz file with predicted ages
        actual_file (str): Path to .npz file with actual ages
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (predicted_ages, actual_ages)
    """
    # Load predicted ages
    if not os.path.exists(predicted_file):
        raise FileNotFoundError(f"Predicted ages file not found: {predicted_file}")
    
    predicted_data = np.load(predicted_file)
    if 'predicted' in predicted_data:
        predicted_ages = predicted_data['predicted']
    else:
        keys = list(predicted_data.keys())
        if keys:
            predicted_ages = predicted_data[keys[0]]
            logging.warning(f"Using key '{keys[0]}' for predicted ages from {predicted_file}")
        else:
            raise ValueError(f"No data found in predicted file: {predicted_file}")
    
    # Load actual ages
    if not os.path.exists(actual_file):
        raise FileNotFoundError(f"Actual ages file not found: {actual_file}")
    
    actual_data = np.load(actual_file)
    if 'actual' in actual_data:
        actual_ages = actual_data['actual']
    else:
        keys = list(actual_data.keys())
        if keys:
            actual_ages = actual_data[keys[0]]
            logging.warning(f"Using key '{keys[0]}' for actual ages from {actual_file}")
        else:
            raise ValueError(f"No data found in actual file: {actual_file}")
    
    # Ensure same length
    min_length = min(len(predicted_ages), len(actual_ages))
    if len(predicted_ages) != len(actual_ages):
        logging.warning(f"Length mismatch: predicted={len(predicted_ages)}, actual={len(actual_ages)}. Using first {min_length} samples.")
        predicted_ages = predicted_ages[:min_length]
        actual_ages = actual_ages[:min_length]
    
    return predicted_ages, actual_ages


def plot_combined_adhd_cohorts(npz_files_dir: str, output_path: str, 
                             title: str = "Brain Age Prediction: ADHD Cohorts") -> None:
    """
    Create combined scatter plot for all ADHD cohorts.
    
    Args:
        npz_files_dir (str): Directory containing .npz files
        output_path (str): Output path for the plot (without extension)
        title (str): Plot title
    """
    setup_fonts()
    
    # Define ADHD cohort datasets and their colors
    adhd_datasets = {
        'CMI-HBN ADHD': {
            'predicted': 'predicted_cmihbn_adhd_ages_most_updated.npz',
            'actual': 'actual_cmihbn_adhd_ages_most_updated.npz',
            'color': '#1f77b4',  # Blue
            'marker': 'o'
        },
        'ADHD200 ADHD': {
            'predicted': 'predicted_adhd200_ages_most_updated.npz',
            'actual': 'actual_adhd200_ages_most_updated.npz',
            'color': '#ff7f0e',  # Orange
            'marker': 's'
        }
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    # Collect all data for overall statistics
    all_actual = []
    all_predicted = []
    dataset_stats = {}
    
    # Plot each dataset
    for dataset_name, dataset_info in adhd_datasets.items():
        predicted_file = os.path.join(npz_files_dir, dataset_info['predicted'])
        actual_file = os.path.join(npz_files_dir, dataset_info['actual'])
        
        try:
            predicted_ages, actual_ages = load_age_data_from_npz(predicted_file, actual_file)
            
            # Calculate metrics for this dataset
            r, p = pearsonr(actual_ages, predicted_ages)
            r_squared = r ** 2
            mae = mean_absolute_error(actual_ages, predicted_ages)
            
            # Store statistics
            dataset_stats[dataset_name] = {
                'r': r, 'r_squared': r_squared, 'mae': mae, 'p': p, 'n': len(actual_ages)
            }
            
            # Plot scatter points
            ax.scatter(actual_ages, predicted_ages, 
                      color=dataset_info['color'], 
                      marker=dataset_info['marker'],
                      alpha=0.6, s=50, 
                      edgecolor='white', linewidth=0.5,
                      label=f'{dataset_name} (n={len(actual_ages)}, R²={r_squared:.3f}, MAE={mae:.2f})')
            
            # Collect for overall statistics
            all_actual.extend(actual_ages)
            all_predicted.extend(predicted_ages)
            
            logging.info(f"{dataset_name}: R²={r_squared:.3f}, MAE={mae:.2f}, r={r:.3f}, p={p:.3f}, n={len(actual_ages)}")
            
        except Exception as e:
            logging.warning(f"Could not load data for {dataset_name}: {e}")
            continue
    
    if not all_actual:
        logging.error("No ADHD cohort data found!")
        return
    
    # Calculate overall statistics
    all_actual = np.array(all_actual)
    all_predicted = np.array(all_predicted)
    overall_r, overall_p = pearsonr(all_actual, all_predicted)
    overall_r_squared = overall_r ** 2
    overall_mae = mean_absolute_error(all_actual, all_predicted)
    
    # Set axis limits
    age_min = min(all_actual.min(), all_predicted.min())
    age_max = max(all_actual.max(), all_predicted.max())
    lims = [age_min - 1, age_max + 2]
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    # Add identity line (perfect prediction)
    ax.plot(lims, lims, 'k--', alpha=0.8, linewidth=2, label='Perfect Prediction')
    
    # Add overall regression line
    z = np.polyfit(all_actual, all_predicted, 1)
    p = np.poly1d(z)
    ax.plot(lims, p(lims), 'k-', alpha=0.8, linewidth=2, label=f'Overall Fit (R²={overall_r_squared:.3f})')
    
    # Format p-value
    if overall_p < 0.001:
        p_text = "P < 0.001"
    else:
        p_text = f"P = {overall_p:.3f}"
    
    # Add overall statistics text
    ax.text(0.05, 0.95,
            f"Overall Statistics:\n"
            f"$\mathit{{r}} = {overall_r:.3f}$\n"
            f"$\mathit{{R}}^2 = {overall_r_squared:.3f}$\n"
            f"$\mathit{{MAE}} = {overall_mae:.2f}$ years\n"
            f"{p_text}\n"
            f"Total N = {len(all_actual)}",
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Customize plot
    ax.set_xlabel('Chronological Age (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Brain Age (years)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    
    # Customize ticks and grid
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    save_figure(fig, output_path)
    logging.info(f"Combined ADHD cohorts plot saved to: {output_path}")
    logging.info(f"Overall ADHD cohorts - R²: {overall_r_squared:.3f}, MAE: {overall_mae:.2f} years, r: {overall_r:.3f}, p: {overall_p:.3f}, N: {len(all_actual)}")


def main():
    """Main function for creating combined ADHD cohorts brain age prediction plot."""
    parser = argparse.ArgumentParser(
        description="Create combined brain age prediction scatter plot for ADHD cohorts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create combined ADHD cohorts plot
  python plot_brain_age_adhd_cohorts.py --npz_dir . --output_dir results/brain_age_plots
  
  # With custom title
  python plot_brain_age_adhd_cohorts.py --npz_dir . --output_dir results/brain_age_plots --title "ADHD Cohorts Brain Age Prediction"
        """
    )
    
    parser.add_argument('--npz_dir', type=str, default='.', help='Directory containing .npz files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for plots')
    parser.add_argument('--title', type=str, default='Brain Age Prediction: ADHD Cohorts', help='Plot title')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create combined ADHD cohorts plot
    output_path = output_dir / "adhd_cohorts_combined_scatter"
    plot_combined_adhd_cohorts(args.npz_dir, str(output_path), args.title)


if __name__ == "__main__":
    main()
