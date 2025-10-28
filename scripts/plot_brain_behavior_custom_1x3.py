#!/usr/bin/env python3
"""
Create custom 1x3 subplot: NKI HY, NKI IN, ADHD200 TD HY

This script creates a combined figure with three subplots showing:
1. NKI-RS TD Hyperactivity
2. NKI-RS TD Inattention
3. ADHD200 TD Hyperactivity

Consistent styling: Arial font, #5A6FA8 dots, #D32F2F line, no top/right spines,
no bounding box around stats, major ticks only.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from plot_styles import setup_arial_font, create_standardized_scatter, DPI, FIGURE_FACECOLOR

# Setup font
setup_arial_font()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Base directories
BASE_DIR = Path("/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior")

# Datasets and their behavioral measures
PLOTS_CONFIG = [
    {
        'dataset': 'nki_rs_td',
        'measure': 'CAARS_CAARS_ADHD_Hyperactive_Impulsive_T',
        'title': 'NKI-RS TD\nHyperactivity'
    },
    {
        'dataset': 'nki_rs_td',
        'measure': 'CAARS_CAARS_ADHD_Inattentive_Inattention_T',
        'title': 'NKI-RS TD\nInattention'
    },
    {
        'dataset': 'adhd200_td',
        'measure': 'Hyper_Impulsive',
        'title': 'ADHD200 TD\nHyperactivity'
    }
]

OUTPUT_DIR = BASE_DIR / "combined_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_results_csv(dataset_dir: Path, measure: str) -> Path:
    """Find the results CSV file for a given measure."""
    csv_files = list(dataset_dir.glob("results_*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No results CSV found in {dataset_dir}")
    
    # Use the first results file (should only be one)
    return csv_files[0]


def load_behavioral_data(dataset_dir: Path, measure: str):
    """Load behavioral data from results CSV."""
    csv_path = find_results_csv(dataset_dir, measure)
    
    df = pd.read_csv(csv_path)
    
    # Find the row for this measure
    measure_row = df[df['Behavioral_Measure'] == measure]
    
    if measure_row.empty:
        raise ValueError(f"Measure '{measure}' not found in {csv_path}")
    
    # Extract statistics
    rho = measure_row['Spearman_Rho'].values[0]
    p_value = measure_row['P_Value'].values[0]
    
    return rho, p_value


def load_predictions(dataset_dir: Path, measure: str):
    """Load predicted and actual values from saved predictions."""
    # Look for prediction files
    pred_files = list(dataset_dir.glob(f"predictions_{measure}*.csv"))
    
    if not pred_files:
        # Try with sanitized measure name
        safe_measure = measure.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
        pred_files = list(dataset_dir.glob(f"predictions_{safe_measure}*.csv"))
    
    if not pred_files:
        raise FileNotFoundError(f"No predictions file found for {measure} in {dataset_dir}")
    
    pred_df = pd.read_csv(pred_files[0])
    
    actual = pred_df['Actual'].values
    predicted = pred_df['Predicted'].values
    
    return actual, predicted


def create_subplot(ax, actual, predicted, rho, p_value, title):
    """Create subplot using centralized styling."""
    mae = mean_absolute_error(actual, predicted)
    p_str = "< 0.001" if p_value < 0.001 else f"= {p_value:.4f}"
    stats_text = f'ρ = {rho:.3f}\np {p_str}\nMAE = {mae:.2f}'
    
    create_standardized_scatter(ax, actual, predicted, title=title,
                               xlabel='Observed Behavioral Score',
                               ylabel='Predicted Behavioral Score',
                               stats_text=stats_text, is_subplot=True)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Create 1x3 subplot figure."""
    
    print("\n" + "="*100)
    print("CUSTOM 1x3 BRAIN-BEHAVIOR PLOT")
    print("="*100)
    print("\nConfiguration:")
    print("  • NKI-RS TD Hyperactivity")
    print("  • NKI-RS TD Inattention")
    print("  • ADHD200 TD Hyperactivity")
    print()
    
    # Create figure with 1 row, 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Process each subplot
    for idx, config in enumerate(PLOTS_CONFIG):
        dataset = config['dataset']
        measure = config['measure']
        title = config['title']
        
        print(f"[{idx+1}/3] Loading {dataset} - {measure}")
        
        dataset_dir = BASE_DIR / dataset
        
        try:
            # Load data
            rho, p_value = load_behavioral_data(dataset_dir, measure)
            actual, predicted = load_predictions(dataset_dir, measure)
            
            print(f"  ✓ Loaded: n={len(actual)}, ρ={rho:.3f}, p={p_value:.4f}")
            
            # Create subplot
            create_subplot(axes[idx], actual, predicted, rho, p_value, title)
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            axes[idx].text(0.5, 0.5, f"Error loading\n{dataset}\n{measure}", 
                          ha='center', va='center', transform=axes[idx].transAxes)
    
    # Adjust spacing between subplots - minimal whitespace
    plt.tight_layout(pad=2.0, w_pad=2.5, h_pad=1.0)
    
    # Save PNG + TIFF + AI
    base_name = "custom_1x3_nki_adhd200_hyperactivity_inattention"
    png_path = OUTPUT_DIR / f"{base_name}.png"
    tiff_path = OUTPUT_DIR / f"{base_name}.tiff"
    ai_path = OUTPUT_DIR / f"{base_name}.ai"
    
    plt.savefig(png_path, dpi=DPI, bbox_inches='tight', facecolor=FIGURE_FACECOLOR, edgecolor='none')
    plt.savefig(tiff_path, dpi=DPI, bbox_inches='tight', facecolor=FIGURE_FACECOLOR, edgecolor='none', format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
    pdf_backend.FigureCanvas(fig).print_pdf(str(ai_path))
    
    print(f"\n✓ Saved: {png_path.name} + {tiff_path.name} + {ai_path.name}")
    
    plt.close()
    
    print("\n" + "="*100)
    print("COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()

