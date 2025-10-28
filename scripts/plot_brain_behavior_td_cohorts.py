#!/usr/bin/env python3
"""
Create combined brain-behavior plots for TD cohorts.

Generates 3-panel subplot figures:
- One figure for Hyperactivity/Impulsivity measures
- One figure for Inattention measures

Each panel shows predicted vs actual behavioral scores for one cohort
(NKI-RS TD, ADHD200 TD, CMI-HBN TD).
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
import numpy as np
from pathlib import Path
import sys
import os

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
sys.path.insert(0, str(Path(__file__).parent))

from logging_utils import (print_section_header, print_step, print_success, 
                           print_warning, print_error, print_info)
from plot_styles import setup_arial_font, DPI, FIGURE_FACECOLOR

# Setup font
setup_arial_font()

# ============================================================================
# PRE-CONFIGURED PATHS
# ============================================================================
BASE_DIR = Path("/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior")

# Input directories for each cohort
NKI_DIR = BASE_DIR / "nki_rs_td"
ADHD200_DIR = BASE_DIR / "adhd200_td"
CMIHBN_DIR = BASE_DIR / "cmihbn_td"

# Output directory
OUTPUT_DIR = BASE_DIR / "combined_plots"

# Behavioral measure mappings
# Map cohort-specific column names to common names
HYPERACTIVITY_MEASURES = {
    'nki': ['B_TOTAL_HYPERACTIVITYRESTLESSNESS', 'B_Total_HyperactivityRestlessness'],  # Try multiple variants
    'adhd200': ['Hyper/Impulsive', 'HyperImpulsive'],
    'cmihbn': ['C3SR_HY_T', 'C3SR,C3SR_HY_T']
}

INATTENTION_MEASURES = {
    'nki': ['A_TOTAL_INATTENTIONMEMORY_PROBLEMS', 'A_Total_InattentionMemory_Problems'],
    'adhd200': ['Inattentive'],
    'cmihbn': ['C3SR_IN_T', 'C3SR,C3SR_IN_T']
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_scatter_plot(cohort_dir, measure_names):
    """Find scatter plot file for a behavioral measure."""
    cohort_dir = Path(cohort_dir)
    
    # Try each possible measure name
    for measure in measure_names:
        # Create safe filename (same logic as in the analysis scripts)
        safe_name = measure.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').replace(',', '')
        pattern = f"scatter_{safe_name}.png"
        
        # Look for the file
        matches = list(cohort_dir.glob(pattern))
        if matches:
            return matches[0]
    
    # If not found, try a more flexible search (any scatter plot with similar keywords)
    # For hyperactivity: look for files with "hyper" or "impuls"
    # For inattention: look for files with "inatt"
    all_scatters = list(cohort_dir.glob("scatter_*.png"))
    for scatter_file in all_scatters:
        filename_lower = scatter_file.stem.lower()
        for measure in measure_names:
            measure_lower = measure.lower()
            # Check if key parts of the measure name are in the filename
            if 'hyper' in measure_lower or 'impuls' in measure_lower:
                if 'hyper' in filename_lower or 'impuls' in filename_lower:
                    return scatter_file
            elif 'inatt' in measure_lower:
                if 'inatt' in filename_lower:
                    return scatter_file
    
    return None


def load_results(cohort_dir, measure_names):
    """Load regression results for a behavioral measure."""
    cohort_dir = Path(cohort_dir)
    results_file = cohort_dir / "linear_regression_results.csv"
    
    if not results_file.exists():
        return None
    
    df = pd.read_csv(results_file)
    
    # Try to find matching measure
    for measure in measure_names:
        matching = df[df['Behavioral_Measure'].str.contains(measure, case=False, na=False)]
        if not matching.empty:
            return matching.iloc[0].to_dict()
    
    return None


def create_combined_plot(measure_type, cohort_data, output_path):
    """
    Create 3-panel subplot for a behavioral measure across cohorts.
    
    Args:
        measure_type: 'Hyperactivity' or 'Inattention'
        cohort_data: dict with keys 'nki', 'adhd200', 'cmihbn', each containing
                     {'plot_path': path, 'results': dict}
        output_path: Path to save combined plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    cohort_names = ['NKI-RS TD', 'ADHD200 TD', 'CMI-HBN TD']
    cohort_keys = ['nki', 'adhd200', 'cmihbn']
    
    for idx, (ax, cohort_name, cohort_key) in enumerate(zip(axes, cohort_names, cohort_keys)):
        data = cohort_data.get(cohort_key)
        
        if data is None or data['plot_path'] is None:
            # No data available
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', fontsize=12, color='gray')
            ax.set_title(cohort_name, fontsize=14, fontweight='bold')
            ax.axis('off')
            continue
        
        # Load the scatter plot image
        img = plt.imread(data['plot_path'])
        ax.imshow(img)
        ax.axis('off')
        
        # Add cohort name as title
        ax.set_title(cohort_name, fontsize=14, fontweight='bold', pad=10)
        
        # Add results text below
        if data['results'] is not None:
            results = data['results']
            n = int(results['N_Subjects'])
            rho = results['Spearman_Rho']
            p = results['P_Value']
            r2 = results['R2']
            
            # Format p-value
            if p < 0.001:
                p_str = "< 0.001"
            else:
                p_str = f"= {p:.3f}"
            
            # Add text annotation
            text = f"N = {n}\nρ = {rho:.3f}, p {p_str}\nR² = {r2:.3f}"
            ax.text(0.5, -0.05, text, transform=ax.transAxes,
                   ha='center', va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Overall title
    fig.suptitle(f'{measure_type} - TD Cohorts', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Save PNG + TIFF + AI
    png_path = Path(output_path)
    tiff_path = png_path.with_suffix('.tiff')
    ai_path = png_path.with_suffix('.ai')
    
    plt.savefig(png_path, dpi=DPI, bbox_inches='tight', facecolor=FIGURE_FACECOLOR, edgecolor='none')
    plt.savefig(tiff_path, dpi=DPI, bbox_inches='tight', facecolor=FIGURE_FACECOLOR, edgecolor='none', format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
    pdf_backend.FigureCanvas(fig).print_pdf(str(ai_path))
    
    plt.close()
    
    print_success(f"Saved: {output_path.name} and {Path(ai_path).name}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print_section_header("COMBINED BRAIN-BEHAVIOR PLOTS - TD COHORTS")
    
    print_info(f"NKI-RS TD:   {NKI_DIR}")
    print_info(f"ADHD200 TD:  {ADHD200_DIR}")
    print_info(f"CMI-HBN TD:  {CMIHBN_DIR}")
    print_info(f"Output:      {OUTPUT_DIR}")
    print()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # HYPERACTIVITY/IMPULSIVITY
    # ========================================================================
    print_step("Processing Hyperactivity/Impulsivity", "Loading data from all cohorts")
    
    hyperactivity_data = {}
    
    for cohort_key, cohort_dir in [('nki', NKI_DIR), ('adhd200', ADHD200_DIR), ('cmihbn', CMIHBN_DIR)]:
        measure_names = HYPERACTIVITY_MEASURES[cohort_key]
        plot_path = find_scatter_plot(cohort_dir, measure_names)
        results = load_results(cohort_dir, measure_names)
        
        hyperactivity_data[cohort_key] = {
            'plot_path': plot_path,
            'results': results
        }
        
        if plot_path:
            print_info(f"  {cohort_key.upper()}: Found plot")
        else:
            print_warning(f"  {cohort_key.upper()}: No plot found")
    
    # Create combined plot
    hyper_output = OUTPUT_DIR / "hyperactivity_td_cohorts.png"
    create_combined_plot("Hyperactivity/Impulsivity", hyperactivity_data, hyper_output)
    print()
    
    # ========================================================================
    # INATTENTION
    # ========================================================================
    print_step("Processing Inattention", "Loading data from all cohorts")
    
    inattention_data = {}
    
    for cohort_key, cohort_dir in [('nki', NKI_DIR), ('adhd200', ADHD200_DIR), ('cmihbn', CMIHBN_DIR)]:
        measure_names = INATTENTION_MEASURES[cohort_key]
        plot_path = find_scatter_plot(cohort_dir, measure_names)
        results = load_results(cohort_dir, measure_names)
        
        inattention_data[cohort_key] = {
            'plot_path': plot_path,
            'results': results
        }
        
        if plot_path:
            print_info(f"  {cohort_key.upper()}: Found plot")
        else:
            print_warning(f"  {cohort_key.upper()}: No plot found")
    
    # Create combined plot
    inatt_output = OUTPUT_DIR / "inattention_td_cohorts.png"
    create_combined_plot("Inattention", inattention_data, inatt_output)
    print()
    
    print_section_header("COMPLETE")
    print_success(f"Combined plots saved to: {OUTPUT_DIR}")
    print_info(f"  • hyperactivity_td_cohorts.png")
    print_info(f"  • inattention_td_cohorts.png")


if __name__ == "__main__":
    main()

