#!/usr/bin/env python3
"""
Create publication-quality multi-panel figure from optimization results.

This script:
1. Reads optimization_summary.csv
2. Identifies measures with significant correlations
3. Creates a multi-panel figure with the best results
4. Exports in publication-ready formats

Usage:
    python plot_best_optimization_results.py --cohort stanford_asd
    python plot_best_optimization_results.py --cohort abide_asd --min-rho 0.25
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
from scipy.stats import spearmanr
import argparse

# Add to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from logging_utils import print_section_header, print_step, print_success, print_warning, print_info
from plot_styles import create_standardized_scatter, get_dataset_title, setup_arial_font, DPI, FIGURE_FACECOLOR

# Setup Arial font
setup_arial_font()

# Base directories
BASE_DIR = Path("/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior")

COHORTS = {
    'stanford_asd': {
        'name': 'Stanford ASD',
        'dataset': 'stanford_asd',
        'dir': BASE_DIR / 'stanford_asd_optimized'
    },
    'abide_asd': {
        'name': 'ABIDE ASD',
        'dataset': 'abide_asd',
        'dir': BASE_DIR / 'abide_asd_optimized'
    },
    'adhd200_td': {
        'name': 'ADHD200 TD',
        'dataset': 'adhd200_td',
        'dir': BASE_DIR / 'adhd200_td_optimized'
    },
    'adhd200_adhd': {
        'name': 'ADHD200 ADHD',
        'dataset': 'adhd200_adhd',
        'dir': BASE_DIR / 'adhd200_adhd_optimized'
    },
    'cmihbn_td': {
        'name': 'CMI-HBN TD',
        'dataset': 'cmihbn_td',
        'dir': BASE_DIR / 'cmihbn_td_optimized'
    },
    'cmihbn_adhd': {
        'name': 'CMI-HBN ADHD',
        'dataset': 'cmihbn_adhd',
        'dir': BASE_DIR / 'cmihbn_adhd_optimized'
    },
    'nki': {
        'name': 'NKI-RS TD',
        'dataset': 'nki_rs_td',
        'dir': BASE_DIR / 'nki_rs_td_optimized'
    }
}


def load_optimization_results(cohort_dir):
    """Load optimization summary and individual result files."""
    print_step("Loading optimization results", str(cohort_dir))
    
    summary_file = cohort_dir / "optimization_summary.csv"
    
    if not summary_file.exists():
        raise FileNotFoundError(f"No optimization_summary.csv found in {cohort_dir}")
    
    summary_df = pd.read_csv(summary_file)
    
    print_info(f"Found {len(summary_df)} behavioral measures", 0)
    
    return summary_df


def filter_significant_results(summary_df, min_rho=0.2, max_pvalue=0.05):
    """Filter for significant results."""
    print_step("Filtering for significant results", f"min_rho={min_rho}, max_p={max_pvalue}")
    
    significant = summary_df[
        (summary_df['Final_Spearman'].abs() >= min_rho) & 
        (summary_df['Final_P_Value'] <= max_pvalue)
    ].copy()
    
    # Sort by absolute Spearman correlation
    significant = significant.sort_values('Final_Spearman', key=abs, ascending=False)
    
    print_info(f"Significant results: {len(significant)}/{len(summary_df)}", 0)
    
    if len(significant) == 0:
        print_warning("No significant results found!")
        return None
    
    # Print summary
    print("\n  Significant Measures:")
    for idx, row in significant.iterrows():
        print(f"    - {row['Measure']:.<40} ρ={row['Final_Spearman']:>6.3f}, p={row['Final_P_Value']:.4f}")
    
    return significant


def load_measure_data(cohort_dir, measure_name):
    """Load the actual data for a specific measure from optimization results."""
    # Try to load from the optimization results CSV
    safe_name = measure_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
    opt_file = cohort_dir / f"optimization_results_{safe_name}.csv"
    
    if not opt_file.exists():
        print_warning(f"Optimization results file not found: {opt_file.name}")
        return None
    
    # We need the actual predictions - check if we saved them
    # For now, we'll use the saved plots
    return None


def create_multi_panel_figure(summary_df, significant_df, cohort_config, output_dir):
    """Create multi-panel figure with best results."""
    print_step("Creating multi-panel figure", f"{len(significant_df)} panels")
    
    n_panels = len(significant_df)
    
    if n_panels == 0:
        print_warning("No panels to plot")
        return
    
    # Determine layout
    if n_panels == 1:
        nrows, ncols = 1, 1
        figsize = (6, 6)
    elif n_panels == 2:
        nrows, ncols = 1, 2
        figsize = (12, 5)
    elif n_panels <= 4:
        nrows, ncols = 2, 2
        figsize = (12, 10)
    elif n_panels <= 6:
        nrows, ncols = 2, 3
        figsize = (15, 10)
    else:
        nrows = (n_panels + 2) // 3
        ncols = 3
        figsize = (15, 5 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_panels == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    dataset_title = get_dataset_title(cohort_config['dataset'])
    
    for idx, (_, row) in enumerate(significant_df.iterrows()):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        
        # Create text summary for this measure (we don't have actual data to plot)
        measure = row['Measure']
        rho = row['Final_Spearman']
        p_val = row['Final_P_Value']
        strategy = row['Best_Strategy']
        model = row['Best_Model']
        n_subj = row['N_Subjects']
        
        # Format p-value
        p_str = f"< 0.001" if p_val < 0.001 else f"= {p_val:.3f}"
        
        # Create info text
        info_text = f"{measure}\n\n"
        info_text += f"r = {rho:.3f}\n"
        info_text += f"p {p_str}\n"
        info_text += f"N = {n_subj}\n\n"
        info_text += f"Strategy: {strategy}\n"
        info_text += f"Model: {model}"
        
        if row['Best_N_Components'] is not None and not pd.isna(row['Best_N_Components']):
            info_text += f"\nComponents: {int(row['Best_N_Components'])}"
        if row['Best_Alpha'] is not None and not pd.isna(row['Best_Alpha']):
            info_text += f"\nα: {row['Best_Alpha']}"
        
        # Display as text panel
        ax.text(0.5, 0.5, info_text, ha='center', va='center', 
                fontsize=10, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    # Hide unused axes
    for idx in range(n_panels, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f"{dataset_title} - Optimization Results (Significant Only)", 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / f"{cohort_config['dataset']}_optimization_summary_figure"
    
    png_path = output_path.with_suffix('.png')
    tiff_path = output_path.with_suffix('.tiff')
    ai_path = output_path.with_suffix('.ai')
    
    plt.savefig(png_path, dpi=DPI, bbox_inches='tight', facecolor=FIGURE_FACECOLOR)
    plt.savefig(tiff_path, dpi=DPI, bbox_inches='tight', facecolor=FIGURE_FACECOLOR,
               format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
    pdf_backend.FigureCanvas(fig).print_pdf(str(ai_path))
    
    plt.close()
    
    print_success(f"Saved: {png_path.name}")
    print_success(f"Saved: {tiff_path.name}")
    print_success(f"Saved: {ai_path.name}")


def create_summary_table(summary_df, significant_df, output_dir):
    """Create a formatted summary table."""
    print_step("Creating summary table", "LaTeX and CSV formats")
    
    # Select columns for table
    table_df = significant_df[[
        'Measure', 'N_Subjects', 'Best_Strategy', 'Best_Model',
        'CV_Spearman', 'Final_Spearman', 'Final_P_Value', 'Final_R2'
    ]].copy()
    
    # Rename columns for publication
    table_df.columns = [
        'Behavioral Measure', 'N', 'Strategy', 'Model',
        'CV ρ', 'Final ρ', 'p-value', 'R²'
    ]
    
    # Format numerical columns
    table_df['CV ρ'] = table_df['CV ρ'].apply(lambda x: f"{x:.3f}")
    table_df['Final ρ'] = table_df['Final ρ'].apply(lambda x: f"{x:.3f}")
    table_df['p-value'] = table_df['p-value'].apply(lambda x: f"{x:.4f}" if x >= 0.001 else "< 0.001")
    table_df['R²'] = table_df['R²'].apply(lambda x: f"{x:.3f}")
    
    # Save CSV
    csv_path = output_dir / "optimization_summary_significant.csv"
    table_df.to_csv(csv_path, index=False)
    print_success(f"Saved CSV: {csv_path.name}")
    
    # Save LaTeX
    latex_path = output_dir / "optimization_summary_significant.tex"
    with open(latex_path, 'w') as f:
        f.write(table_df.to_latex(index=False, escape=False))
    print_success(f"Saved LaTeX: {latex_path.name}")
    
    return table_df


def main():
    parser = argparse.ArgumentParser(
        description="Create publication figures from optimization results"
    )
    parser.add_argument(
        '--cohort', '-c',
        choices=list(COHORTS.keys()),
        required=True,
        help="Cohort to analyze"
    )
    parser.add_argument(
        '--min-rho',
        type=float,
        default=0.2,
        help="Minimum absolute Spearman correlation (default: 0.2)"
    )
    parser.add_argument(
        '--max-pvalue',
        type=float,
        default=0.05,
        help="Maximum p-value (default: 0.05)"
    )
    
    args = parser.parse_args()
    
    cohort_config = COHORTS[args.cohort]
    cohort_dir = cohort_config['dir']
    
    print_section_header(f"OPTIMIZATION RESULTS VISUALIZATION - {cohort_config['name'].upper()}")
    print()
    
    if not cohort_dir.exists():
        print_warning(f"Directory not found: {cohort_dir}")
        print_info("Have you run the optimization yet?", 0)
        return
    
    try:
        # Load results
        summary_df = load_optimization_results(cohort_dir)
        print()
        
        # Filter significant
        significant_df = filter_significant_results(summary_df, args.min_rho, args.max_pvalue)
        
        if significant_df is None or len(significant_df) == 0:
            print_warning("No significant results to plot!")
            print_info(f"Try lowering --min-rho (currently {args.min_rho})", 0)
            return
        
        print()
        
        # Create multi-panel figure
        create_multi_panel_figure(summary_df, significant_df, cohort_config, cohort_dir)
        print()
        
        # Create summary table
        create_summary_table(summary_df, significant_df, cohort_dir)
        print()
        
        print_success("✓ All figures and tables created!")
        print_info(f"Location: {cohort_dir}", 0)
        
    except Exception as e:
        print_warning(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

