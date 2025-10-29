#!/usr/bin/env python3
"""
Create comprehensive summary figure from optimization results.

Shows all behavioral measures with their optimization performance,
highlighting best strategies and significant correlations.

Usage:
    python create_optimization_summary_figure.py --cohort stanford_asd
    python create_optimization_summary_figure.py --cohort abide_asd --min-rho 0.25
    python create_optimization_summary_figure.py --all  # All cohorts

Author: Brain-Behavior Optimization Team
Date: 2024
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
import warnings
warnings.filterwarnings('ignore')

# Add to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from logging_utils import (print_section_header, print_step, print_success, 
                           print_warning, print_error, print_info, print_completion)
from plot_styles import setup_arial_font, DPI, FIGURE_FACECOLOR

# Setup Arial font globally
setup_arial_font()

# ============================================================================
# COHORT CONFIGURATIONS
# ============================================================================

COHORTS = {
    'abide_asd': {
        'name': 'ABIDE ASD',
        'results_dir': '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/abide_asd_optimized'
    },
    'stanford_asd': {
        'name': 'Stanford ASD',
        'results_dir': '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/stanford_asd_optimized'
    },
    'adhd200_td': {
        'name': 'ADHD200 TD',
        'results_dir': '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/adhd200_td_optimized'
    },
    'adhd200_adhd': {
        'name': 'ADHD200 ADHD',
        'results_dir': '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/adhd200_adhd_optimized'
    },
    'cmihbn_td': {
        'name': 'CMI-HBN TD',
        'results_dir': '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/cmihbn_td_optimized'
    },
    'cmihbn_adhd': {
        'name': 'CMI-HBN ADHD',
        'results_dir': '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/cmihbn_adhd_optimized'
    },
    'nki_rs_td': {
        'name': 'NKI-RS TD',
        'results_dir': '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/nki_rs_td_optimized'
    }
}

# Strategy color mapping
STRATEGY_COLORS = {
    'PCA': '#1f77b4',  # Blue
    'PCA+Selection': '#ff7f0e',  # Orange
    'Selection': '#2ca02c',  # Green
    'Raw': '#d62728',  # Red
    'Unknown': '#7f7f7f'  # Gray
}

# ============================================================================
# LOADING FUNCTIONS
# ============================================================================

def load_optimization_summary(results_dir):
    """Load optimization summary CSV."""
    summary_path = Path(results_dir) / 'optimization_summary.csv'
    
    if not summary_path.exists():
        raise FileNotFoundError(f"Optimization summary not found: {summary_path}")
    
    df = pd.read_csv(summary_path)
    return df


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def create_summary_bar_plot(df, cohort_name, output_dir, min_rho=None):
    """
    Create bar plot showing Spearman correlations for all measures.
    Color-coded by optimization strategy.
    """
    print_step(f"Creating summary figure for {cohort_name}", "")
    
    # Filter by minimum rho if specified
    if min_rho is not None:
        df = df[df['Final_Spearman'].abs() >= min_rho]
        print_info(f"Filtered to {len(df)} measures with |ρ| >= {min_rho}", 0)
    
    if len(df) == 0:
        print_warning("No measures to plot after filtering")
        return None
    
    # Sort by absolute Spearman correlation
    df = df.sort_values('Final_Spearman', key=abs, ascending=True)
    
    # Determine figure size based on number of measures
    n_measures = len(df)
    fig_height = max(6, n_measures * 0.4)
    
    fig, ax = plt.subplots(figsize=(10, fig_height))
    
    # Get colors based on strategy
    colors = [STRATEGY_COLORS.get(strategy, STRATEGY_COLORS['Unknown']) 
              for strategy in df['Best_Strategy']]
    
    # Create horizontal bar plot
    y_pos = np.arange(len(df))
    bars = ax.barh(y_pos, df['Final_Spearman'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add vertical line at x=0
    ax.axvline(x=0, color='black', linewidth=1, linestyle='-', alpha=0.3)
    
    # Add significance markers (p < 0.05, p < 0.01, p < 0.001)
    for i, (idx, row) in enumerate(df.iterrows()):
        x_pos = row['Final_Spearman']
        if row['Final_P_Value'] < 0.001:
            marker = '***'
        elif row['Final_P_Value'] < 0.01:
            marker = '**'
        elif row['Final_P_Value'] < 0.05:
            marker = '*'
        else:
            marker = ''
        
        if marker:
            # Place marker at end of bar
            offset = 0.01 if x_pos > 0 else -0.01
            ax.text(x_pos + offset, i, marker, 
                   ha='left' if x_pos > 0 else 'right', 
                   va='center', fontsize=10, fontweight='bold')
    
    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['Measure'], fontsize=9)
    ax.set_xlabel('Spearman ρ', fontsize=12, fontweight='bold')
    ax.set_title(f'{cohort_name} - Brain-Behavior Optimization Results', 
                fontsize=14, fontweight='bold', pad=15)
    
    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Create legend for strategies
    unique_strategies = df['Best_Strategy'].unique()
    legend_patches = [mpatches.Patch(color=STRATEGY_COLORS.get(s, STRATEGY_COLORS['Unknown']), 
                                     label=s, alpha=0.8) 
                     for s in unique_strategies]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=9, 
             title='Optimization Strategy', framealpha=0.9)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir)
    if min_rho is not None:
        filename = f'optimization_summary_minrho{min_rho:.2f}'
    else:
        filename = 'optimization_summary'
    
    png_path = output_path / f'{filename}.png'
    pdf_path = output_path / f'{filename}.pdf'
    
    plt.savefig(png_path, dpi=DPI, bbox_inches='tight', facecolor=FIGURE_FACECOLOR)
    plt.savefig(pdf_path, bbox_inches='tight', facecolor=FIGURE_FACECOLOR)
    plt.close()
    
    print_success(f"Saved summary figure: {png_path.name}")
    print_success(f"Saved summary figure: {pdf_path.name}")
    
    return png_path


def create_detailed_table(df, cohort_name, output_dir, min_rho=None):
    """Create detailed table with key metrics."""
    print_step("Creating detailed metrics table", "")
    
    # Filter by minimum rho if specified
    if min_rho is not None:
        df = df[df['Final_Spearman'].abs() >= min_rho]
    
    if len(df) == 0:
        print_warning("No measures to include in table")
        return None
    
    # Sort by absolute Spearman correlation (descending)
    df_sorted = df.sort_values('Final_Spearman', key=abs, ascending=False)
    
    # Select key columns for table
    table_df = df_sorted[[
        'Measure', 'N_Subjects', 'Best_Strategy', 'Best_Model',
        'Final_Spearman', 'Final_P_Value', 'Final_R2'
    ]].copy()
    
    # Format columns
    table_df['Final_Spearman'] = table_df['Final_Spearman'].apply(lambda x: f"{x:.3f}")
    table_df['Final_P_Value'] = table_df['Final_P_Value'].apply(
        lambda x: "< 0.001" if x < 0.001 else f"{x:.4f}")
    table_df['Final_R2'] = table_df['Final_R2'].apply(lambda x: f"{x:.3f}")
    
    # Rename columns
    table_df.columns = ['Behavioral Measure', 'N', 'Strategy', 'Model', 
                        'Spearman ρ', 'P-value', 'R²']
    
    # Save
    output_path = Path(output_dir)
    if min_rho is not None:
        filename = f'optimization_metrics_minrho{min_rho:.2f}.csv'
    else:
        filename = 'optimization_metrics.csv'
    
    csv_path = output_path / filename
    table_df.to_csv(csv_path, index=False)
    
    print_success(f"Saved metrics table: {csv_path.name}")
    print_info(f"Total measures: {len(table_df)}", 0)
    print_info(f"Significant (p<0.05): {len(df[df['Final_P_Value'] < 0.05])}", 0)
    
    return csv_path


def print_summary_statistics(df, cohort_name):
    """Print summary statistics."""
    print()
    print_section_header(f"SUMMARY STATISTICS - {cohort_name}")
    
    print_info(f"Total behavioral measures analyzed: {len(df)}")
    
    # Significance levels
    sig_001 = len(df[df['Final_P_Value'] < 0.001])
    sig_01 = len(df[(df['Final_P_Value'] >= 0.001) & (df['Final_P_Value'] < 0.01)])
    sig_05 = len(df[(df['Final_P_Value'] >= 0.01) & (df['Final_P_Value'] < 0.05)])
    non_sig = len(df[df['Final_P_Value'] >= 0.05])
    
    print_info(f"  p < 0.001: {sig_001}")
    print_info(f"  p < 0.01:  {sig_01}")
    print_info(f"  p < 0.05:  {sig_05}")
    print_info(f"  n.s.:      {non_sig}")
    
    # Spearman correlation range
    print()
    print_info(f"Spearman ρ range: [{df['Final_Spearman'].min():.3f}, {df['Final_Spearman'].max():.3f}]")
    print_info(f"Mean |ρ|: {df['Final_Spearman'].abs().mean():.3f}")
    print_info(f"Median |ρ|: {df['Final_Spearman'].abs().median():.3f}")
    
    # Best result
    best_idx = df['Final_Spearman'].abs().idxmax()
    best_row = df.loc[best_idx]
    print()
    print_info(f"Best correlation:")
    print(f"    Measure: {best_row['Measure']}")
    print(f"    ρ = {best_row['Final_Spearman']:.3f}, p = {best_row['Final_P_Value']:.4f}")
    print(f"    Strategy: {best_row['Best_Strategy']}, Model: {best_row['Best_Model']}")
    
    # Strategy distribution
    print()
    print_info("Strategy distribution:")
    strategy_counts = df['Best_Strategy'].value_counts()
    for strategy, count in strategy_counts.items():
        pct = 100 * count / len(df)
        print(f"    {strategy}: {count} ({pct:.1f}%)")
    
    print()


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def analyze_cohort(cohort_key, min_rho=None):
    """Analyze a single cohort."""
    config = COHORTS[cohort_key]
    
    print_section_header(f"OPTIMIZATION SUMMARY - {config['name'].upper()}")
    
    results_dir = Path(config['results_dir'])
    
    if not results_dir.exists():
        print_error(f"Results directory not found: {results_dir}")
        return False
    
    try:
        # Load optimization summary
        df = load_optimization_summary(results_dir)
        
        # Print statistics
        print_summary_statistics(df, config['name'])
        
        # Create visualizations
        create_summary_bar_plot(df, config['name'], results_dir, min_rho)
        create_detailed_table(df, config['name'], results_dir, min_rho)
        
        print()
        print_completion(f"{config['name']} summary complete!")
        print_info(f"Results saved to: {results_dir}")
        
        return True
        
    except Exception as e:
        print()
        print_error(f"Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create optimization summary figures and tables"
    )
    parser.add_argument(
        '--cohort', '-c',
        choices=list(COHORTS.keys()),
        help="Cohort to analyze"
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help="Analyze all cohorts"
    )
    parser.add_argument(
        '--min-rho',
        type=float,
        default=None,
        help="Minimum absolute Spearman rho to include in plots (e.g., 0.25)"
    )
    
    args = parser.parse_args()
    
    if not args.cohort and not args.all:
        parser.error("Must specify either --cohort or --all")
    
    # Determine which cohorts to process
    if args.all:
        cohorts_to_process = list(COHORTS.keys())
    else:
        cohorts_to_process = [args.cohort]
    
    # Process each cohort
    results = {}
    for cohort_key in cohorts_to_process:
        success = analyze_cohort(cohort_key, min_rho=args.min_rho)
        results[cohort_key] = success
        print("\n" + "="*100 + "\n")
    
    # Overall summary
    print_section_header("OVERALL SUMMARY")
    for cohort_key, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {COHORTS[cohort_key]['name']:.<50} {status}")
    print()


if __name__ == "__main__":
    main()
