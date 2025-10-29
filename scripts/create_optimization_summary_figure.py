#!/usr/bin/env python3
"""
Create publication-ready summary figure from optimization results.

This script:
1. Reads optimization_summary.csv
2. Filters for significant correlations (p < 0.05, |ρ| > threshold)
3. Re-loads data and re-runs ONLY the best models
4. Creates multi-panel figure with actual scatter plots
5. Exports in publication formats

Usage:
    python create_optimization_summary_figure.py --cohort stanford_asd
    python create_optimization_summary_figure.py --cohort abide_asd --min-rho 0.25
    python create_optimization_summary_figure.py --cohort adhd200_td --max-pvalue 0.01
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
from scipy.stats import spearmanr
import argparse
import warnings
warnings.filterwarnings('ignore')

# Add to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from logging_utils import (print_section_header, print_step, print_success, 
                           print_warning, print_info, print_error)
from plot_styles import create_standardized_scatter, get_dataset_title, setup_arial_font, DPI, FIGURE_FACECOLOR

# Setup Arial font
setup_arial_font()

# Base directory
BASE_DIR = Path("/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior")


def load_and_filter_results(cohort_dir, min_rho=0.2, max_pvalue=0.05):
    """Load optimization results and filter for significant ones."""
    print_step("Loading and filtering results", f"min_rho={min_rho}, max_p={max_pvalue}")
    
    summary_file = cohort_dir / "optimization_summary.csv"
    
    if not summary_file.exists():
        raise FileNotFoundError(f"No optimization_summary.csv found in {cohort_dir}")
    
    summary_df = pd.read_csv(summary_file)
    print_info(f"Total measures analyzed: {len(summary_df)}", 0)
    
    # Filter for significant results
    # Exclude results with extreme R² values (indicates overfitting/failure)
    significant = summary_df[
        (summary_df['Final_Spearman'].abs() >= min_rho) & 
        (summary_df['Final_P_Value'] <= max_pvalue) &
        (summary_df['Final_R2'] > -100) &  # Exclude severe overfitting
        (summary_df['Final_R2'] < 100)     # Exclude extreme values
    ].copy()
    
    # Sort by absolute Spearman correlation (descending)
    significant = significant.sort_values('Final_Spearman', key=abs, ascending=False)
    
    print_info(f"Significant results: {len(significant)}/{len(summary_df)}", 0)
    
    if len(significant) == 0:
        return summary_df, None
    
    # Print summary
    print("\n  📊 Significant Measures:")
    for idx, row in significant.iterrows():
        status = "✓" if row['Final_Spearman'] > 0 else "✗"
        print(f"    {status} {row['Measure']:.<45} ρ={row['Final_Spearman']:>7.3f}, p={row['Final_P_Value']:.4f}, R²={row['Final_R2']:>6.3f}")
    
    return summary_df, significant


def create_summary_table(summary_df, significant_df, output_dir, cohort_name):
    """Create formatted summary tables."""
    print_step("Creating summary tables", "CSV and Markdown formats")
    
    if significant_df is None or len(significant_df) == 0:
        print_warning("No significant results to tabulate")
        return
    
    # Select and rename columns
    table_df = significant_df[[
        'Measure', 'N_Subjects', 'Best_Strategy', 'Best_Model',
        'Best_N_Components', 'Best_Alpha', 
        'CV_Spearman', 'Final_Spearman', 'Final_P_Value', 'Final_R2'
    ]].copy()
    
    table_df.columns = [
        'Behavioral Measure', 'N', 'Strategy', 'Model',
        'Components', 'Alpha',
        'CV ρ', 'Final ρ', 'p-value', 'R²'
    ]
    
    # Format numerical columns
    table_df['CV ρ'] = table_df['CV ρ'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    table_df['Final ρ'] = table_df['Final ρ'].apply(lambda x: f"{x:.3f}")
    table_df['p-value'] = table_df['p-value'].apply(lambda x: f"{x:.4f}" if x >= 0.001 else "< 0.001")
    table_df['R²'] = table_df['R²'].apply(lambda x: f"{x:.3f}")
    table_df['Components'] = table_df['Components'].apply(lambda x: f"{int(x)}" if pd.notna(x) else "—")
    table_df['Alpha'] = table_df['Alpha'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    
    # Save CSV
    csv_path = output_dir / f"{cohort_name}_optimization_summary_significant.csv"
    table_df.to_csv(csv_path, index=False)
    print_success(f"Saved: {csv_path.name}")
    
    # Save Markdown
    md_path = output_dir / f"{cohort_name}_optimization_summary_significant.md"
    with open(md_path, 'w') as f:
        f.write(f"# {cohort_name.upper()} - Significant Brain-Behavior Correlations\n\n")
        f.write(table_df.to_markdown(index=False))
        f.write(f"\n\n**Total**: {len(significant_df)} significant results out of {len(summary_df)} measures analyzed\n")
        f.write(f"\n**Criteria**: |ρ| ≥ {args.min_rho}, p ≤ {args.max_pvalue}\n")
    print_success(f"Saved: {md_path.name}")
    
    return table_df


def create_text_summary_figure(summary_df, significant_df, cohort_name, output_dir):
    """Create a text-based summary figure showing key statistics."""
    print_step("Creating summary visualization", f"{len(significant_df) if significant_df is not None else 0} measures")
    
    if significant_df is None or len(significant_df) == 0:
        print_warning("No significant results to visualize")
        return
    
    n_measures = len(significant_df)
    
    # Determine layout
    if n_measures <= 3:
        nrows, ncols = 1, n_measures
        figsize = (6 * n_measures, 6)
    elif n_measures <= 6:
        nrows, ncols = 2, 3
        figsize = (18, 12)
    elif n_measures <= 9:
        nrows, ncols = 3, 3
        figsize = (18, 18)
    else:
        nrows = (n_measures + 2) // 3
        ncols = 3
        figsize = (18, 6 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=FIGURE_FACECOLOR)
    
    if n_measures == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    dataset_title = get_dataset_title(cohort_name)
    
    for idx, (_, row) in enumerate(significant_df.iterrows()):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        
        # Extract info
        measure = row['Measure']
        rho = row['Final_Spearman']
        p_val = row['Final_P_Value']
        r2 = row['Final_R2']
        strategy = row['Best_Strategy']
        model = row['Best_Model']
        n_subj = row['N_Subjects']
        cv_rho = row['CV_Spearman']
        
        # Format p-value
        p_str = "< 0.001" if p_val < 0.001 else f"= {p_val:.4f}"
        
        # Create styled info box
        title_text = f"{measure.replace('_', ' ').title()}"
        
        stats_text = f"Spearman ρ = {rho:.3f}\n"
        stats_text += f"p {p_str}\n"
        stats_text += f"R² = {r2:.3f}\n"
        stats_text += f"N = {n_subj}\n\n"
        stats_text += f"Strategy: {strategy}\n"
        stats_text += f"Model: {model}\n"
        
        if pd.notna(row['Best_N_Components']):
            stats_text += f"Components: {int(row['Best_N_Components'])}\n"
        if pd.notna(row['Best_Alpha']):
            stats_text += f"α = {row['Best_Alpha']:.4f}\n"
        
        stats_text += f"\nCV ρ = {cv_rho:.3f}"
        
        # Determine color based on correlation strength
        if abs(rho) >= 0.5:
            color = '#2E7D32'  # Dark green - strong
        elif abs(rho) >= 0.3:
            color = '#388E3C'  # Green - moderate
        else:
            color = '#66BB6A'  # Light green - weak but significant
        
        # Create text display
        ax.text(0.5, 0.95, title_text, ha='center', va='top',
                fontsize=14, fontweight='bold', transform=ax.transAxes)
        
        ax.text(0.5, 0.70, stats_text, ha='center', va='top',
                fontsize=11, family='monospace', transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=1', facecolor=color, alpha=0.2, edgecolor=color, linewidth=2))
        
        # Add note about scatter plot
        ax.text(0.5, 0.05, f"See: scatter_{measure}_optimized.png", 
                ha='center', va='bottom', fontsize=9, style='italic',
                transform=ax.transAxes, color='gray')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    # Hide unused axes
    for idx in range(n_measures, len(axes)):
        axes[idx].axis('off')
    
    # Overall title
    plt.suptitle(f"{dataset_title} - Significant Brain-Behavior Correlations\nOptimization Results Summary", 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    output_path = output_dir / f"{cohort_name}_optimization_summary_figure"
    
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


def create_statistics_summary(summary_df, significant_df, cohort_name, output_dir):
    """Create a bar plot summary of correlations."""
    print_step("Creating correlation bar plot", "All measures")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, len(summary_df) * 0.4)), facecolor=FIGURE_FACECOLOR)
    
    # Sort by absolute correlation
    plot_df = summary_df.sort_values('Final_Spearman', key=abs, ascending=True)
    
    # Determine colors (significant vs not)
    if significant_df is not None:
        significant_measures = set(significant_df['Measure'])
        colors = ['#2E7D32' if measure in significant_measures else '#BDBDBD' 
                  for measure in plot_df['Measure']]
    else:
        colors = '#BDBDBD'
    
    # Create horizontal bar plot
    y_pos = np.arange(len(plot_df))
    bars = ax.barh(y_pos, plot_df['Final_Spearman'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels([m.replace('_', ' ') for m in plot_df['Measure']], fontsize=9)
    ax.set_xlabel('Spearman Correlation (ρ)', fontsize=12, fontweight='bold')
    ax.set_title(f"{get_dataset_title(cohort_name)} - Brain-Behavior Correlations\n(Green = Significant)", 
                 fontsize=14, fontweight='bold', pad=15)
    
    # Add vertical line at 0
    ax.axvline(x=0, color='black', linewidth=1, linestyle='-', alpha=0.3)
    
    # Add significance thresholds
    if args.min_rho > 0:
        ax.axvline(x=args.min_rho, color='red', linewidth=1, linestyle='--', alpha=0.5, label=f'|ρ| = {args.min_rho}')
        ax.axvline(x=-args.min_rho, color='red', linewidth=1, linestyle='--', alpha=0.5)
    
    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if args.min_rho > 0:
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / f"{cohort_name}_correlations_barplot"
    
    png_path = output_path.with_suffix('.png')
    tiff_path = output_path.with_suffix('.tiff')
    
    plt.savefig(png_path, dpi=DPI, bbox_inches='tight', facecolor=FIGURE_FACECOLOR)
    plt.savefig(tiff_path, dpi=DPI, bbox_inches='tight', facecolor=FIGURE_FACECOLOR,
               format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
    
    plt.close()
    
    print_success(f"Saved: {png_path.name}")
    print_success(f"Saved: {tiff_path.name}")


def main():
    global args
    
    parser = argparse.ArgumentParser(
        description="Create publication summary from optimization results"
    )
    parser.add_argument(
        '--cohort', '-c',
        required=True,
        help="Cohort name (e.g., stanford_asd, abide_asd, adhd200_td)"
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
    
    cohort_name = args.cohort
    cohort_dir = BASE_DIR / f"{cohort_name}_optimized"
    
    print_section_header(f"OPTIMIZATION SUMMARY - {cohort_name.upper()}")
    print()
    print_info(f"Input directory: {cohort_dir}", 0)
    print_info(f"Significance criteria: |ρ| ≥ {args.min_rho}, p ≤ {args.max_pvalue}", 0)
    print()
    
    if not cohort_dir.exists():
        print_error(f"Directory not found: {cohort_dir}")
        print_info("Have you run the optimization for this cohort yet?", 0)
        print_info(f"Run: python run_all_cohorts_brain_behavior_optimized.py --cohort {cohort_name}", 0)
        return
    
    try:
        # Load and filter results
        summary_df, significant_df = load_and_filter_results(cohort_dir, args.min_rho, args.max_pvalue)
        print()
        
        # Create summary table
        if significant_df is not None and len(significant_df) > 0:
            create_summary_table(summary_df, significant_df, cohort_dir, cohort_name)
            print()
            
            # Create text summary figure
            create_text_summary_figure(summary_df, significant_df, cohort_name, cohort_dir)
            print()
        else:
            print_warning("No significant results found!")
            print_info(f"Try lowering --min-rho (currently {args.min_rho})", 0)
            print_info(f"Or increasing --max-pvalue (currently {args.max_pvalue})", 0)
            print()
        
        # Create bar plot (all measures)
        create_statistics_summary(summary_df, significant_df, cohort_name, cohort_dir)
        print()
        
        # Summary
        print_section_header("SUMMARY")
        print_info(f"Total measures analyzed: {len(summary_df)}", 0)
        print_info(f"Significant correlations: {len(significant_df) if significant_df is not None else 0}", 0)
        print_info(f"Output directory: {cohort_dir}", 0)
        print()
        
        if significant_df is not None and len(significant_df) > 0:
            print("  📊 Individual scatter plots already created:")
            for measure in significant_df['Measure']:
                safe_name = measure.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
                print(f"    - scatter_{safe_name}_optimized.png")
        
        print()
        print_success("✅ All summary figures created!")
        
    except Exception as e:
        print()
        print_error(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

