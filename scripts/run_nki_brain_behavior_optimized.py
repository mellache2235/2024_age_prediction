#!/usr/bin/env python3
"""
Optimized Brain-Behavior Analysis for NKI-RS TD Cohort (FIXED VERSION)

This script performs brain-behavior correlation analysis with comprehensive optimization:
1. Loads NKI IG scores and behavioral data (CAARS, Conners, RBS)
2. Performs optimization across multiple strategies
3. Uses nested cross-validation for robust evaluation
4. Reports best parameters and performance

Usage:
    python run_nki_brain_behavior_optimized_FIXED.py
    python run_nki_brain_behavior_optimized_FIXED.py --max-measures 3

Copy this to Oak:
    scp run_nki_brain_behavior_optimized_FIXED.py oak:/oak/.../scripts/run_nki_brain_behavior_optimized.py

Author: Brain-Behavior Optimization Team
Date: 2024
"""

import os
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
                           print_warning, print_error, print_info, print_completion)
from plot_styles import create_standardized_scatter, get_dataset_title, setup_arial_font, DPI, FIGURE_FACECOLOR
from optimized_brain_behavior_core import optimize_comprehensive, evaluate_model, remove_outliers

# Setup Arial font
setup_arial_font()

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET = "nki_rs_td"
IG_CSV = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/integrated_gradients/nki_cog_dev_wIDS_features_IG_convnet_regressor_single_model_fold_0.csv"
BEHAVIORAL_DIR = Path("/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/FLUX/assessment_data")
OUTPUT_DIR = Path("/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/nki_rs_td_optimized")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA LOADING (FROM ENHANCED SCRIPT - EXACT LOGIC)
# ============================================================================

def load_ig_scores():
    """Load IG scores (from enhanced script logic)."""
    print_step("Loading IG scores", f"From {Path(IG_CSV).name}")
    
    ig_df = pd.read_csv(IG_CSV)
    
    # Extract subject IDs and IG scores
    subject_ids = ig_df['subject_id'].values
    ig_cols = [col for col in ig_df.columns if col.startswith('ROI_') or col != 'subject_id']
    
    # Remove subject_id from ig_cols if it got included
    ig_cols = [col for col in ig_cols if col != 'subject_id']
    
    ig_matrix = ig_df[ig_cols].values
    
    print_info("IG subjects", len(subject_ids))
    print_info("IG features (ROIs)", ig_matrix.shape[1])
    
    return subject_ids, ig_matrix, ig_cols


def load_nki_behavioral_data():
    """Load and merge all NKI behavioral data files (from enhanced script - EXACT LOGIC)."""
    print_step("Loading NKI behavioral data", "From multiple files")
    
    # Find all behavioral files
    behavioral_files = []
    for pattern in ['*CAARS*.csv', '*Conners*.csv', '*RBS*.csv']:
        behavioral_files.extend(BEHAVIORAL_DIR.glob(pattern))
    
    if not behavioral_files:
        raise ValueError(f"No behavioral files found in {BEHAVIORAL_DIR}")
    
    print_info("Found files", len(behavioral_files))
    for f in behavioral_files:
        print(f"  • {f.name}")
    
    # Load and merge all behavioral files
    all_dfs = []
    for bf in behavioral_files:
        df = pd.read_csv(bf)
        
        # Identify subject ID column (FLEXIBLE - includes 'anonymized')
        id_col = None
        for col in df.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in ['id', 'subject', 'identifier', 'anonymized']):
                id_col = col
                break
        
        if id_col is None:
            print_warning(f"No subject ID column in {bf.name}, skipping")
            continue
        
        # Standardize to 'subject_id'
        if id_col != 'subject_id':
            df = df.rename(columns={id_col: 'subject_id'})
        
        # Convert subject IDs to string
        df['subject_id'] = df['subject_id'].astype(str)
        
        all_dfs.append(df)
    
    # Merge all dataframes on subject_id
    if not all_dfs:
        raise ValueError("No valid behavioral files found")
    
    merged_df = all_dfs[0]
    for df in all_dfs[1:]:
        merged_df = pd.merge(merged_df, df, on='subject_id', how='outer')
    
    # Auto-detect behavioral columns (CAARS, Conners, RBS)
    behavioral_cols = [col for col in merged_df.columns if 
                       'CAARS' in col.upper() or 
                       'CONNERS' in col.upper() or
                       'RBS' in col.upper() or
                       ('TOTAL' in col.upper() and ('INATTENTION' in col.upper() or 'HYPERACTIVITY' in col.upper())) or
                       ('T-SCORE' in col.upper() or 'T_SCORE' in col.upper())]
    
    # Exclude subject_id if it got included
    behavioral_cols = [col for col in behavioral_cols if col != 'subject_id']
    
    if not behavioral_cols:
        raise ValueError("No behavioral columns found")
    
    print_info("Behavioral subjects", len(merged_df))
    print_info("Behavioral measures", len(behavioral_cols))
    print(f"  Sample columns: {behavioral_cols[:5]}...")
    
    return merged_df, behavioral_cols


def merge_data(ig_df, beh_df):
    """Merge IG and behavioral data."""
    print_step("Merging data", "Matching subject IDs")
    
    # Remove duplicates in behavioral data
    beh_df = beh_df.drop_duplicates(subset='subject_id', keep='first')
    
    merged = pd.merge(ig_df, beh_df, on='subject_id', how='inner')
    
    print_success(f"Merged: {len(merged)} subjects with both IG and behavioral data")
    
    if len(merged) < 10:
        raise ValueError(f"Insufficient overlap: only {len(merged)} common subjects")
    
    return merged


def create_scatter_plot(results, measure_name, best_params, output_dir):
    """Create scatter plot using centralized styling."""
    y_actual = results['y_actual']
    y_pred = results['y_pred']
    rho = results['rho']
    p_value = results['p_value']
    
    # Format p-value
    p_str = "< 0.001" if p_value < 0.001 else f"= {p_value:.3f}"
    
    # Create stats text with model info
    strategy = best_params.get('strategy', 'Unknown')
    info_parts = [f"{strategy}"]
    if best_params.get('n_components'):
        info_parts.append(f"comp={best_params['n_components']}")
    if best_params.get('alpha'):
        info_parts.append(f"α={best_params['alpha']}")
    
    model_info = "\n".join(info_parts)
    stats_text = f"r = {rho:.3f}\np {p_str}\n{model_info}"
    
    # Get standardized title
    title = get_dataset_title(DATASET)
    
    # Create descriptive filename with method
    safe_name = measure_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
    method_name = best_params.get('strategy', 'Unknown').replace('+', '_')
    
    if best_params.get('n_components'):
        filename = f'scatter_{safe_name}_{method_name}_comp{best_params["n_components"]}_optimized'
    elif best_params.get('n_features'):
        filename = f'scatter_{safe_name}_{method_name}_k{best_params["n_features"]}_optimized'
    else:
        filename = f'scatter_{safe_name}_{method_name}_optimized'
    
    save_path = OUTPUT_DIR / filename
    
    # Use centralized plotting function
    fig, ax = plt.subplots(figsize=(6, 6))
    
    create_standardized_scatter(
        ax, y_actual, y_pred,
        title=title + " (Optimized)",
        xlabel='Observed Behavioral Score',
        ylabel='Predicted Behavioral Score',
        stats_text=stats_text,
        is_subplot=False
    )
    
    # Save with centralized export (PNG + TIFF + AI)
    plt.tight_layout()
    
    png_path = save_path.with_suffix('.png')
    tiff_path = save_path.with_suffix('.tiff')
    ai_path = save_path.with_suffix('.ai')
    
    plt.savefig(png_path, dpi=DPI, bbox_inches='tight', facecolor=FIGURE_FACECOLOR, edgecolor='none')
    plt.savefig(tiff_path, dpi=DPI, bbox_inches='tight', facecolor=FIGURE_FACECOLOR, edgecolor='none',
               format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
    pdf_backend.FigureCanvas(fig).print_pdf(str(ai_path))
    
    plt.close()
    
    print(f"  ✓ Saved: {png_path.name}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Optimized brain-behavior analysis for NKI-RS TD"
    )
    parser.add_argument(
        '--max-measures', '-m',
        type=int,
        default=None,
        help="Maximum number of behavioral measures to analyze (for testing)"
    )
    
    args = parser.parse_args()
    
    print_section_header("OPTIMIZED BRAIN-BEHAVIOR ANALYSIS - NKI-RS TD")
    print()
    
    try:
        # Load data
        subject_ids_ig, ig_matrix, ig_cols = load_ig_scores()
        behavioral_df, behavioral_cols = load_nki_behavioral_data()
        
        print()
        
        # Merge datasets
        ig_df_full = pd.DataFrame({
            'subject_id': subject_ids_ig,
            **{col: ig_matrix[:, i] for i, col in enumerate(ig_cols)}
        })
        
        merged_df = merge_data(ig_df_full, behavioral_df)
        
        print()
        
        X = merged_df[ig_cols].values
        
        print_info("Final IG matrix shape", f"{X.shape} (subjects x ROIs)")
        
        # Limit measures if specified
        if args.max_measures:
            behavioral_cols = behavioral_cols[:args.max_measures]
            print_warning(f"Limited to {args.max_measures} measures for testing")
        
        print()
        
        # Analyze each behavioral measure
        all_results = []
        
        for measure in behavioral_cols:
            print_section_header(f"ANALYZING: {measure}")
            
            # Get behavioral scores for this measure
            y = merged_df[measure].values
            
            # Remove NaN values
            valid_mask = ~np.isnan(y)
            X_valid = X[valid_mask]
            y_valid = y[valid_mask]
            
            # Remove outliers
            X_valid, y_valid, n_outliers = remove_outliers(X_valid, y_valid)
            
            if n_outliers > 0:
                print_info("Removed outliers", n_outliers)
            
            if len(y_valid) < 20:
                print_warning(f"Insufficient data for {measure}: only {len(y_valid)} subjects")
                continue
            
            print_info("Valid subjects", len(y_valid))
            print()
            
            # Optimize model
            best_model, best_params, cv_score, opt_results = \
                optimize_comprehensive(X_valid, y_valid, measure, verbose=True)
            
            # Evaluate on all data with integrity checking
            eval_results = evaluate_model(best_model, X_valid, y_valid, verbose=True)
            
            # Create visualization
            create_scatter_plot(eval_results, measure, best_params, OUTPUT_DIR)
            
            # Save optimization results
            safe_name = measure.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
            method_name = best_params.get('strategy', 'Unknown').replace('+', '_')
            
            opt_results.to_csv(OUTPUT_DIR / f"optimization_results_{safe_name}.csv", index=False)
            
            # Save predictions for integrity verification
            predictions_df = pd.DataFrame({
                'Actual': eval_results['y_actual'],
                'Predicted': eval_results['y_pred'],
                'Residual': eval_results['y_actual'] - eval_results['y_pred']
            })
            pred_filename = f"predictions_{safe_name}_{method_name}.csv"
            predictions_df.to_csv(OUTPUT_DIR / pred_filename, index=False)
            print_info("Saved predictions", pred_filename)
            
            # Store summary
            all_results.append({
                'Measure': measure,
                'N_Subjects': len(y_valid),
                'N_Outliers_Removed': n_outliers,
                'Best_Strategy': best_params.get('strategy', 'N/A'),
                'Best_Model': best_params.get('model', 'N/A'),
                'Best_N_Components': best_params.get('n_components', None),
                'Best_Alpha': best_params.get('alpha', None),
                'Feature_Selection': best_params.get('feature_selection', None),
                'N_Features': best_params.get('n_features', None),
                'CV_Spearman': cv_score,
                'Final_Spearman': eval_results['rho'],
                'Final_P_Value': eval_results['p_value'],
                'Final_R2': eval_results['r2'],
                'Final_MAE': eval_results['mae']
            })
            
            print()
        
        # Save summary
        if all_results:
            summary_df = pd.DataFrame(all_results)
            summary_df.to_csv(OUTPUT_DIR / "optimization_summary.csv", index=False)
            
            print()
            print_completion("NKI-RS TD Brain-Behavior Analysis Complete!")
            print_info("Results saved to", str(OUTPUT_DIR))
            print()
            print("="*100)
            print("BEST PERFORMANCES (Sorted by Spearman ρ)")
            print("="*100)
            summary_sorted = summary_df.sort_values('Final_Spearman', ascending=False)
            print(summary_sorted[['Measure', 'Final_Spearman', 'Best_Strategy', 'Best_Model']].to_string(index=False))
            print()
            print(f"\n  HIGHEST CORRELATION: ρ = {summary_sorted.iloc[0]['Final_Spearman']:.4f}")
            print(f"  Measure: {summary_sorted.iloc[0]['Measure']}")
            print()
        else:
            print_warning("No results to save")
        
    except Exception as e:
        print()
        print_error(f"Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

