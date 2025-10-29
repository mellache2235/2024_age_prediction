#!/usr/bin/env python3
"""
Optimized Brain-Behavior Analysis for NKI-RS TD Cohort

Based on run_nki_brain_behavior_enhanced.py with optimization strategies added.
Uses the exact same data loading logic as the enhanced version.

Usage:
    python run_nki_brain_behavior_optimized.py
    python run_nki_brain_behavior_optimized.py --max-measures 5

Author: Brain-Behavior Optimization Team
Date: 2024
"""

import sys
import os
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

# Setup Arial font globally
setup_arial_font()

# ============================================================================
# PRE-CONFIGURED PATHS (SAME AS ENHANCED VERSION)
# ============================================================================
DATASET = "nki_rs_td"
IG_CSV = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/integrated_gradients/nki_cog_dev_wIDS_features_IG_convnet_regressor_single_model_fold_0.csv"
BEHAVIORAL_DIR = "/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/FLUX/assessment_data"
OUTPUT_DIR = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/nki_rs_td_optimized"

# ============================================================================
# DATA LOADING FUNCTIONS (EXACT COPY FROM ENHANCED VERSION)
# ============================================================================

def load_nki_ig_scores(ig_csv):
    """Load NKI IG scores (exact copy from enhanced version)."""
    print_step("Loading IG scores", f"From {Path(ig_csv).name}")
    
    df = pd.read_csv(ig_csv)
    
    # Drop Unnamed: 0 if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    # Identify subject ID column
    id_col = None
    for col in ['subject_id', 'id', 'ID', 'Subject_ID']:
        if col in df.columns:
            id_col = col
            break
    
    if id_col is None:
        raise ValueError("No subject ID column found in IG CSV")
    
    # Standardize to 'subject_id'
    if id_col != 'subject_id':
        df = df.rename(columns={id_col: 'subject_id'})
    
    # Convert subject IDs to string
    df['subject_id'] = df['subject_id'].astype(str)
    
    # ROI columns are all columns except subject_id
    roi_cols = [col for col in df.columns if col != 'subject_id']
    
    print_info(f"IG subjects: {len(df)}")
    print_info(f"IG features (ROIs): {len(roi_cols)}")
    print_info(f"Sample ROI columns: {roi_cols[:3]}...")
    
    return df, roi_cols


def load_nki_behavioral_data(behavioral_dir):
    """Load NKI behavioral data (exact copy from enhanced version)."""
    print_step("Loading NKI behavioral data", f"From {Path(behavioral_dir).name}")
    
    behavioral_dir = Path(behavioral_dir)
    
    # Look for multiple behavioral files
    behavioral_files = []
    for pattern in ['*CAARS*.csv', '*Conners*.csv', '*RBS*.csv']:
        behavioral_files.extend(behavioral_dir.glob(pattern))
    
    if not behavioral_files:
        raise ValueError(f"No behavioral files found in {behavioral_dir}")
    
    print_info(f"Found {len(behavioral_files)} behavioral files:")
    for f in behavioral_files:
        print_info(f"  • {f.name}")
    
    # Load and merge all behavioral files
    all_dfs = []
    for bf in behavioral_files:
        df = pd.read_csv(bf)
        
        # Identify subject ID column
        id_col = None
        for col in df.columns:
            if 'id' in col.lower() or 'subject' in col.lower():
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
    
    print_info(f"Total behavioral subjects: {len(merged_df)}")
    print_info(f"Behavioral measures: {len(behavioral_cols)}")
    print_info(f"Sample columns: {behavioral_cols[:5]}...")
    
    return merged_df, behavioral_cols


def merge_data(ig_df, behavioral_df):
    """Merge IG and behavioral data by subject ID (exact copy from enhanced version)."""
    print_step("Merging data", "Matching subject IDs")
    
    print_info(f"IG subjects: {len(ig_df)}")
    print_info(f"Behavioral subjects: {len(behavioral_df)}")
    
    # Remove duplicates in behavioral data (keep first)
    behavioral_df = behavioral_df.drop_duplicates(subset='subject_id', keep='first')
    print_info(f"Behavioral subjects after deduplication: {len(behavioral_df)}")
    
    # Merge on subject_id
    merged = pd.merge(ig_df, behavioral_df, on='subject_id', how='inner')
    
    common_subjects = len(merged)
    print_success(f"Merged: {common_subjects} subjects with both IG and behavioral data")
    
    if common_subjects < 10:
        raise ValueError(f"Insufficient overlap: only {common_subjects} common subjects")
    
    return merged


# ============================================================================
# VISUALIZATION (OPTIMIZED VERSION - WITH STRATEGY INFO)
# ============================================================================

def create_scatter_plot(results, measure_name, best_params, output_dir):
    """Create scatter plot with optimization info."""
    y_actual = results['y_actual']
    y_pred = results['y_pred']
    rho = results['rho']
    p_value = results['p_value']
    
    p_str = "< 0.001" if p_value < 0.001 else f"= {p_value:.3f}"
    
    # Build model info text from best params
    strategy = best_params.get('strategy', 'Unknown')
    info_parts = [f"{strategy}"]
    if best_params.get('n_components'):
        info_parts.append(f"comp={best_params['n_components']}")
    if best_params.get('alpha'):
        info_parts.append(f"α={best_params['alpha']}")
    
    model_info = "\n".join(info_parts)
    stats_text = f"r = {rho:.3f}\np {p_str}\n{model_info}"
    
    title = get_dataset_title(DATASET) + " (Optimized)"
    safe_name = measure_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
    save_path = Path(output_dir) / f'scatter_{safe_name}_optimized'
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    create_standardized_scatter(
        ax, y_actual, y_pred,
        title=title,
        xlabel='Observed Behavioral Score',
        ylabel='Predicted Behavioral Score',
        stats_text=stats_text,
        is_subplot=False
    )
    
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
# MAIN ANALYSIS (ENHANCED + OPTIMIZATION)
# ============================================================================

def main():
    """Main analysis pipeline."""
    parser = argparse.ArgumentParser(description="NKI-RS TD optimized brain-behavior analysis")
    parser.add_argument('--max-measures', type=int, default=None,
                       help="Maximum number of behavioral measures to analyze (for testing)")
    args = parser.parse_args()
    
    print_section_header("OPTIMIZED BRAIN-BEHAVIOR ANALYSIS - NKI-RS TD")
    
    print_info(f"IG CSV:          {IG_CSV}")
    print_info(f"Behavioral DIR:  {BEHAVIORAL_DIR}")
    print_info(f"Output:          {OUTPUT_DIR}")
    print()
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Load data (using enhanced version logic)
        ig_df, roi_cols = load_nki_ig_scores(IG_CSV)
        behavioral_df, behavioral_cols = load_nki_behavioral_data(BEHAVIORAL_DIR)
        
        # 2. Merge data
        merged_df = merge_data(ig_df, behavioral_df)
        
        # Extract IG matrix
        ig_matrix = merged_df[roi_cols].values
        
        print_info(f"IG matrix shape: {ig_matrix.shape} (subjects x ROIs)")
        print()
        
        # Limit measures if specified
        if args.max_measures:
            behavioral_cols = behavioral_cols[:args.max_measures]
            print_warning(f"Limited to {args.max_measures} measures for testing")
        
        # 3. Analyze each behavioral measure with optimization
        all_results = []
        
        for measure in behavioral_cols:
            print()
            print_section_header(f"ANALYZING: {measure}")
            
            # Get behavioral scores (convert to numeric)
            behavioral_scores = pd.to_numeric(merged_df[measure], errors='coerce').values
            
            # Prepare data
            X = ig_matrix
            y = behavioral_scores
            
            # Remove NaN
            valid_mask = ~np.isnan(y)
            X_valid = X[valid_mask]
            y_valid = y[valid_mask]
            
            n_invalid = np.sum(~valid_mask)
            if n_invalid > 0:
                print_info(f"Removed {n_invalid} subjects with missing/invalid data")
            
            # Check sufficient data BEFORE outlier removal
            if len(y_valid) < 20:
                print_warning(f"Insufficient valid data: {len(y_valid)} subjects (need at least 20)")
                continue
            
            # Remove outliers
            X_valid, y_valid, n_outliers = remove_outliers(X_valid, y_valid)
            
            if n_outliers > 0:
                print_info(f"Removed {n_outliers} outliers")
            
            # Check again after outlier removal
            if len(y_valid) < 20:
                print_warning(f"Insufficient data after outlier removal: {len(y_valid)} subjects")
                continue
            
            print_info(f"Valid subjects: {len(y_valid)}")
            
            # Optimize
            best_model, best_params, cv_score, opt_results = \
                optimize_comprehensive(X_valid, y_valid, measure, verbose=True)
            
            # Evaluate
            eval_results = evaluate_model(best_model, X_valid, y_valid)
            
            # Create visualization
            create_scatter_plot(eval_results, measure, best_params, OUTPUT_DIR)
            
            # Save optimization results
            safe_name = measure.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
            opt_results.to_csv(Path(OUTPUT_DIR) / f"optimization_results_{safe_name}.csv", index=False)
            
            # Save predictions
            method_name = best_params.get('strategy', 'Unknown').replace('+', '_')
            predictions_df = pd.DataFrame({
                'Actual': eval_results['y_actual'],
                'Predicted': eval_results['y_pred'],
                'Residual': eval_results['y_actual'] - eval_results['y_pred']
            })
            pred_filename = f"predictions_{safe_name}_{method_name}.csv"
            predictions_df.to_csv(Path(OUTPUT_DIR) / pred_filename, index=False)
            
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
        
        # Save summary
        if all_results:
            summary_df = pd.DataFrame(all_results)
            summary_df.to_csv(Path(OUTPUT_DIR) / "optimization_summary.csv", index=False)
            
            print()
            print_completion("NKI-RS TD Optimized Analysis Complete!")
            print_info(f"Results saved to: {OUTPUT_DIR}")
            print()
            print("="*100)
            print("BEST PERFORMANCES (Sorted by Spearman ρ)")
            print("="*100)
            summary_sorted = summary_df.sort_values('Final_Spearman', ascending=False, key=abs)
            print(summary_sorted[['Measure', 'Final_Spearman', 'Best_Strategy', 'Best_Model']].to_string(index=False))
            print()
            print(f"\n  HIGHEST CORRELATION: ρ = {summary_sorted.iloc[0]['Final_Spearman']:.4f}")
            print(f"  Measure: {summary_sorted.iloc[0]['Measure']}")
            print()
        else:
            print_warning("No results generated - all measures had insufficient data")
        
    except Exception as e:
        print()
        print_error(f"Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
