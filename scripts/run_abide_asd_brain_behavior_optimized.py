#!/usr/bin/env python3
"""
Optimized Brain-Behavior Analysis for ABIDE ASD Cohort

Uses EXACT same data loading logic as enhanced script, but adds comprehensive optimization.

Usage:
    python run_abide_asd_brain_behavior_optimized.py
    python run_abide_asd_brain_behavior_optimized.py --max-measures 2
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
import argparse
import warnings
warnings.filterwarnings('ignore')

# Add to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from logging_utils import (print_section_header, print_step, print_success, 
                           print_warning, print_error, print_info, print_completion)
from plot_styles import create_standardized_scatter, get_dataset_title, setup_arial_font, DPI, FIGURE_FACECOLOR
from optimized_brain_behavior_core import optimize_comprehensive, evaluate_model, remove_outliers, apply_fdr_correction

# Setup Arial font
setup_arial_font()

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET = "abide_asd"
IG_CSV = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/integrated_gradients/abide_asd_features_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS.csv"
PKLZ_DIR = "/oak/stanford/groups/menon/deriveddata/public/abide/restfmri/timeseries/group_level/brainnetome/normz/"
OUTPUT_DIR = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/abide_asd_optimized"

ABIDE_SITES = ['NYU', 'SDSU', 'STANFORD', 'Stanford', 'TCD-1', 'UM', 'USM', 'Yale']

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# DATA LOADING (EXACT SAME AS ENHANCED SCRIPT)
# ============================================================================

def load_abide_ig_scores(ig_csv):
    """Load ABIDE ASD IG scores (from enhanced script)."""
    print_step("Loading IG scores", f"From {Path(ig_csv).name}")
    
    df = pd.read_csv(ig_csv)
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    id_col = None
    for col in ['subjid', 'subject_id', 'id', 'ID', 'Subject_ID']:
        if col in df.columns:
            id_col = col
            break
    
    if id_col is None:
        raise ValueError("No subject ID column found in IG CSV")
    
    print_info(f"Using ID column: {id_col}")
    
    if id_col != 'subject_id':
        df = df.rename(columns={id_col: 'subject_id'})
    
    df['subject_id'] = df['subject_id'].astype(str)
    roi_cols = [col for col in df.columns if col != 'subject_id']
    
    print_info(f"IG subjects: {len(df)}")
    print_info(f"IG features (ROIs): {len(roi_cols)}")
    
    return df, roi_cols


def load_abide_behavioral_data(pklz_dir, sites):
    """Load ABIDE behavioral data (from enhanced script - EXACT logic)."""
    print_step("Loading ABIDE behavioral data", f"From {Path(pklz_dir).name}")
    
    pklz_dir = Path(pklz_dir)
    
    # Find all .pklz files matching sites AND ending with 246ROIs.pklz
    all_files = os.listdir(pklz_dir)
    filtered_files = []
    for file_name in all_files:
        if '246ROIs.pklz' in file_name and any(site in file_name for site in sites):
            filtered_files.append(pklz_dir / file_name)
    
    if not filtered_files:
        raise ValueError(f"No 246ROIs.pklz files found for sites: {sites}")
    
    print_info(f"Found {len(filtered_files)} 246ROIs.pklz files for {len(sites)} sites")
    
    # Load and concatenate all dataframes
    appended_data = []
    for file_path in filtered_files:
        try:
            data = pd.read_pickle(file_path)
            data = data[~pd.isna(data)]
            appended_data.append(data)
            print_info(f"  Loaded {len(data)} subjects from {file_path.name}")
        except Exception as e:
            print_warning(f"  Could not load {file_path.name}: {e}")
    
    if not appended_data:
        raise ValueError("No data loaded from .pklz files")
    
    combined_df = pd.concat(appended_data, ignore_index=True)
    combined_df['label'] = combined_df['label'].astype(str)
    
    print_info(f"Total subjects loaded: {len(combined_df)}")
    
    # Filter for ASD subjects only (label == '1')
    asd_df = combined_df[combined_df['label'] == '1'].copy()
    print_info(f"ASD subjects (label='1'): {len(asd_df)}")
    
    # Filter for developmental age (age <= 21)
    if 'age' in asd_df.columns:
        asd_df['age'] = pd.to_numeric(asd_df['age'], errors='coerce')
        asd_df = asd_df[asd_df['age'] <= 21]
        print_info(f"ASD subjects age <= 21: {len(asd_df)}")
    
    asd_df = asd_df.reset_index(drop=True)
    
    # Find subject ID column
    id_col = None
    for col in ['subjid', 'subject_id', 'id', 'ID', 'Subject_ID']:
        if col in asd_df.columns:
            id_col = col
            break
    
    if id_col is None:
        raise ValueError(f"No subject ID column found")
    
    print_info(f"Using ID column from .pklz: {id_col}")
    
    if id_col != 'subject_id':
        asd_df = asd_df.rename(columns={id_col: 'subject_id'})
    
    asd_df['subject_id'] = asd_df['subject_id'].astype(str)
    
    # Check for ADOS columns
    target_ados = ['ados_total', 'ados_social', 'ados_comm']
    ados_cols = []
    
    print_info(f"Looking for ADOS columns: {target_ados}")
    
    for target in target_ados:
        if target in asd_df.columns:
            ados_cols.append(target)
            print_info(f"  Found: {target}")
        else:
            found = False
            for col in asd_df.columns:
                if col.lower() == target:
                    ados_cols.append(col)
                    print_info(f"  Found: {col} (matched {target})")
                    found = True
                    break
            if not found:
                print_warning(f"  NOT FOUND: {target}")
    
    if ados_cols:
        print_info(f"ADOS behavioral measures found: {ados_cols}")
        # Convert ADOS columns to numeric and filter missing data codes
        for col in ados_cols:
            asd_df[col] = pd.to_numeric(asd_df[col], errors='coerce')
            
            # Replace missing data codes
            asd_df.loc[asd_df[col] < 0, col] = np.nan
            asd_df.loc[asd_df[col] > 100, col] = np.nan
            
            non_null = asd_df[col].notna().sum()
            print_info(f"  {col}: {non_null} non-null values out of {len(asd_df)} (after removing missing codes)")
            if non_null > 0:
                print_info(f"    Range: [{asd_df[col].min():.2f}, {asd_df[col].max():.2f}]")
    
    return asd_df, ados_cols


def merge_data(ig_df, behavioral_df):
    """Merge IG and behavioral data (from enhanced script - handles ID stripping)."""
    print_step("Merging data", "Matching subject IDs")
    
    print_info(f"IG subjects: {len(ig_df)}")
    print_info(f"Behavioral subjects: {len(behavioral_df)}")
    
    # Debug: Check ID formats
    print_info(f"Sample IG IDs: {list(ig_df['subject_id'].head(5))}")
    print_info(f"Sample behavioral IDs: {list(behavioral_df['subject_id'].head(5))}")
    
    # Remove duplicates in behavioral data
    behavioral_df = behavioral_df.drop_duplicates(subset='subject_id', keep='first')
    print_info(f"Behavioral subjects after deduplication: {len(behavioral_df)}")
    
    # Check for overlapping IDs
    ig_ids = set(ig_df['subject_id'])
    behav_ids = set(behavioral_df['subject_id'])
    overlap_ids = ig_ids.intersection(behav_ids)
    print_info(f"Overlapping subject IDs (exact match): {len(overlap_ids)}")
    
    # If poor overlap, try stripping leading zeros (ABIDE often has this issue)
    if len(overlap_ids) < 200:
        print_warning(f"Few overlapping IDs ({len(overlap_ids)})! Trying to strip leading zeros...")
        
        behavioral_df_stripped = behavioral_df.copy()
        behavioral_df_stripped['subject_id_stripped'] = behavioral_df_stripped['subject_id'].str.lstrip('0')
        
        ig_df_stripped = ig_df.copy()
        ig_df_stripped['subject_id_stripped'] = ig_df_stripped['subject_id'].str.lstrip('0')
        
        overlap_stripped = set(ig_df_stripped['subject_id_stripped']).intersection(
            set(behavioral_df_stripped['subject_id_stripped']))
        print_info(f"Overlapping IDs after stripping zeros: {len(overlap_stripped)}")
        
        if len(overlap_stripped) > len(overlap_ids):
            print_success(f"✓ Better overlap with stripped IDs! Using stripped IDs for merge.")
            merged = pd.merge(ig_df_stripped.drop(columns=['subject_id']), 
                            behavioral_df_stripped.drop(columns=['subject_id']), 
                            on='subject_id_stripped', how='inner')
            merged = merged.rename(columns={'subject_id_stripped': 'subject_id'})
        else:
            print_warning(f"No improvement with stripped IDs. Using original merge.")
            merged = pd.merge(ig_df, behavioral_df, on='subject_id', how='inner')
    else:
        print_info(f"Good overlap ({len(overlap_ids)} subjects). Using exact ID matching.")
        merged = pd.merge(ig_df, behavioral_df, on='subject_id', how='inner')
    
    print_success(f"Merged: {len(merged)} subjects with both IG and behavioral data")
    
    if len(merged) < 10:
        raise ValueError(f"Insufficient overlap: only {len(merged)} common subjects")
    
    return merged


def create_scatter_plot(results, measure_name, best_params, output_dir):
    """Create scatter plot with method in filename."""
    y_actual = results['y_actual']
    y_pred = results['y_pred']
    rho = results['rho']
    p_value = results['p_value']
    
    p_str = "< 0.001" if p_value < 0.001 else f"= {p_value:.3f}"
    
    strategy = best_params.get('strategy', 'Unknown')
    info_parts = [f"{strategy}"]
    if best_params.get('n_components'):
        info_parts.append(f"comp={best_params['n_components']}")
    if best_params.get('alpha'):
        info_parts.append(f"α={best_params['alpha']}")
    
    model_info = "\n".join(info_parts)
    stats_text = f"r = {rho:.3f}\np {p_str}\n{model_info}"
    
    title = get_dataset_title(DATASET)
    safe_name = measure_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
    method_name = best_params.get('strategy', 'Unknown').replace('+', '_')
    
    # Descriptive filename with method
    if best_params.get('n_components'):
        filename = f'scatter_{safe_name}_{method_name}_comp{best_params["n_components"]}_optimized'
    elif best_params.get('n_features'):
        filename = f'scatter_{safe_name}_{method_name}_k{best_params["n_features"]}_optimized'
    else:
        filename = f'scatter_{safe_name}_{method_name}_optimized'
    
    save_path = Path(output_dir) / filename
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    create_standardized_scatter(
        ax, y_actual, y_pred,
        title=title + " (Optimized)",
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
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Optimized brain-behavior analysis for ABIDE ASD")
    parser.add_argument('--max-measures', '-m', type=int, default=None,
                       help="Maximum number of behavioral measures to analyze (for testing)")
    args = parser.parse_args()
    
    print_section_header("OPTIMIZED BRAIN-BEHAVIOR ANALYSIS - ABIDE ASD")
    print()
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Load data (using enhanced script logic)
        ig_df, roi_cols = load_abide_ig_scores(IG_CSV)
        behavioral_df, ados_cols = load_abide_behavioral_data(PKLZ_DIR, ABIDE_SITES)
        
        # 2. Merge data (with ID stripping logic)
        merged_df = merge_data(ig_df, behavioral_df)
        
        # Extract IG matrix
        X = merged_df[roi_cols].values
        
        print_info(f"IG matrix shape: {X.shape} (subjects x ROIs)")
        print_info(f"ADOS measures to analyze: {ados_cols}")
        
        # Limit measures if specified
        if args.max_measures:
            ados_cols = ados_cols[:args.max_measures]
            print_warning(f"Limited to {args.max_measures} measures for testing")
        
        print()
        
        # 3. Analyze each ADOS measure with optimization
        all_results = []
        
        for measure in ados_cols:
            print_section_header(f"ANALYZING: {measure}")
            
            y = pd.to_numeric(merged_df[measure], errors='coerce').values
            
            # Remove NaN
            valid_mask = ~np.isnan(y)
            X_valid = X[valid_mask]
            y_valid = y[valid_mask]
            
            # Remove outliers
            X_valid, y_valid, n_outliers = remove_outliers(X_valid, y_valid)
            
            if n_outliers > 0:
                print_info(f"Removed {n_outliers} outliers")
            
            if len(y_valid) < 20:
                print_warning(f"Insufficient data: {len(y_valid)} subjects")
                continue
            
            print_info(f"Valid subjects: {len(y_valid)}")
            print()
            
            # Set random seed for reproducibility
            np.random.seed(RANDOM_SEED)
            
            # Optimize
            best_model, best_params, cv_score, opt_results = \
                optimize_comprehensive(X_valid, y_valid, measure, verbose=True, random_seed=RANDOM_SEED)
            
            # Evaluate with integrity checking
            eval_results = evaluate_model(best_model, X_valid, y_valid, verbose=True)
            
            # Create visualization
            create_scatter_plot(eval_results, measure, best_params, OUTPUT_DIR)
            
            # Save optimization results
            safe_name = measure.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
            method_name = best_params.get('strategy', 'Unknown').replace('+', '_')
            
            opt_results.to_csv(Path(OUTPUT_DIR) / f"optimization_results_{safe_name}.csv", index=False)
            
            # Save predictions
            # Ensure arrays are 1-dimensional (flatten if needed)
            y_actual_flat = np.asarray(eval_results['y_actual']).flatten()
            y_pred_flat = np.asarray(eval_results['y_pred']).flatten()
            
            predictions_df = pd.DataFrame({
                'Actual': y_actual_flat,
                'Predicted': y_pred_flat,
                'Residual': y_actual_flat - y_pred_flat
            })
            pred_filename = f"predictions_{safe_name}_{method_name}.csv"
            predictions_df.to_csv(Path(OUTPUT_DIR) / pred_filename, index=False)
            print_info(f"Saved predictions: {pred_filename}")
            
            # Verify predictions are reasonable
            if eval_results['y_pred'].std() < 0.01:
                print_warning(f"⚠️  Model for {measure} predicts nearly constant values!")
            elif abs(eval_results['r2']) > 10:
                print_warning(f"⚠️  Model for {measure} has extreme R² ({eval_results['r2']:.1f}) - likely overfitting!")
            
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
        
        # 4. Apply FDR correction and save summary
        if all_results:
            summary_df = pd.DataFrame(all_results)
            
            # Apply FDR correction across all measures
            if len(all_results) > 1:
                print()
                print_step("Applying FDR correction", f"Across {len(all_results)} ADOS measures")
                
                p_values = summary_df['Final_P_Value'].values
                corrected_p, rejected = apply_fdr_correction(p_values, alpha=0.05)
                
                summary_df['FDR_Corrected_P'] = corrected_p
                summary_df['FDR_Significant'] = rejected
                
                print_info(f"Significant before FDR: {(summary_df['Final_P_Value'] < 0.05).sum()}/{len(all_results)}", 0)
                print_info(f"Significant after FDR:  {rejected.sum()}/{len(all_results)}", 0)
            else:
                summary_df['FDR_Corrected_P'] = summary_df['Final_P_Value']
                summary_df['FDR_Significant'] = summary_df['Final_P_Value'] < 0.05
            
            summary_df.to_csv(Path(OUTPUT_DIR) / "optimization_summary.csv", index=False)
            
            print()
            print_completion("ABIDE ASD Brain-Behavior Analysis Complete!")
            print_info(f"Results saved to: {OUTPUT_DIR}")
            print()
            print("="*100)
            print("BEST PERFORMANCES (Sorted by Spearman ρ)")
            print("="*100)
            summary_sorted = summary_df.sort_values('Final_Spearman', ascending=False)
            
            # Format p-values for display
            summary_sorted['P_Display'] = summary_sorted['Final_P_Value'].apply(
                lambda p: '< 0.001' if p < 0.001 else f'{p:.4f}'
            )
            
            # Format FDR-corrected p-values if available
            if 'FDR_Corrected_P' in summary_sorted.columns:
                summary_sorted['P_FDR'] = summary_sorted['FDR_Corrected_P'].apply(
                    lambda p: '< 0.001' if p < 0.001 else f'{p:.4f}'
                )
                summary_sorted['Sig'] = summary_sorted['FDR_Significant'].apply(lambda x: '***' if x else '')
                
                print(summary_sorted[['Measure', 'N_Subjects', 'Final_Spearman', 'P_Display', 'P_FDR', 'Sig', 'Best_Strategy']].to_string(index=False))
            else:
                print(summary_sorted[['Measure', 'N_Subjects', 'Final_Spearman', 'P_Display', 'Best_Strategy', 'Best_Model']].to_string(index=False))
            print()
            
            best_row = summary_sorted.iloc[0]
            p_str = '< 0.001' if best_row['Final_P_Value'] < 0.001 else f"{best_row['Final_P_Value']:.4f}"
            print(f"\n  HIGHEST CORRELATION: ρ = {best_row['Final_Spearman']:.4f}, p {p_str} (N = {best_row['N_Subjects']})")
            print(f"  Measure: {best_row['Measure']}")
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

