#!/usr/bin/env python3
"""
Optimized Brain-Behavior Analysis for CMI-HBN ADHD Cohort

Uses EXACT same data loading logic as enhanced script (with diagnosis file), 
adds comprehensive optimization.

Usage:
    python run_cmihbn_adhd_brain_behavior_optimized.py
    python run_cmihbn_adhd_brain_behavior_optimized.py --max-measures 2
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

# Add to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from logging_utils import (print_section_header, print_step, print_success, 
                           print_warning, print_error, print_info, print_completion)
from plot_styles import create_standardized_scatter, get_dataset_title, setup_arial_font, DPI, FIGURE_FACECOLOR
from optimized_brain_behavior_core import optimize_comprehensive, evaluate_model, remove_outliers, apply_fdr_correction

setup_arial_font()

# ============================================================================
# CONFIGURATION (FROM ENHANCED SCRIPT)
# ============================================================================

DATASET = "cmihbn_adhd"
PKLZ_DIR = "/oak/stanford/groups/menon/deriveddata/public/cmihbn/restfmri/timeseries/group_level/brainnetome/normz"
IG_CSV = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/integrated_gradients/cmihbn_adhd_no_cutoffs_features_all_sites_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS_single_model_predictions.csv"
DIAGNOSIS_CSV = "/oak/stanford/groups/menon/projects/mellache/2024_FM_ADHD/scripts/prepare_data/cmihbn/Diagnosis_ClinicianConsensus.csv"
C3SR_FILE = "/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/prepare_data/adhd/C3SR.csv"
OUTPUT_DIR = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/cmihbn_adhd_optimized"

RANDOM_SEED = 42

# ============================================================================
# DATA LOADING (EXACT SAME AS ENHANCED SCRIPT)
# ============================================================================

def load_cmihbn_adhd_subjects(diagnosis_csv):
    """Load CMI-HBN ADHD subjects from diagnosis CSV (from enhanced script)."""
    print_step("Loading ADHD diagnosis data", f"From {Path(diagnosis_csv).name}")
    
    if not Path(diagnosis_csv).exists():
        raise ValueError(f"Diagnosis CSV not found: {diagnosis_csv}")
    
    diagnosis = pd.read_csv(diagnosis_csv)
    
    # Rename EID to subject_id
    if 'EID' in diagnosis.columns:
        diagnosis = diagnosis.rename(columns={'EID': 'subject_id'})
        print_info(f"Renamed 'EID' to 'subject_id'")
    elif 'subject_id' not in diagnosis.columns:
        raise ValueError(f"No 'EID' or 'subject_id' column found")
    
    diagnosis['subject_id'] = diagnosis['subject_id'].astype(str)
    
    if 'DX_01_Sub' in diagnosis.columns:
        diagnosis['DX_01_Sub'] = diagnosis['DX_01_Sub'].astype('string')
    else:
        raise ValueError(f"'DX_01_Sub' column not found")
    
    print_info(f"Total subjects in diagnosis file: {len(diagnosis)}")
    
    # Filter for ADHD subjects
    df_adhd = diagnosis.loc[diagnosis['DX_01_Sub'] == 'Attention-Deficit/Hyperactivity Disorder', :]
    adhd_ids = df_adhd['subject_id'].unique()
    
    print_info(f"ADHD subjects: {len(adhd_ids)}")
    
    return adhd_ids


def load_cmihbn_data(pklz_dir, adhd_ids, c3sr_file):
    """Load CMI-HBN imaging and behavioral data (from enhanced script)."""
    print_step("Loading CMI-HBN imaging data", f"From {Path(pklz_dir).name}")
    
    # Load run1 .pklz files
    pklz_files = [f for f in Path(pklz_dir).glob('*.pklz') if 'run1' in f.name]
    
    if not pklz_files:
        raise ValueError(f"No run1 .pklz files found in {pklz_dir}")
    
    print_info(f"Found {len(pklz_files)} run1 .pklz files")
    
    data_list = []
    for pklz_file in pklz_files:
        data_new = pd.read_pickle(pklz_file)
        data_list.append(data_new)
        print_info(f"  Loaded {len(data_new)} from {pklz_file.name}")
    
    data = pd.concat(data_list, ignore_index=True)
    print_info(f"Total subjects: {len(data)}")
    
    data['subject_id'] = data['subject_id'].astype(str)
    data = data.drop_duplicates(subset='subject_id', keep='first')
    print_info(f"After deduplication: {len(data)}")
    
    # Filter for ADHD subjects from diagnosis file
    print_info(f"Filtering for ADHD subjects from diagnosis...")
    adhd_data = data[data['subject_id'].isin(adhd_ids)]
    print_info(f"ADHD subjects in imaging data: {len(adhd_data)}")
    
    # Filter by mean_fd < 0.5
    adhd_data = adhd_data[adhd_data['mean_fd'] < 0.5]
    print_info(f"After mean_fd < 0.5: {len(adhd_data)}")
    
    # Load C3SR
    print_step("Loading C3SR", f"From {Path(c3sr_file).name}")
    c3sr = pd.read_csv(c3sr_file)
    
    id_col = None
    for col in c3sr.columns:
        if 'id' in col.lower() or 'identifier' in col.lower():
            id_col = col
            break
    
    if id_col is None:
        raise ValueError("No ID column in C3SR")
    
    c3sr['subject_id'] = c3sr[id_col].apply(lambda x: str(x)[:12])
    
    # Find behavioral columns
    behavioral_cols = []
    for col in c3sr.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in ['c3sr_hy_t', 'c3sr_in_t']) or \
           (('_t' in col_lower or 't_score' in col_lower) and any(kw in col_lower for kw in ['hyperactiv', 'inattent', 'adhd'])):
            behavioral_cols.append(col)
    
    if not behavioral_cols:
        behavioral_cols = [col for col in c3sr.columns if '_T' in col or 'T_Score' in col]
    
    print_info(f"C3SR behavioral columns: {len(behavioral_cols)}")
    
    # Merge
    merged = adhd_data.merge(c3sr[['subject_id'] + behavioral_cols], on='subject_id', how='inner')
    print_info(f"Merged subjects (ADHD imaging + C3SR): {len(merged)}")
    
    return merged, behavioral_cols


def load_ig_scores(ig_csv):
    """Load IG scores."""
    print_step("Loading IG scores", f"From {Path(ig_csv).name}")
    
    df = pd.read_csv(ig_csv)
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    id_col = None
    for col in ['subject_id', 'subjid', 'id', 'ID']:
        if col in df.columns:
            id_col = col
            break
    
    if id_col is None:
        raise ValueError("No subject ID column found")
    
    if id_col != 'subject_id':
        df = df.rename(columns={id_col: 'subject_id'})
    
    df['subject_id'] = df['subject_id'].astype(str)
    roi_cols = [col for col in df.columns if col != 'subject_id']
    
    print_info(f"IG subjects: {len(df)}")
    print_info(f"IG features: {len(roi_cols)}")
    
    return df, roi_cols


def merge_data(ig_df, beh_df):
    """Merge IG and behavioral data."""
    print_step("Merging data", "Matching subject IDs")
    
    merged = pd.merge(ig_df, beh_df, on='subject_id', how='inner')
    print_success(f"Merged: {len(merged)} subjects")
    
    if len(merged) < 10:
        raise ValueError(f"Insufficient overlap: {len(merged)} subjects")
    
    return merged


def create_scatter_plot(results, measure_name, best_params, output_dir):
    """Create scatter plot."""
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
    parser = argparse.ArgumentParser(description="Optimized brain-behavior for CMI-HBN ADHD")
    parser.add_argument('--max-measures', '-m', type=int, default=None)
    args = parser.parse_args()
    
    print_section_header("OPTIMIZED BRAIN-BEHAVIOR ANALYSIS - CMI-HBN ADHD")
    print()
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Load ADHD subjects from diagnosis file
        adhd_ids = load_cmihbn_adhd_subjects(DIAGNOSIS_CSV)
        print()
        
        # 2. Load imaging and behavioral data
        behavioral_df, c3sr_cols = load_cmihbn_data(PKLZ_DIR, adhd_ids, C3SR_FILE)
        print()
        
        # 3. Load IG scores
        ig_df, roi_cols = load_ig_scores(IG_CSV)
        print()
        
        # 4. Merge
        merged_df = merge_data(ig_df, behavioral_df)
        
        X = merged_df[roi_cols].values
        
        print_info(f"IG matrix shape: {X.shape}")
        print_info(f"C3SR measures: {c3sr_cols}")
        
        if args.max_measures:
            c3sr_cols = c3sr_cols[:args.max_measures]
            print_warning(f"Limited to {args.max_measures} measures")
        
        print()
        
        # 5. Analyze each measure
        all_results = []
        
        for measure in c3sr_cols:
            print_section_header(f"ANALYZING: {measure}")
            
            y = pd.to_numeric(merged_df[measure], errors='coerce').values
            
            valid_mask = ~np.isnan(y)
            X_valid = X[valid_mask]
            y_valid = y[valid_mask]
            
            X_valid, y_valid, n_outliers = remove_outliers(X_valid, y_valid)
            
            if n_outliers > 0:
                print_info(f"Removed {n_outliers} outliers")
            
            if len(y_valid) < 20:
                print_warning(f"Insufficient data: {len(y_valid)} subjects")
                continue
            
            print_info(f"Valid subjects: {len(y_valid)}")
            print()
            
            np.random.seed(RANDOM_SEED)
            
            best_model, best_params, cv_score, opt_results = \
                optimize_comprehensive(X_valid, y_valid, measure, verbose=True, random_seed=RANDOM_SEED)
            
            eval_results = evaluate_model(best_model, X_valid, y_valid, verbose=True)
            
            create_scatter_plot(eval_results, measure, best_params, OUTPUT_DIR)
            
            safe_name = measure.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
            method_name = best_params.get('strategy', 'Unknown').replace('+', '_')
            
            opt_results.to_csv(Path(OUTPUT_DIR) / f"optimization_results_{safe_name}.csv", index=False)
            
            y_actual_flat = np.asarray(eval_results['y_actual']).flatten()
            y_pred_flat = np.asarray(eval_results['y_pred']).flatten()
            
            predictions_df = pd.DataFrame({
                'Actual': y_actual_flat,
                'Predicted': y_pred_flat,
                'Residual': y_actual_flat - y_pred_flat
            })
            predictions_df.to_csv(Path(OUTPUT_DIR) / f"predictions_{safe_name}_{method_name}.csv", index=False)
            
            all_results.append({
                'Measure': measure,
                'N_Subjects': int(len(y_valid)),
                'N_Outliers_Removed': int(n_outliers),
                'Best_Strategy': best_params.get('strategy', 'N/A'),
                'Best_Model': best_params.get('model', 'N/A'),
                'Best_N_Components': best_params.get('n_components', None),
                'Best_Alpha': best_params.get('alpha', None),
                'Feature_Selection': best_params.get('feature_selection', None),
                'N_Features': best_params.get('n_features', None),
                'CV_Spearman': float(cv_score) if cv_score is not None else None,
                'Final_Spearman': float(np.asarray(eval_results['rho']).item()),
                'Final_P_Value': float(np.asarray(eval_results['p_value']).item()),
                'Final_R2': float(np.asarray(eval_results['r2']).item()),
                'Final_MAE': float(np.asarray(eval_results['mae']).item())
            })
            
            print()
        
        # 6. Apply FDR and save
        if all_results:
            summary_df = pd.DataFrame(all_results)
            
            if len(all_results) > 1:
                print()
                print_step("Applying FDR correction", f"Across {len(all_results)} measures")
                
                p_values = summary_df['Final_P_Value'].values
                corrected_p, rejected = apply_fdr_correction(p_values, alpha=0.05)
                
                summary_df['FDR_Corrected_P'] = corrected_p
                summary_df['FDR_Significant'] = rejected
                
                print_info(f"Significant before FDR: {(summary_df['Final_P_Value'] < 0.05).sum()}/{len(all_results)}")
                print_info(f"Significant after FDR: {rejected.sum()}/{len(all_results)}")
                
                if rejected.sum() > 0:
                    print()
                    print("  ✅ Measures surviving FDR correction:")
                    for _, row in summary_df[summary_df['FDR_Significant']].iterrows():
                        p_str = '< 0.001' if row['Final_P_Value'] < 0.001 else f"{row['Final_P_Value']:.4f}"
                        p_fdr = '< 0.001' if row['FDR_Corrected_P'] < 0.001 else f"{row['FDR_Corrected_P']:.4f}"
                        print(f"    {row['Measure']:.<50} ρ={row['Final_Spearman']:.3f}, p={p_str}, p_FDR={p_fdr}")
            else:
                summary_df['FDR_Corrected_P'] = summary_df['Final_P_Value']
                summary_df['FDR_Significant'] = summary_df['Final_P_Value'] < 0.05
            
            summary_df.to_csv(Path(OUTPUT_DIR) / "optimization_summary.csv", index=False)
            
            print()
            print_completion("CMI-HBN ADHD Analysis Complete!")
            print_info(f"Results: {OUTPUT_DIR}")
            print()
            
            summary_sorted = summary_df.sort_values('Final_Spearman', ascending=False)
            print("="*100)
            print("RESULTS")
            print("="*100)
            print(summary_sorted[['Measure', 'N_Subjects', 'Final_Spearman', 'FDR_Significant']].to_string(index=False))
            print()
        
    except Exception as e:
        print()
        print_error(f"Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

