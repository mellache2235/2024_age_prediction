#!/usr/bin/env python3
"""
Universal Optimized Brain-Behavior Analysis for All Cohorts

Comprehensive optimization to maximize brain-behavior correlations.
Supports: ABIDE ASD, ADHD200 (TD & ADHD), CMI-HBN (TD & ADHD), NKI, Stanford ASD

Usage:
    python run_all_cohorts_brain_behavior_optimized.py --cohort abide_asd
    python run_all_cohorts_brain_behavior_optimized.py --cohort adhd200_td --max-measures 2
    python run_all_cohorts_brain_behavior_optimized.py --cohort cmihbn_adhd --n-jobs 4
    python run_all_cohorts_brain_behavior_optimized.py --all  # Run all cohorts

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
# COHORT CONFIGURATIONS
# ============================================================================

COHORTS = {
    'abide_asd': {
        'name': 'ABIDE ASD',
        'dataset': 'abide_asd',
        'ig_csv': '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/integrated_gradients/abide_asd_features_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS.csv',
        'data_type': 'pklz',
        'data_path': '/oak/stanford/groups/menon/deriveddata/public/abide/restfmri/timeseries/group_level/brainnetome/normz/',
        'output_dir': '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/abide_asd_optimized',
        'beh_columns': ['ados_total', 'ados_comm', 'ados_social']  # From PKLZ
    },
    'adhd200_td': {
        'name': 'ADHD200 TD',
        'dataset': 'adhd200_td',
        'ig_csv': '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/integrated_gradients/adhd200_td_features_all_sites_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS_single_model_predictions.csv',
        'data_type': 'pklz_file',
        'data_path': '/oak/stanford/groups/menon/deriveddata/public/adhd200/restfmri/timeseries/group_level/brainnetome/normz/adhd200_run-rest_brainnetome_mean_regMov-6param_wmcsf_dt1_bpf008-09_normz_246ROIs.pklz',
        'output_dir': '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/adhd200_td_optimized',
        'beh_columns': ['Hyper_Impulsive', 'Inattentive']  # From PKLZ
    },
    'adhd200_adhd': {
        'name': 'ADHD200 ADHD',
        'dataset': 'adhd200_adhd',
        'ig_csv': '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/integrated_gradients/adhd200_adhd_features_all_sites_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS_single_model_predictions.csv',
        'data_type': 'pklz_file',
        'data_path': '/oak/stanford/groups/menon/deriveddata/public/adhd200/restfmri/timeseries/group_level/brainnetome/normz/adhd200_run-rest_brainnetome_mean_regMov-6param_wmcsf_dt1_bpf008-09_normz_246ROIs.pklz',
        'output_dir': '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/adhd200_adhd_optimized',
        'beh_columns': ['Hyper_Impulsive', 'Inattentive']  # From PKLZ
    },
    'cmihbn_td': {
        'name': 'CMI-HBN TD',
        'dataset': 'cmihbn_td',
        'ig_csv': '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/integrated_gradients/cmihbn_td_features_all_sites_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS_single_model_predictions.csv',
        'data_type': 'c3sr',
        'data_path': '/oak/stanford/groups/menon/deriveddata/public/cmihbn/restfmri/timeseries/group_level/brainnetome/normz',
        'beh_csv': '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/prepare_data/adhd/C3SR.csv',
        'output_dir': '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/cmihbn_td_optimized',
        'beh_columns': None  # Will be auto-detected from C3SR
    },
    'cmihbn_adhd': {
        'name': 'CMI-HBN ADHD',
        'dataset': 'cmihbn_adhd',
        'ig_csv': '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/integrated_gradients/cmihbn_adhd_no_cutoffs_features_all_sites_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS_single_model_predictions.csv',
        'data_type': 'c3sr',
        'data_path': '/oak/stanford/groups/menon/deriveddata/public/cmihbn/restfmri/timeseries/group_level/brainnetome/normz',
        'beh_csv': '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/prepare_data/adhd/C3SR.csv',
        'output_dir': '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/cmihbn_adhd_optimized',
        'beh_columns': None  # Will be auto-detected from C3SR
    }
}

# ============================================================================
# DATA LOADING FUNCTIONS (FROM ENHANCED SCRIPTS)
# ============================================================================

def load_ig_scores(ig_csv):
    """Universal IG score loader."""
    print_step("Loading IG scores", f"From {Path(ig_csv).name}")
    
    df = pd.read_csv(ig_csv)
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    # Find subject ID column
    id_col = None
    for col in ['subject_id', 'subjid', 'id', 'ID', 'Subject_ID', 'record_id']:
        if col in df.columns:
            id_col = col
            break
    
    if id_col is None:
        raise ValueError("No subject ID column found in IG CSV")
    
    if id_col != 'subject_id':
        df = df.rename(columns={id_col: 'subject_id'})
    
    df['subject_id'] = df['subject_id'].astype(str)
    roi_cols = [col for col in df.columns if col != 'subject_id']
    
    print_info(f"IG subjects: {len(df)}", 0)
    print_info(f"IG features (ROIs): {len(roi_cols)}", 0)
    
    return df, roi_cols


def load_behavioral_data(config):
    """Load behavioral data based on cohort type."""
    data_type = config['data_type']
    
    if data_type == 'pklz' or data_type == 'pklz_file':
        return load_pklz_behavioral(config)
    elif data_type == 'c3sr':
        return load_c3sr_behavioral(config)
    else:
        raise ValueError(f"Unknown data type: {data_type}")


def load_pklz_behavioral(config):
    """Load behavioral data from PKLZ files."""
    import pickle
    import gzip
    
    print_step("Loading PKLZ behavioral data", "")
    
    data_path = config['data_path']
    
    # Handle both directory and single file
    if Path(data_path).is_file():
        pklz_files = [Path(data_path)]
    else:
        pklz_files = list(Path(data_path).glob("*.pklz"))
    
    if not pklz_files:
        raise ValueError(f"No PKLZ files found in {data_path}")
    
    all_data = {}
    for pklz_file in pklz_files:
        # Try gzipped first, then fall back to plain pickle
        try:
            with gzip.open(pklz_file, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
                all_data.update(data)
        except gzip.BadGzipFile:
            # File is not gzipped, try plain pickle
            with open(pklz_file, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
                all_data.update(data)
    
    # Extract subject IDs and behavioral measures
    subject_ids = []
    behavioral_data = {col: [] for col in config['beh_columns']}
    
    for subj_id, subj_data in all_data.items():
        subject_ids.append(str(subj_id))
        for col in config['beh_columns']:
            if col in subj_data:
                behavioral_data[col].append(subj_data[col])
            else:
                behavioral_data[col].append(np.nan)
    
    beh_df = pd.DataFrame({'subject_id': subject_ids, **behavioral_data})
    
    print_info(f"Behavioral subjects: {len(beh_df)}", 0)
    print_info(f"Behavioral measures: {config['beh_columns']}", 0)
    
    return beh_df, config['beh_columns']


def load_c3sr_behavioral(config):
    """Load C3SR behavioral data for CMI-HBN cohorts."""
    import pickle
    import gzip
    
    print_step("Loading C3SR behavioral data", "")
    
    # Load PKLZ for subject IDs
    pklz_files = list(Path(config['data_path']).glob("*.pklz"))
    all_data = {}
    for pklz_file in pklz_files:
        # Try gzipped first, then fall back to plain pickle
        try:
            with gzip.open(pklz_file, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
                all_data.update(data)
        except gzip.BadGzipFile:
            # File is not gzipped, try plain pickle
            with open(pklz_file, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
                all_data.update(data)
    
    # Load C3SR CSV
    c3sr_df = pd.read_csv(config['beh_csv'])
    
    # Standardize C3SR column names
    if 'Identifiers' in c3sr_df.columns:
        c3sr_df = c3sr_df.rename(columns={'Identifiers': 'subject_id'})
    
    c3sr_df['subject_id'] = c3sr_df['subject_id'].astype(str)
    
    # Get behavioral columns (T-scores)
    beh_cols = [col for col in c3sr_df.columns if '_T' in col or 'T-Score' in col]
    
    # Filter to subjects in PKLZ
    pklz_subjects = [str(sid) for sid in all_data.keys()]
    c3sr_df = c3sr_df[c3sr_df['subject_id'].isin(pklz_subjects)]
    
    print_info(f"Behavioral subjects: {len(c3sr_df)}", 0)
    print_info(f"Behavioral measures: {len(beh_cols)}", 0)
    
    return c3sr_df, beh_cols


def merge_data(ig_df, beh_df):
    """Merge IG and behavioral data."""
    print_step("Merging data", "Matching subject IDs")
    
    merged = pd.merge(ig_df, beh_df, on='subject_id', how='inner')
    
    print_success(f"Merged: {len(merged)} subjects with both IG and behavioral data")
    
    if len(merged) < 10:
        raise ValueError(f"Insufficient overlap: only {len(merged)} common subjects")
    
    return merged


def create_scatter_plot(results, measure_name, best_params, output_dir, dataset_name):
    """Create scatter plot using centralized styling."""
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
    
    title = get_dataset_title(dataset_name)
    safe_name = measure_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
    save_path = Path(output_dir) / f'scatter_{safe_name}_optimized'
    
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


def analyze_cohort(cohort_key, max_measures=None):
    """Analyze a single cohort."""
    config = COHORTS[cohort_key]
    
    print_section_header(f"OPTIMIZED BRAIN-BEHAVIOR ANALYSIS - {config['name'].upper()}")
    print()
    
    # Create output directory
    Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
    
    try:
        # Load data
        ig_df, roi_cols = load_ig_scores(config['ig_csv'])
        beh_df, beh_cols = load_behavioral_data(config)
        
        # Merge
        merged_df = merge_data(ig_df, beh_df)
        
        # Extract IG matrix
        X = merged_df[roi_cols].values
        
        print_info(f"IG matrix shape: {X.shape} (subjects x ROIs)", 0)
        print_info(f"Behavioral measures to analyze: {len(beh_cols)}", 0)
        
        # Limit measures if specified
        if max_measures:
            beh_cols = beh_cols[:max_measures]
            print_warning(f"Limited to {max_measures} measures for testing")
        
        print()
        
        # Analyze each behavioral measure
        all_results = []
        
        for measure in beh_cols:
            print()
            print_section_header(f"ANALYZING: {measure}")
            
            if measure not in merged_df.columns:
                print_warning(f"Column '{measure}' not found - skipping")
                continue
            
            y = merged_df[measure].values
            
            # Remove NaN
            valid_mask = ~np.isnan(y)
            X_valid = X[valid_mask]
            y_valid = y[valid_mask]
            
            # Remove outliers
            X_valid, y_valid, n_outliers = remove_outliers(X_valid, y_valid)
            
            if n_outliers > 0:
                print_info(f"Removed {n_outliers} outliers", 0)
            
            if len(y_valid) < 20:
                print_warning(f"Insufficient data: {len(y_valid)} subjects")
                continue
            
            print_info(f"Valid subjects: {len(y_valid)}", 0)
            
            # Optimize
            best_model, best_params, cv_score, opt_results = \
                optimize_comprehensive(X_valid, y_valid, measure, verbose=True)
            
            # Evaluate
            eval_results = evaluate_model(best_model, X_valid, y_valid)
            
            # Create visualization
            create_scatter_plot(eval_results, measure, best_params, config['output_dir'], config['dataset'])
            
            # Save optimization results
            safe_name = measure.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
            opt_results.to_csv(Path(config['output_dir']) / f"optimization_results_{safe_name}.csv", index=False)
            
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
            summary_df.to_csv(Path(config['output_dir']) / "optimization_summary.csv", index=False)
            
            print()
            print_completion(f"{config['name']} Brain-Behavior Analysis Complete!")
            print_info(f"Results saved to: {config['output_dir']}", 0)
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
        description="Universal optimized brain-behavior analysis for all cohorts"
    )
    parser.add_argument(
        '--cohort', '-c',
        choices=list(COHORTS.keys()),
        help="Cohort to analyze"
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help="Run all cohorts"
    )
    parser.add_argument(
        '--max-measures', '-m',
        type=int,
        default=None,
        help="Maximum number of behavioral measures to analyze (for testing)"
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
        success = analyze_cohort(cohort_key, max_measures=args.max_measures)
        results[cohort_key] = success
        print("\n" + "="*100 + "\n")
    
    # Summary
    print_section_header("OVERALL SUMMARY")
    for cohort_key, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {COHORTS[cohort_key]['name']:.<50} {status}")
    print()


if __name__ == "__main__":
    main()

