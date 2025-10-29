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
        'data_type': 'abide_pklz',  # Special ABIDE format
        'data_path': '/oak/stanford/groups/menon/deriveddata/public/abide/restfmri/timeseries/group_level/brainnetome/normz/',
        'sites': ['NYU', 'SDSU', 'STANFORD', 'Stanford', 'TCD-1', 'UM', 'USM', 'Yale'],  # ABIDE sites
        'output_dir': '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/abide_asd_optimized',
        'beh_columns': ['ados_total', 'ados_comm', 'ados_social']
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
    
    if data_type == 'abide_pklz':
        return load_abide_behavioral(config)
    elif data_type == 'pklz' or data_type == 'pklz_file':
        return load_pklz_behavioral(config)
    elif data_type == 'c3sr':
        return load_c3sr_behavioral(config)
    else:
        raise ValueError(f"Unknown data type: {data_type}")


def load_abide_behavioral(config):
    """Load ABIDE behavioral data from pandas .pklz files (from enhanced script logic)."""
    print_step("Loading ABIDE behavioral data", f"From {Path(config['data_path']).name}")
    
    pklz_dir = Path(config['data_path'])
    sites = config['sites']
    
    # Find all .pklz files matching the specified sites AND ending with 246ROIs.pklz
    all_files = os.listdir(pklz_dir)
    filtered_files = []
    for file_name in all_files:
        if '246ROIs.pklz' in file_name and any(site in file_name for site in sites):
            filtered_files.append(pklz_dir / file_name)
    
    if not filtered_files:
        raise ValueError(f"No 246ROIs.pklz files found for sites: {sites}")
    
    print_info(f"Found {len(filtered_files)} 246ROIs.pklz files for {len(sites)} sites", 0)
    
    # Load and concatenate all dataframes
    appended_data = []
    for file_path in filtered_files:
        try:
            # ABIDE uses pd.read_pickle (NOT gzip)
            data = pd.read_pickle(file_path)
            # Remove NaN entries
            data = data[~pd.isna(data)]
            appended_data.append(data)
            print_info(f"  Loaded {len(data)} subjects from {file_path.name}", 0)
        except Exception as e:
            print_warning(f"  Could not load {file_path.name}: {e}")
    
    if not appended_data:
        raise ValueError("No data loaded from .pklz files")
    
    # Concatenate all data
    combined_df = pd.concat(appended_data, ignore_index=True)
    combined_df['label'] = combined_df['label'].astype(str)
    
    print_info(f"Total subjects loaded: {len(combined_df)}", 0)
    
    # Filter for ASD subjects only (label == '1')
    asd_df = combined_df[combined_df['label'] == '1'].copy()
    print_info(f"ASD subjects (label='1'): {len(asd_df)}", 0)
    
    # Filter for developmental age (age <= 21)
    if 'age' in asd_df.columns:
        asd_df['age'] = pd.to_numeric(asd_df['age'], errors='coerce')
        asd_df = asd_df[asd_df['age'] <= 21]
        print_info(f"ASD subjects age <= 21: {len(asd_df)}", 0)
    
    asd_df = asd_df.reset_index(drop=True)
    
    # Find subject ID column
    id_col = None
    for col in ['subjid', 'subject_id', 'id', 'ID']:
        if col in asd_df.columns:
            id_col = col
            break
    
    if id_col is None:
        raise ValueError(f"No subject ID column found. Columns: {list(asd_df.columns)}")
    
    if id_col != 'subject_id':
        asd_df = asd_df.rename(columns={id_col: 'subject_id'})
    
    asd_df['subject_id'] = asd_df['subject_id'].astype(str)
    
    # Check for ADOS columns
    ados_cols = []
    for target in config['beh_columns']:
        if target in asd_df.columns:
            ados_cols.append(target)
            print_info(f"  Found: {target}", 0)
        else:
            for col in asd_df.columns:
                if col.lower() == target.lower():
                    asd_df = asd_df.rename(columns={col: target})
                    ados_cols.append(target)
                    print_info(f"  Renamed: {col} -> {target}", 0)
                    break
    
    if not ados_cols:
        print_warning(f"No ADOS columns found. Available: {list(asd_df.columns)[:10]}")
        ados_cols = [col for col in asd_df.columns if 'ados' in col.lower()]
        if ados_cols:
            print_info(f"Using alternative ADOS columns: {ados_cols}", 0)
    
    print_info(f"Final subjects: {len(asd_df)}, Measures: {ados_cols}", 0)
    
    return asd_df[['subject_id'] + ados_cols], ados_cols


def load_pklz_behavioral(config):
    """Load behavioral data from ADHD200 PKLZ file (DataFrame format)."""
    print_step("Loading ADHD200 behavioral data", f"From {Path(config['data_path']).name}")
    
    data_path = Path(config['data_path'])
    
    # Load PKLZ file using pandas (it's a DataFrame, not a dict)
    data = pd.read_pickle(data_path)
    print_info(f"Total subjects loaded: {len(data)}", 0)
    
    # ADHD200-specific filtering (from enhanced script)
    # Filter out TR != 2.5
    if 'tr' in data.columns:
        data = data[data['tr'] != 2.5]
        print_info(f"After TR != 2.5 filter: {len(data)}", 0)
    
    # Remove 'pending' labels
    if 'label' in data.columns:
        data = data[data['label'] != 'pending']
        print_info(f"After removing 'pending': {len(data)}", 0)
    
    # Filter by mean_fd < 0.5
    if 'mean_fd' in data.columns:
        data = data[data['mean_fd'] < 0.5]
        print_info(f"After mean_fd < 0.5: {len(data)}", 0)
    
    # Convert subject_id to string
    if 'subject_id' in data.columns:
        data['subject_id'] = data['subject_id'].astype(str)
    
    # Remove duplicates
    data = data.drop_duplicates(subset='subject_id', keep='first')
    print_info(f"After deduplication: {len(data)}", 0)
    
    # Filter for TD vs ADHD based on cohort
    if 'label' in data.columns:
        data['label'] = pd.to_numeric(data['label'], errors='coerce')
        data = data[~data['label'].isna()]
        
        if 'adhd200_td' in config['output_dir']:
            filtered_data = data[data['label'] == 0]
            print_info(f"TD subjects (label=0): {len(filtered_data)}", 0)
        elif 'adhd200_adhd' in config['output_dir']:
            filtered_data = data[data['label'] == 1]
            print_info(f"ADHD subjects (label=1): {len(filtered_data)}", 0)
        else:
            filtered_data = data
    else:
        filtered_data = data
    
    # Check for behavioral columns (try both naming conventions)
    behavioral_cols = []
    for col in ['Hyper/Impulsive', 'Inattentive', 'Hyper_Impulsive']:
        if col in filtered_data.columns:
            behavioral_cols.append(col)
    
    if not behavioral_cols:
        print_warning(f"No behavioral columns found. Columns: {list(filtered_data.columns)[:10]}")
        return filtered_data[['subject_id']], []
    
    print_info(f"Behavioral columns found: {behavioral_cols}", 0)
    
    # Convert behavioral columns to numeric (handle nested arrays)
    for col in behavioral_cols:
        def extract_value(x):
            if isinstance(x, pd.Series):
                return float(x.iloc[0]) if len(x) > 0 else np.nan
            elif isinstance(x, np.ndarray):
                return float(x[0]) if len(x) > 0 else np.nan
            else:
                return float(x)
        
        filtered_data[col] = filtered_data[col].apply(extract_value)
        filtered_data[col] = pd.to_numeric(filtered_data[col], errors='coerce')
        filtered_data[col] = filtered_data[col].replace(-999.0, np.nan)  # Missing code
        
        non_null = filtered_data[col].notna().sum()
        print_info(f"  {col}: {non_null} non-null values", 0)
    
    # IMPORTANT: For TD cohort, filter for NYU site only (scale consistency)
    if 'adhd200_td' in config['output_dir'] and 'site' in filtered_data.columns:
        print_info(f"Filtering for NYU site only (scale consistency)", 0)
        filtered_data = filtered_data[filtered_data['site'] == 'NYU']
        print_info(f"NYU subjects: {len(filtered_data)}", 0)
    
    print_info(f"Final subjects: {len(filtered_data)}", 0)
    
    return filtered_data[['subject_id'] + behavioral_cols], behavioral_cols


def load_c3sr_behavioral(config):
    """Load C3SR behavioral data for CMI-HBN cohorts (from enhanced script logic)."""
    print_step("Loading CMI-HBN behavioral data", "")
    
    pklz_dir = Path(config['data_path'])
    
    # Load all run1 .pklz files (CMI-HBN format)
    pklz_files = [f for f in pklz_dir.glob('*.pklz') if 'run1' in f.name]
    
    if not pklz_files:
        raise ValueError(f"No run1 .pklz files found in {pklz_dir}")
    
    print_info(f"Found {len(pklz_files)} run1 .pklz files", 0)
    
    # Load and concatenate (pd.read_pickle, NOT gzip)
    data_list = []
    for pklz_file in pklz_files:
        data_new = pd.read_pickle(pklz_file)
        data_list.append(data_new)
        print_info(f"  Loaded {len(data_new)} from {pklz_file.name}", 0)
    
    data = pd.concat(data_list, ignore_index=True)
    print_info(f"Total subjects loaded: {len(data)}", 0)
    
    # Convert subject_id to string
    data['subject_id'] = data['subject_id'].astype(str)
    
    # Remove duplicates
    data = data.drop_duplicates(subset='subject_id', keep='first')
    print_info(f"After deduplication: {len(data)}", 0)
    
    # Filter for TD vs ADHD based on cohort
    if 'label' in data.columns:
        # Check label format
        if data['label'].dtype == 'object':
            # String labels
            if 'cmihbn_td' in config['output_dir']:
                filtered_data = data[data['label'].str.lower() == 'td']
                print_info(f"TD subjects (label='td'): {len(filtered_data)}", 0)
            elif 'cmihbn_adhd' in config['output_dir']:
                filtered_data = data[data['label'].str.lower() == 'adhd']
                print_info(f"ADHD subjects (label='adhd'): {len(filtered_data)}", 0)
            else:
                filtered_data = data
        else:
            # Numeric labels
            data = data[data['label'] != 'pending']
            data['label'] = pd.to_numeric(data['label'], errors='coerce')
            data = data[~data['label'].isna()]
            data = data[data['label'] != 99]
            
            if 'cmihbn_td' in config['output_dir']:
                filtered_data = data[data['label'] == 0]
                print_info(f"TD subjects (label=0): {len(filtered_data)}", 0)
            elif 'cmihbn_adhd' in config['output_dir']:
                filtered_data = data[data['label'] == 1]
                print_info(f"ADHD subjects (label=1): {len(filtered_data)}", 0)
            else:
                filtered_data = data
    else:
        filtered_data = data
    
    # Filter by mean_fd < 0.5
    if 'mean_fd' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['mean_fd'] < 0.5]
        print_info(f"After mean_fd < 0.5: {len(filtered_data)}", 0)
    
    # Load C3SR CSV
    print_step("Merging with C3SR", f"From {Path(config['beh_csv']).name}")
    
    c3sr = pd.read_csv(config['beh_csv'])
    
    # Find ID column in C3SR
    id_col = None
    for col in c3sr.columns:
        if 'id' in col.lower() or 'identifier' in col.lower():
            id_col = col
            break
    
    if id_col is None:
        raise ValueError(f"No ID column in C3SR. Columns: {list(c3sr.columns)}")
    
    # Process C3SR subject IDs (take first 12 characters)
    c3sr['subject_id'] = c3sr[id_col].apply(lambda x: str(x)[:12])
    
    # Find behavioral columns
    behavioral_cols = []
    for col in c3sr.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in ['c3sr_hy_t', 'c3sr_in_t']) or \
           (('_t' in col_lower or 't_score' in col_lower) and any(kw in col_lower for kw in ['hyperactiv', 'inattent', 'adhd'])):
            behavioral_cols.append(col)
    
    if not behavioral_cols:
        behavioral_cols = [col for col in c3sr.columns if '_T' in col or 'T_Score' in col or 'T-Score' in col]
    
    print_info(f"C3SR behavioral columns: {len(behavioral_cols)}", 0)
    
    # Merge PKLZ subjects with C3SR
    merged = filtered_data.merge(c3sr[['subject_id'] + behavioral_cols], on='subject_id', how='inner')
    
    print_info(f"Merged subjects (PKLZ + C3SR): {len(merged)}", 0)
    
    return merged[['subject_id'] + behavioral_cols], behavioral_cols


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
    
    # Add method info to filename
    method_name = best_params.get('strategy', 'Unknown').replace('+', '_')
    model_name = best_params.get('model', '')
    
    # Create descriptive filename
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
            
            # Convert to numeric, coercing errors to NaN
            y_series = pd.to_numeric(merged_df[measure], errors='coerce')
            y = y_series.values
            
            # Remove NaN
            valid_mask = ~np.isnan(y)
            X_valid = X[valid_mask]
            y_valid = y[valid_mask]
            
            n_invalid = np.sum(~valid_mask)
            if n_invalid > 0:
                print_info(f"Removed {n_invalid} subjects with missing/invalid data", 0)
            
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
            
            # Evaluate with integrity checking
            eval_results = evaluate_model(best_model, X_valid, y_valid, verbose=True)
            
            # Create visualization
            create_scatter_plot(eval_results, measure, best_params, config['output_dir'], config['dataset'])
            
            # Save optimization results
            safe_name = measure.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
            method_name = best_params.get('strategy', 'Unknown').replace('+', '_')
            
            opt_results.to_csv(Path(config['output_dir']) / f"optimization_results_{safe_name}.csv", index=False)
            
            # Save predictions for integrity verification (with method in filename)
            predictions_df = pd.DataFrame({
                'Actual': eval_results['y_actual'],
                'Predicted': eval_results['y_pred'],
                'Residual': eval_results['y_actual'] - eval_results['y_pred']
            })
            pred_filename = f"predictions_{safe_name}_{method_name}.csv"
            predictions_df.to_csv(Path(config['output_dir']) / pred_filename, index=False)
            print_info(f"Saved predictions to: {pred_filename}", 0)
            
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

