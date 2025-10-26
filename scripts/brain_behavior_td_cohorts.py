#!/usr/bin/env python3
"""
Brain-behavior analysis for TD cohorts (CMI-HBN TD and ADHD200 TD).

This script performs PCA-based brain-behavior correlation analysis using IG scores
for TD individuals from CMI-HBN and ADHD200 datasets.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import logging
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from os import listdir
from os.path import isfile, join

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add utils to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
utils_path = str(project_root / 'utils')
if utils_path not in sys.path:
    sys.path.append(utils_path)

from logging_utils import (print_section_header, print_step, print_success, 
                           print_warning, print_error, print_info, print_completion)


def benjamini_hochberg_correction(p_values, alpha=0.05):
    """Apply Benjamini-Hochberg FDR correction to p-values."""
    p_values = np.array(p_values)
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]
    
    # Calculate BH critical values
    bh_critical = (np.arange(1, n + 1) / n) * alpha
    
    # Find largest k such that p(k) <= (k/n) * alpha
    rejected = np.zeros(n, dtype=bool)
    corrected_p = np.ones(n)
    
    for i in range(n):
        if sorted_p_values[i] <= bh_critical[i]:
            rejected[sorted_indices[i]] = True
            corrected_p[sorted_indices[i]] = min(1.0, sorted_p_values[i] * n / (i + 1))
        else:
            break
    
    return corrected_p, rejected


def load_cmihbn_td_data(data_dir: str, behavioral_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load CMI-HBN TD data with behavioral scores.
    
    Args:
        data_dir: Directory containing imaging timeseries data
        behavioral_csv: Path to C3SR behavioral data CSV
        
    Returns:
        Tuple of (imaging_data, behavioral_data)
    """
    print_step(1, "LOADING CMI-HBN TD DATA", f"From {data_dir}")
    
    # Load imaging data (run1 files only)
    files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
    
    count = 0
    data = None
    for file in files:
        if 'run1' in file:
            count += 1
            file_data = np.load(join(data_dir, file), allow_pickle=True)
            data = pd.concat([data, file_data]) if data is not None else file_data
    
    print_info(f"Loaded {count} run1 files")
    
    # Filter data
    data['label'] = data['label'].astype(str).astype(int)
    df = data[(data['label'] != 99) & (data['mean_fd'] < 0.5)]
    
    # Filter for TD only (label == 0)
    df_td = df[df['label'] == 0].copy()
    print_info(f"TD subjects after filtering: {len(df_td)}")
    
    # Load behavioral data
    c3sr = pd.read_csv(behavioral_csv)
    c3sr['Identifiers'] = c3sr['Identifiers'].apply(lambda x: x[0:12]).astype('str')
    
    # Merge with behavioral data
    df_td['id_short'] = df_td['id'].astype(str).apply(lambda x: x[0:12])
    merged_data = df_td.merge(c3sr, left_on='id_short', right_on='Identifiers', how='inner')
    
    print_success(f"Merged data: {len(merged_data)} subjects with behavioral scores")
    
    return merged_data, c3sr


def load_adhd200_td_data(data_file: str) -> pd.DataFrame:
    """
    Load ADHD200 TD data with behavioral scores.
    
    Args:
        data_file: Path to ADHD200 .pklz file
        
    Returns:
        DataFrame with TD subjects and behavioral scores
    """
    print_step(1, "LOADING ADHD200 TD DATA", f"From {data_file}")
    
    # Load data
    hy_data = np.load(data_file, allow_pickle=True)
    
    # Convert to proper types
    hy_data['subject_id'] = hy_data['subject_id'].astype('str')
    hy_data['site'] = hy_data['site'].astype('str')
    
    # Remove duplicates
    hy_data = hy_data.drop_duplicates(subset='subject_id', keep='first')
    hy_data = hy_data.reset_index(drop=True)
    
    print_info(f"Total subjects after deduplication: {len(hy_data)}")
    
    # Filter for TD subjects (DX == 0)
    if 'DX' in hy_data.columns:
        df_td = hy_data[hy_data['DX'] == 0].copy()
    elif 'label' in hy_data.columns:
        df_td = hy_data[hy_data['label'] == 0].copy()
    else:
        print_warning("No DX or label column found, using all subjects")
        df_td = hy_data.copy()
    
    print_info(f"TD subjects: {len(df_td)}")
    
    # Handle NaNs in behavioral columns
    behavioral_cols = ['Hyper/Impulsive', 'Inattentive']
    for col in behavioral_cols:
        if col in df_td.columns:
            n_nan = df_td[col].isna().sum()
            if n_nan > 0:
                print_warning(f"Removing {n_nan} subjects with NaN in {col}")
                df_td = df_td[df_td[col].notna()]
    
    print_success(f"Final TD subjects with complete behavioral data: {len(df_td)}")
    
    return df_td


def load_ig_scores(ig_csv: str) -> pd.DataFrame:
    """
    Load IG scores from CSV file.
    
    Args:
        ig_csv: Path to IG CSV file (should have subject_id column + ROI columns)
        
    Returns:
        DataFrame with IG scores
    """
    print_step(2, "LOADING IG SCORES", f"From {ig_csv}")
    
    ig_data = pd.read_csv(ig_csv)
    print_info(f"Total IG entries: {len(ig_data)}")
    print_info(f"Columns: {list(ig_data.columns[:5])}... ({len(ig_data.columns)} total)")
    
    # Check for subject_id column
    if 'subject_id' not in ig_data.columns and 'id' not in ig_data.columns:
        print_error("No 'subject_id' or 'id' column found in IG CSV")
        return pd.DataFrame()
    
    # Standardize column name
    if 'id' in ig_data.columns and 'subject_id' not in ig_data.columns:
        ig_data = ig_data.rename(columns={'id': 'subject_id'})
    
    print_success(f"Loaded IG scores for {len(ig_data)} subjects")
    
    return ig_data


def merge_ig_with_behavioral(ig_data: pd.DataFrame, 
                            behavioral_data: pd.DataFrame,
                            behavioral_columns: List[str],
                            id_column_ig: str = 'subject_id',
                            id_column_behavior: str = 'subject_id') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge IG data with behavioral data based on subject IDs.
    
    Args:
        ig_data: DataFrame with IG scores (subject_id + ROI columns)
        behavioral_data: DataFrame with behavioral scores
        behavioral_columns: List of behavioral columns to keep
        id_column_ig: Name of ID column in IG data
        id_column_behavior: Name of ID column in behavioral data
        
    Returns:
        Tuple of (merged_ig_data, merged_behavioral_data) with matched subjects
    """
    print_step(3, "MERGING IG WITH BEHAVIORAL DATA", "Matching subject IDs")
    
    # Ensure ID columns are strings
    ig_data[id_column_ig] = ig_data[id_column_ig].astype(str)
    behavioral_data[id_column_behavior] = behavioral_data[id_column_behavior].astype(str)
    
    # Find common subjects
    ig_ids = set(ig_data[id_column_ig])
    behavior_ids = set(behavioral_data[id_column_behavior])
    common_ids = ig_ids.intersection(behavior_ids)
    
    print_info(f"IG subjects: {len(ig_ids)}")
    print_info(f"Behavioral subjects: {len(behavior_ids)}")
    print_info(f"Common subjects: {len(common_ids)}")
    
    if len(common_ids) == 0:
        print_error("No common subjects found between IG and behavioral data!")
        return pd.DataFrame(), pd.DataFrame()
    
    # Filter both datasets to common subjects
    ig_filtered = ig_data[ig_data[id_column_ig].isin(common_ids)].copy()
    behavior_filtered = behavioral_data[behavioral_data[id_column_behavior].isin(common_ids)].copy()
    
    # Sort by ID to ensure alignment
    ig_filtered = ig_filtered.sort_values(id_column_ig).reset_index(drop=True)
    behavior_filtered = behavior_filtered.sort_values(id_column_behavior).reset_index(drop=True)
    
    # Keep only behavioral columns of interest
    cols_to_keep = [id_column_behavior] + [col for col in behavioral_columns if col in behavior_filtered.columns]
    behavior_filtered = behavior_filtered[cols_to_keep]
    
    print_success(f"Merged data: {len(ig_filtered)} subjects with both IG and behavioral scores")
    
    return ig_filtered, behavior_filtered


def perform_pca_analysis(ig_data: pd.DataFrame, n_components: int = 10) -> Tuple[PCA, np.ndarray]:
    """
    Perform PCA on IG scores.
    
    Args:
        ig_data: DataFrame with IG scores (regions as columns)
        n_components: Number of PCA components
        
    Returns:
        Tuple of (PCA model, transformed data)
    """
    print_step(3, "PERFORMING PCA", f"Using {n_components} components")
    
    # Standardize data
    scaler = StandardScaler()
    ig_scaled = scaler.fit_transform(ig_data)
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_scores = pca.fit_transform(ig_scaled)
    
    # Print variance explained
    variance_explained = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_explained)
    
    print_info(f"Variance explained by first 3 PCs: {cumulative_variance[2]:.2%}")
    print_info(f"Variance explained by all {n_components} PCs: {cumulative_variance[-1]:.2%}")
    
    return pca, pca_scores


def correlate_with_behavior(pca_scores: np.ndarray, 
                            behavioral_scores: pd.Series,
                            behavior_name: str) -> Dict:
    """
    Correlate PCA scores with behavioral scores.
    
    Args:
        pca_scores: PCA component scores (n_subjects x n_components)
        behavioral_scores: Behavioral scores for each subject
        behavior_name: Name of behavioral measure
        
    Returns:
        Dictionary with correlation results
    """
    print_step(4, f"CORRELATING WITH {behavior_name.upper()}", f"N = {len(behavioral_scores)}")
    
    n_components = pca_scores.shape[1]
    correlations = []
    p_values = []
    
    for i in range(n_components):
        r, p = pearsonr(pca_scores[:, i], behavioral_scores)
        correlations.append(r)
        p_values.append(p)
    
    # Apply FDR correction
    corrected_p, rejected = benjamini_hochberg_correction(p_values)
    
    # Find significant components
    sig_components = np.where(rejected)[0]
    
    results = {
        'behavior': behavior_name,
        'n_subjects': len(behavioral_scores),
        'correlations': correlations,
        'p_values': p_values,
        'corrected_p': corrected_p,
        'significant_components': sig_components,
        'n_significant': len(sig_components)
    }
    
    # Print results
    if len(sig_components) > 0:
        print_success(f"Found {len(sig_components)} significant components (FDR < 0.05):")
        for idx in sig_components:
            print_info(f"  PC{idx+1}: r = {correlations[idx]:.3f}, p = {p_values[idx]:.3e}, p_corr = {corrected_p[idx]:.3e}")
    else:
        print_warning("No significant components found after FDR correction")
    
    return results


def save_results(results: Dict, output_path: str):
    """Save analysis results to CSV."""
    df = pd.DataFrame({
        'Component': [f'PC{i+1}' for i in range(len(results['correlations']))],
        'Correlation_r': results['correlations'],
        'P_value': results['p_values'],
        'P_value_corrected': results['corrected_p'],
        'Significant': [i in results['significant_components'] for i in range(len(results['correlations']))]
    })
    
    df.to_csv(output_path, index=False)
    print_success(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Brain-behavior analysis for TD cohorts")
    parser.add_argument('--dataset', type=str, required=True, choices=['cmihbn_td', 'adhd200_td'],
                       help='Dataset to analyze')
    parser.add_argument('--data_dir', type=str,
                       help='Directory containing imaging timeseries data (for CMI-HBN)')
    parser.add_argument('--data_file', type=str,
                       help='Path to data file (for ADHD200 .pklz)')
    parser.add_argument('--behavioral_csv', type=str,
                       help='Path to behavioral data CSV (C3SR for CMI-HBN)')
    parser.add_argument('--ig_csv', type=str, required=True,
                       help='Path to IG scores CSV')
    parser.add_argument('--behavioral_columns', type=str, nargs='+', 
                       default=['HY', 'IN'],
                       help='Behavioral columns to analyze')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--n_components', type=int, default=10,
                       help='Number of PCA components (default: 10)')
    
    args = parser.parse_args()
    
    print_section_header(f"BRAIN-BEHAVIOR ANALYSIS - {args.dataset.upper()}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load IG scores first
    ig_data = load_ig_scores(args.ig_csv)
    if ig_data.empty:
        return
    
    # Load behavioral data based on dataset
    if args.dataset == 'cmihbn_td':
        if not args.data_dir or not args.behavioral_csv:
            print_error("--data_dir and --behavioral_csv required for cmihbn_td")
            return
        behavioral_data, _ = load_cmihbn_td_data(args.data_dir, args.behavioral_csv)
        # Map column names for CMI-HBN
        column_mapping = {'HY': 'HY', 'IN': 'IN'}
        id_column_behavior = 'id'
        
    elif args.dataset == 'adhd200_td':
        if not args.data_file:
            print_error("--data_file required for adhd200_td")
            return
        behavioral_data = load_adhd200_td_data(args.data_file)
        # Map column names for ADHD200
        column_mapping = {'HY': 'Hyper/Impulsive', 'IN': 'Inattentive'}
        id_column_behavior = 'subject_id'
        
    else:
        print_error(f"Dataset {args.dataset} not implemented")
        return
    
    # Merge IG with behavioral data
    print()
    ig_merged, behavior_merged = merge_ig_with_behavioral(
        ig_data, 
        behavioral_data, 
        list(column_mapping.values()),
        id_column_ig='subject_id',
        id_column_behavior=id_column_behavior
    )
    
    if ig_merged.empty:
        print_error("No subjects with both IG and behavioral data")
        return
    
    # Prepare IG matrix (subjects x regions) - exclude subject_id column
    roi_columns = [col for col in ig_merged.columns if col != 'subject_id']
    ig_matrix = ig_merged[roi_columns].values
    
    print()
    print_info(f"IG matrix shape: {ig_matrix.shape} (subjects x ROIs)")
    
    # Perform PCA
    print()
    pca, pca_scores = perform_pca_analysis(ig_matrix, n_components=args.n_components)
    
    # Correlate with each behavioral measure
    print()
    all_results = []
    for behavior_col in args.behavioral_columns:
        # Map to actual column name in data
        actual_col = column_mapping.get(behavior_col, behavior_col)
        
        if actual_col in behavior_merged.columns:
            behavioral_scores = behavior_merged[actual_col].values
            
            # Remove any remaining NaNs
            valid_mask = ~np.isnan(behavioral_scores)
            if not valid_mask.all():
                n_removed = (~valid_mask).sum()
                print_warning(f"Removing {n_removed} subjects with NaN in {actual_col}")
                behavioral_scores = behavioral_scores[valid_mask]
                pca_scores_filtered = pca_scores[valid_mask]
            else:
                pca_scores_filtered = pca_scores
            
            results = correlate_with_behavior(pca_scores_filtered, behavioral_scores, behavior_col)
            all_results.append(results)
            
            # Save results
            output_path = output_dir / f"{args.dataset}_{behavior_col}_pca_correlations.csv"
            save_results(results, str(output_path))
            print()
        else:
            print_warning(f"Behavioral column '{actual_col}' (mapped from '{behavior_col}') not found in data")
            print_info(f"Available columns: {list(behavior_merged.columns)}")
    
    print_completion(f"{args.dataset.upper()} Brain-Behavior Analysis", 
                    [str(output_dir / f"{args.dataset}_{behavior_col}_pca_correlations.csv") 
                     for col in args.behavioral_columns if column_mapping.get(col, col) in behavior_merged.columns])


if __name__ == "__main__":
    main()

