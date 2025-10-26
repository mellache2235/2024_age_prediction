#!/usr/bin/env python3
"""
Simplified brain-behavior analysis for TD cohorts using .pklz files and IG CSV.

This script:
1. Loads IG CSV with subject IDs
2. Loads behavioral data from .pklz files
3. Matches subjects between IG and behavioral data
4. Performs PCA on IG scores
5. Correlates PCA components with behavioral measures (HY, IN)
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
    
    bh_critical = (np.arange(1, n + 1) / n) * alpha
    
    rejected = np.zeros(n, dtype=bool)
    corrected_p = np.ones(n)
    
    for i in range(n):
        if sorted_p_values[i] <= bh_critical[i]:
            rejected[sorted_indices[i]] = True
            corrected_p[sorted_indices[i]] = min(1.0, sorted_p_values[i] * n / (i + 1))
        else:
            break
    
    return corrected_p, rejected


def load_pklz_td_data(pklz_path: str, dataset_name: str) -> pd.DataFrame:
    """
    Load TD subjects from .pklz file(s).
    
    For CMI-HBN: pklz_path should be a directory containing multiple run1 .pklz files
    For ADHD200: pklz_path should be a single .pklz file
    
    Args:
        pklz_path: Path to .pklz file or directory containing .pklz files
        dataset_name: Name of dataset (for logging)
        
    Returns:
        DataFrame with TD subjects and behavioral data
    """
    print_step(1, f"LOADING {dataset_name.upper()} DATA", f"From {pklz_path}")
    
    # Check if path is a directory (CMI-HBN) or file (ADHD200)
    path_obj = Path(pklz_path)
    
    if path_obj.is_dir():
        # Load multiple .pklz files from directory
        print_info(f"Loading multiple .pklz files from directory...")
        from os import listdir
        from os.path import isfile, join
        
        files = [f for f in listdir(pklz_path) if isfile(join(pklz_path, f))]
        
        # For CMI-HBN: Load only run1 files
        # For ADHD200: Load all .pklz files
        if 'cmihbn' in dataset_name.lower():
            pklz_files = [f for f in files if 'run1' in f and f.endswith('.pklz')]
            print_info(f"CMI-HBN: Loading run1 files only")
        else:
            pklz_files = [f for f in files if f.endswith('.pklz')]
            print_info(f"ADHD200: Loading all .pklz files")
        
        print_info(f"Found {len(pklz_files)} .pklz files")
        
        data = None
        for i, filename in enumerate(pklz_files):
            file_path = join(pklz_path, filename)
            file_data = np.load(file_path, allow_pickle=True)
            
            if i == 0:
                data = file_data
            else:
                data = pd.concat([data, file_data], ignore_index=True)
        
        if data is None:
            print_error("No .pklz files found in directory")
            return pd.DataFrame()
        
        print_info(f"Loaded {len(data)} total subjects from {len(pklz_files)} files")
    else:
        # ADHD200: Load single file
        print_info(f"Loading single file: {path_obj.name}")
        data = np.load(pklz_path, allow_pickle=True)
        print_info(f"Loaded {len(data)} subjects")
    
    # Convert to proper types
    data['subject_id'] = data['subject_id'].astype('str')
    if 'site' in data.columns:
        data['site'] = data['site'].astype('str')
    
    # Convert DX to numeric (for ADHD200)
    if 'DX' in data.columns:
        data['DX'] = pd.to_numeric(data['DX'], errors='coerce')
    
    # Convert label to numeric (for CMI-HBN and ADHD200)
    if 'label' in data.columns:
        # Use pd.to_numeric directly (handles both numeric and string values)
        # errors='coerce' will convert 'pending', 'nan' strings to NaN
        data['label'] = pd.to_numeric(data['label'], errors='coerce')
    
    # Remove duplicates
    data = data.drop_duplicates(subset='subject_id', keep='first')
    data = data.reset_index(drop=True)
    
    print_info(f"Total subjects after deduplication: {len(data)}")
    
    # Filter for TD subjects
    if 'DX' in data.columns:
        # Remove subjects with NaN DX
        df_valid = data[data['DX'].notna()].copy()
        print_info(f"Valid subjects (non-NaN DX): {len(df_valid)}")
        
        # Filter for TD (DX == 0)
        df_td = df_valid[df_valid['DX'] == 0].copy()
    elif 'label' in data.columns:
        # Remove subjects with NaN labels (e.g., 'pending' or 'nan' converted to NaN)
        df_valid = data[data['label'].notna()].copy()
        print_info(f"Valid subjects (non-NaN labels): {len(df_valid)}")
        
        # Remove label == 99 (invalid/missing labels for both CMI-HBN and ADHD200)
        df_valid = df_valid[df_valid['label'] != 99].copy()
        print_info(f"Valid subjects (label != 99): {len(df_valid)}")
        
        # Filter for TD (label == 0)
        df_td = df_valid[df_valid['label'] == 0].copy()
    else:
        print_warning("No DX or label column found, using all subjects")
        df_td = data.copy()
    
    print_info(f"TD subjects: {len(df_td)}")
    
    # Filter for quality (mean_fd < 0.5)
    if 'mean_fd' in df_td.columns:
        n_before = len(df_td)
        df_td = df_td[df_td['mean_fd'] < 0.5].copy()
        n_removed = n_before - len(df_td)
        if n_removed > 0:
            print_info(f"Removed {n_removed} subjects with mean_fd >= 0.5")
            print_info(f"Remaining subjects: {len(df_td)}")
    
    # Check for behavioral columns
    behavioral_cols = ['Hyper/Impulsive', 'Inattentive', 'HY', 'IN']
    available_cols = [col for col in behavioral_cols if col in df_td.columns]
    
    if available_cols:
        print_info(f"Available behavioral columns: {available_cols}")
        
        # Remove subjects with NaN in behavioral columns
        for col in available_cols:
            n_nan = df_td[col].isna().sum()
            if n_nan > 0:
                print_warning(f"Removing {n_nan} subjects with NaN in {col}")
                df_td = df_td[df_td[col].notna()]
    else:
        print_warning(f"No behavioral columns found. Available: {list(df_td.columns)}")
    
    print_success(f"Final TD subjects with complete data: {len(df_td)}")
    
    return df_td


def load_ig_csv(ig_csv: str) -> pd.DataFrame:
    """
    Load IG scores from CSV.
    
    Automatically detects and uses all columns except 'subject_id'/'id' as ROI features.
    ROI columns can have any names (e.g., subdivision names, Brodmann areas, etc.).
    """
    print_step(2, "LOADING IG SCORES", f"From {Path(ig_csv).name}")
    
    ig_data = pd.read_csv(ig_csv)
    
    # Standardize ID column name
    if 'id' in ig_data.columns and 'subject_id' not in ig_data.columns:
        ig_data = ig_data.rename(columns={'id': 'subject_id'})
    
    if 'subject_id' not in ig_data.columns:
        print_error("No 'subject_id' or 'id' column found in IG CSV")
        return pd.DataFrame()
    
    ig_data['subject_id'] = ig_data['subject_id'].astype('str')
    
    # Get ROI column names (all except subject_id)
    roi_columns = [col for col in ig_data.columns if col != 'subject_id']
    
    print_info(f"IG subjects: {len(ig_data)}")
    print_info(f"IG features (ROIs): {len(roi_columns)}")
    print_info(f"Sample ROI columns: {roi_columns[:3]}...")
    
    return ig_data


def merge_data(ig_data: pd.DataFrame, behavioral_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Merge IG and behavioral data by subject ID."""
    print_step(3, "MERGING DATA", "Matching subject IDs")
    
    # Find common subjects
    ig_ids = set(ig_data['subject_id'])
    behavior_ids = set(behavioral_data['subject_id'])
    common_ids = ig_ids.intersection(behavior_ids)
    
    print_info(f"IG subjects: {len(ig_ids)}")
    print_info(f"Behavioral subjects: {len(behavior_ids)}")
    print_info(f"Common subjects: {len(common_ids)}")
    
    if len(common_ids) == 0:
        print_error("No common subjects found!")
        return pd.DataFrame(), pd.DataFrame()
    
    # Filter and sort
    ig_merged = ig_data[ig_data['subject_id'].isin(common_ids)].sort_values('subject_id').reset_index(drop=True)
    behavior_merged = behavioral_data[behavioral_data['subject_id'].isin(common_ids)].sort_values('subject_id').reset_index(drop=True)
    
    print_success(f"Merged: {len(ig_merged)} subjects with both IG and behavioral data")
    
    return ig_merged, behavior_merged


def perform_pca(ig_matrix: np.ndarray, n_components: int = 10) -> Tuple[PCA, np.ndarray]:
    """Perform PCA on IG scores."""
    print_step(4, "PERFORMING PCA", f"Using {n_components} components")
    
    # Standardize
    scaler = StandardScaler()
    ig_scaled = scaler.fit_transform(ig_matrix)
    
    # PCA
    pca = PCA(n_components=n_components)
    pca_scores = pca.fit_transform(ig_scaled)
    
    # Variance explained
    variance_explained = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_explained)
    
    print_info(f"Variance explained by PC1-3: {cumulative_variance[2]:.2%}")
    print_info(f"Variance explained by all {n_components} PCs: {cumulative_variance[-1]:.2%}")
    
    return pca, pca_scores


def correlate_with_behavior(pca_scores: np.ndarray, behavioral_scores: np.ndarray, behavior_name: str) -> Dict:
    """Correlate PCA scores with behavioral scores."""
    print_step(5, f"CORRELATING WITH {behavior_name.upper()}", f"N = {len(behavioral_scores)}")
    
    n_components = pca_scores.shape[1]
    correlations = []
    p_values = []
    
    for i in range(n_components):
        r, p = pearsonr(pca_scores[:, i], behavioral_scores)
        correlations.append(r)
        p_values.append(p)
    
    # FDR correction
    corrected_p, rejected = benjamini_hochberg_correction(p_values)
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
    """Save results to CSV."""
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
    parser = argparse.ArgumentParser(description="Brain-behavior analysis for TD cohorts (simplified)")
    parser.add_argument('--dataset', type=str, required=True, choices=['cmihbn_td', 'adhd200_td'],
                       help='Dataset to analyze')
    parser.add_argument('--pklz_file', type=str, required=True,
                       help='Path to .pklz file with behavioral data')
    parser.add_argument('--ig_csv', type=str, required=True,
                       help='Path to IG CSV file with subject_id column')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--n_components', type=int, default=10,
                       help='Number of PCA components (default: 10)')
    
    args = parser.parse_args()
    
    print_section_header(f"BRAIN-BEHAVIOR ANALYSIS - {args.dataset.upper()}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    behavioral_data = load_pklz_td_data(args.pklz_file, args.dataset)
    print()
    
    ig_data = load_ig_csv(args.ig_csv)
    if ig_data.empty:
        return
    print()
    
    # Merge
    ig_merged, behavior_merged = merge_data(ig_data, behavioral_data)
    if ig_merged.empty:
        return
    print()
    
    # Prepare IG matrix
    roi_columns = [col for col in ig_merged.columns if col != 'subject_id']
    ig_matrix = ig_merged[roi_columns].values
    print_info(f"IG matrix shape: {ig_matrix.shape} (subjects x ROIs)")
    print()
    
    # PCA
    pca, pca_scores = perform_pca(ig_matrix, n_components=args.n_components)
    print()
    
    # Determine behavioral columns based on dataset
    if args.dataset == 'cmihbn_td':
        behavioral_cols = {'HY': 'HY', 'IN': 'IN'}
    else:  # adhd200_td
        behavioral_cols = {'HY': 'Hyper/Impulsive', 'IN': 'Inattentive'}
    
    # Correlate with behavioral measures
    output_files = []
    for short_name, full_name in behavioral_cols.items():
        if full_name in behavior_merged.columns:
            behavioral_scores = behavior_merged[full_name].values
            
            # Remove NaNs
            valid_mask = ~np.isnan(behavioral_scores)
            if not valid_mask.all():
                n_removed = (~valid_mask).sum()
                print_warning(f"Removing {n_removed} subjects with NaN in {full_name}")
                behavioral_scores = behavioral_scores[valid_mask]
                pca_scores_filtered = pca_scores[valid_mask]
            else:
                pca_scores_filtered = pca_scores
            
            results = correlate_with_behavior(pca_scores_filtered, behavioral_scores, short_name)
            
            # Save
            output_path = output_dir / f"{args.dataset}_{short_name}_pca_correlations.csv"
            save_results(results, str(output_path))
            output_files.append(str(output_path))
            print()
        else:
            print_warning(f"Behavioral column '{full_name}' not found")
            print()
    
    print_completion(f"{args.dataset.upper()} Brain-Behavior Analysis", output_files)


if __name__ == "__main__":
    main()

