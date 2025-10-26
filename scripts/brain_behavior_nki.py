#!/usr/bin/env python3
"""
Brain-behavior analysis for NKI-RS TD using IG CSV and behavioral data.

This script:
1. Loads IG CSV with subject IDs
2. Loads behavioral data (CAARS measures) from .bin file or CSV
3. Matches subjects between IG and behavioral data
4. Performs PCA on IG scores
5. Correlates PCA components with CAARS behavioral measures (HY, IN)
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
import pickle

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


def load_nki_behavioral_data(behavioral_file: str) -> pd.DataFrame:
    """
    Load NKI behavioral data from CAARS CSV file.
    
    Expected format: 8100_CAARS-S-S_20191009.csv with subject IDs and CAARS scores
    
    Args:
        behavioral_file: Path to CAARS CSV file
        
    Returns:
        DataFrame with behavioral data
    """
    print_step(1, "LOADING NKI CAARS DATA", f"From {Path(behavioral_file).name}")
    
    try:
        # Load CSV
        data = pd.read_csv(behavioral_file)
        
        print_info(f"Loaded {len(data)} rows")
        print_info(f"Columns: {list(data.columns)}")
        
        # Standardize subject ID column
        # Common column names: 'Anonymized ID', 'ID', 'Subject', 'subject_id', 'id'
        id_columns = [col for col in data.columns if any(x in col.lower() for x in ['id', 'subject', 'anonymized'])]
        
        if id_columns:
            # Use the first ID-like column
            id_col = id_columns[0]
            print_info(f"Using '{id_col}' as subject ID column")
            if id_col != 'subject_id':
                data = data.rename(columns={id_col: 'subject_id'})
        elif 'subject_id' not in data.columns:
            print_error("No subject ID column found in CAARS data")
            print_info(f"Available columns: {list(data.columns)}")
            return pd.DataFrame()
        
        data['subject_id'] = data['subject_id'].astype('str')
        
        # Look for CAARS columns (Hyperactivity/Impulsivity and Inattention)
        caars_cols = [col for col in data.columns if col != 'subject_id' and 
                     any(x in col.lower() for x in ['caars', 'hyper', 'inattent', 'impuls', 'adhd', 'dsmiv'])]
        
        if caars_cols:
            print_success(f"Found {len(caars_cols)} CAARS-related columns:")
            for col in caars_cols[:10]:  # Show first 10
                print_info(f"  - {col}")
            if len(caars_cols) > 10:
                print_info(f"  ... and {len(caars_cols) - 10} more")
        else:
            print_warning("No CAARS-related columns found")
            print_info("Looking for columns with: caars, hyper, inattent, impuls, adhd, dsmiv")
        
        print_success(f"Loaded CAARS data for {len(data)} subjects")
        
        return data
        
    except Exception as e:
        print_error(f"Error loading CAARS data: {e}")
        import traceback
        print_error(traceback.format_exc())
        return pd.DataFrame()


def load_ig_csv(ig_csv: str) -> pd.DataFrame:
    """Load IG scores from CSV."""
    print_step(2, "LOADING IG SCORES", f"From {Path(ig_csv).name}")
    
    ig_data = pd.read_csv(ig_csv)
    
    # Drop 'Unnamed: 0' column if it exists (index column from CSV)
    if 'Unnamed: 0' in ig_data.columns:
        ig_data = ig_data.drop(columns=['Unnamed: 0'])
        print_info("Dropped 'Unnamed: 0' index column")
    
    # Standardize ID column name
    if 'id' in ig_data.columns and 'subject_id' not in ig_data.columns:
        ig_data = ig_data.rename(columns={'id': 'subject_id'})
    
    if 'subject_id' not in ig_data.columns:
        print_error("No 'subject_id' or 'id' column found in IG CSV")
        return pd.DataFrame()
    
    ig_data['subject_id'] = ig_data['subject_id'].astype('str')
    
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
        print_info(f"Sample IG IDs: {list(ig_ids)[:5]}")
        print_info(f"Sample behavioral IDs: {list(behavior_ids)[:5]}")
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
    parser = argparse.ArgumentParser(description="Brain-behavior analysis for NKI-RS TD")
    parser.add_argument('--behavioral_file', type=str, required=True,
                       help='Path to behavioral data file (.bin or .csv with CAARS measures)')
    parser.add_argument('--ig_csv', type=str, required=True,
                       help='Path to IG CSV file with subject_id column')
    parser.add_argument('--behavioral_columns', type=str, nargs='+',
                       help='Behavioral column names (e.g., CAARS_HY CAARS_IN)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--n_components', type=int, default=10,
                       help='Number of PCA components (default: 10)')
    
    args = parser.parse_args()
    
    print_section_header("BRAIN-BEHAVIOR ANALYSIS - NKI-RS TD")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    behavioral_data = load_nki_behavioral_data(args.behavioral_file)
    if behavioral_data.empty:
        return
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
    
    # Determine behavioral columns
    if args.behavioral_columns:
        behavioral_cols = args.behavioral_columns
    else:
        # Auto-detect CAARS columns
        behavioral_cols = [col for col in behavior_merged.columns 
                          if any(x in col.lower() for x in ['caars', 'hyper', 'inattent']) 
                          and col != 'subject_id']
        print_info(f"Auto-detected behavioral columns: {behavioral_cols}")
    
    # Correlate with behavioral measures
    output_files = []
    for col in behavioral_cols:
        if col in behavior_merged.columns:
            behavioral_scores = behavior_merged[col].values
            
            # Convert to numeric, coercing errors to NaN
            try:
                behavioral_scores = pd.to_numeric(behavioral_scores, errors='coerce')
            except Exception as e:
                print_error(f"Could not convert {col} to numeric: {e}")
                continue
            
            # Convert to numpy array if it's a Series
            if isinstance(behavioral_scores, pd.Series):
                behavioral_scores = behavioral_scores.values
            
            # Ensure it's a numpy array
            behavioral_scores = np.array(behavioral_scores, dtype=float)
            
            # Remove NaNs
            valid_mask = ~np.isnan(behavioral_scores)
            n_valid = valid_mask.sum()
            
            if n_valid < 10:
                print_warning(f"Only {n_valid} valid subjects for {col}, skipping (need at least 10)")
                print()
                continue
            
            if not valid_mask.all():
                n_removed = (~valid_mask).sum()
                print_warning(f"Removing {n_removed} subjects with NaN in {col}")
                behavioral_scores = behavioral_scores[valid_mask]
                pca_scores_filtered = pca_scores[valid_mask]
            else:
                pca_scores_filtered = pca_scores
            
            # Verify lengths match
            if len(pca_scores_filtered) != len(behavioral_scores):
                print_error(f"Length mismatch: PCA scores={len(pca_scores_filtered)}, behavioral={len(behavioral_scores)}")
                print()
                continue
            
            results = correlate_with_behavior(pca_scores_filtered, behavioral_scores, col)
            
            # Save
            # Clean up filename (remove special characters)
            safe_col_name = col.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
            output_path = output_dir / f"nki_{safe_col_name}_pca_correlations.csv"
            save_results(results, str(output_path))
            output_files.append(str(output_path))
            print()
        else:
            print_warning(f"Behavioral column '{col}' not found in data")
            print()
    
    if output_files:
        print_completion("NKI-RS TD Brain-Behavior Analysis", output_files)
    else:
        print_error("No behavioral columns were analyzed")


if __name__ == "__main__":
    main()

