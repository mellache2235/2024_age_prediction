#!/usr/bin/env python3
"""
Brain-behavior correlation analysis for CMIHBN TD and ADHD200 TD cohorts.

This script performs brain-behavior correlation analysis with FDR correction
for multiple behavioral metrics.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / 'utils'))

from data_utils import load_finetune_dataset_w_ids, reshape_data
from statistical_utils import (
    multiple_correlation_analysis,
    apply_multiple_comparison_correction,
    correlation_analysis
)
from plotting_utils import (
    plot_correlation_matrix,
    plot_age_prediction,
    save_figure
)


def load_behavioral_data(data_path: str, 
                        behavioral_columns: List[str]) -> pd.DataFrame:
    """
    Load behavioral data from CSV file.
    
    Args:
        data_path (str): Path to behavioral data CSV
        behavioral_columns (List[str]): List of behavioral column names
        
    Returns:
        pd.DataFrame: Behavioral data
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Behavioral data file not found: {data_path}")
    
    behavioral_data = pd.read_csv(data_path)
    
    # Check if required columns exist
    missing_columns = [col for col in behavioral_columns if col not in behavioral_data.columns]
    if missing_columns:
        print(f"Warning: Missing behavioral columns: {missing_columns}")
        available_columns = [col for col in behavioral_columns if col in behavioral_data.columns]
        behavioral_columns = available_columns
    
    return behavioral_data[behavioral_columns + ['subject_id']]


def extract_brain_features(feature_data: np.ndarray, 
                         feature_indices: Optional[List[int]] = None) -> np.ndarray:
    """
    Extract brain features from feature attribution data.
    
    Args:
        feature_data (np.ndarray): Feature attribution data
        feature_indices (List[int], optional): Indices of features to extract
        
    Returns:
        np.ndarray: Extracted brain features
    """
    if feature_indices is not None:
        return feature_data[:, feature_indices]
    else:
        # Use mean across all features
        return np.mean(feature_data, axis=1)


def perform_brain_behavior_correlation(ig_scores: np.ndarray,
                                     behavioral_data: pd.DataFrame,
                                     subject_ids: List[str],
                                     behavioral_columns: List[str],
                                     correction_method: str = 'fdr_bh',
                                     alpha: float = 0.05) -> pd.DataFrame:
    """
    Perform brain-behavior correlation analysis using IG scores for each individual.
    
    Args:
        ig_scores (np.ndarray): Integrated Gradient scores for each individual (n_subjects × n_features)
        behavioral_data (pd.DataFrame): Behavioral data
        subject_ids (List[str]): Subject IDs
        behavioral_columns (List[str]): Behavioral column names
        correction_method (str): Multiple comparison correction method
        alpha (float): Significance level
        
    Returns:
        pd.DataFrame: Correlation results with corrected p-values
    """
    # Create combined dataset using IG scores
    combined_data = pd.DataFrame({
        'subject_id': subject_ids,
        'ig_scores': ig_scores.tolist()
    })
    
    # Merge with behavioral data
    merged_data = combined_data.merge(behavioral_data, on='subject_id', how='inner')
    
    if len(merged_data) == 0:
        print("Warning: No matching subjects found between brain and behavioral data")
        return pd.DataFrame()
    
    # Perform multiple correlation analysis
    correlation_results = multiple_correlation_analysis(
        data=merged_data,
        target_variable='brain_features',
        predictor_variables=behavioral_columns,
        method='pearson',
        correction_method=correction_method,
        alpha=alpha
    )
    
    return correlation_results


def analyze_site_specific_correlations(brain_features: np.ndarray,
                                     behavioral_data: pd.DataFrame,
                                     subject_ids: List[str],
                                     site_info: List[str],
                                     behavioral_columns: List[str],
                                     correction_method: str = 'fdr_bh',
                                     alpha: float = 0.05) -> Dict[str, pd.DataFrame]:
    """
    Perform site-specific brain-behavior correlation analysis.
    
    Args:
        brain_features (np.ndarray): Brain feature data
        behavioral_data (pd.DataFrame): Behavioral data
        subject_ids (List[str]): Subject IDs
        site_info (List[str]): Site information for each subject
        behavioral_columns (List[str]): Behavioral column names
        correction_method (str): Multiple comparison correction method
        alpha (float): Significance level
        
    Returns:
        Dict[str, pd.DataFrame]: Site-specific correlation results
    """
    # Create combined dataset with site information
    combined_data = pd.DataFrame({
        'subject_id': subject_ids,
        'brain_features': brain_features.tolist(),
        'site': site_info
    })
    
    # Merge with behavioral data
    merged_data = combined_data.merge(behavioral_data, on='subject_id', how='inner')
    
    if len(merged_data) == 0:
        print("Warning: No matching subjects found between brain and behavioral data")
        return {}
    
    # Get unique sites
    unique_sites = merged_data['site'].unique()
    site_results = {}
    
    for site in unique_sites:
        print(f"Analyzing site: {site}")
        
        # Filter data for this site
        site_data = merged_data[merged_data['site'] == site]
        
        if len(site_data) < 10:  # Need minimum sample size
            print(f"Warning: Site {site} has only {len(site_data)} subjects, skipping...")
            continue
        
        # Perform correlation analysis for this site using IG scores
        site_correlations = multiple_correlation_analysis(
            data=site_data,
            target_variable='ig_scores',
            predictor_variables=behavioral_columns,
            method='pearson',
            correction_method=correction_method,
            alpha=alpha
        )
        
        site_results[site] = site_correlations
    
    return site_results


def create_correlation_visualizations(correlation_results: pd.DataFrame,
                                    output_dir: str = "results/brain_behavior") -> None:
    """
    Create visualizations for correlation results.
    
    Args:
        correlation_results (pd.DataFrame): Correlation results
        output_dir (str): Output directory for figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if correlation_results.empty:
        print("No correlation results to visualize")
        return
    
    # Create correlation matrix
    behavioral_vars = correlation_results['predictor'].values
    correlations = correlation_results['correlation'].values
    
    # Create a simple correlation matrix (1x1 for single brain feature)
    corr_matrix = np.array([[1.0, correlations[0]], [correlations[0], 1.0]])
    
    # Plot correlation matrix
    corr_fig = plot_correlation_matrix(
        corr_matrix,
        ['Brain Features'] + behavioral_vars.tolist(),
        title="Brain-Behavior Correlation Matrix",
        save_path=os.path.join(output_dir, "correlation_matrix.png")
    )
    
    # Create summary plot of correlations
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by correlation strength
    sorted_results = correlation_results.sort_values('correlation', key=abs, ascending=False)
    
    # Create bar plot
    bars = ax.barh(range(len(sorted_results)), sorted_results['correlation'])
    
    # Color bars by significance
    colors = ['red' if sig else 'gray' for sig in sorted_results['significant']]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add labels
    ax.set_yticks(range(len(sorted_results)))
    ax.set_yticklabels(sorted_results['predictor'])
    ax.set_xlabel('Correlation Coefficient')
    ax.set_title('Brain-Behavior Correlations')
    
    # Add significance indicators
    for i, (corr, sig) in enumerate(zip(sorted_results['correlation'], sorted_results['significant'])):
        ax.text(corr + 0.01 if corr > 0 else corr - 0.01, i, 
                '*' if sig else '', ha='left' if corr > 0 else 'right', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()


def analyze_cmihbn_td_cohort(data_dir: str,
                           output_dir: str = "results/brain_behavior/cmihbn_td") -> Dict[str, any]:
    """
    Analyze CMIHBN TD cohort.
    
    Args:
        data_dir (str): Directory containing CMIHBN data
        output_dir (str): Output directory
        
    Returns:
        Dict[str, any]: Analysis results
    """
    print("Analyzing CMIHBN TD cohort...")
    
    # Define behavioral measures for CMIHBN
    behavioral_columns = [
        'age', 'sex', 'iq', 'adhd_symptoms', 'anxiety_symptoms', 'depression_symptoms'
    ]
    
    # Load data (placeholder - adjust paths as needed)
    # This is a template - actual implementation would depend on data structure
    try:
        # Load brain features
        brain_data_path = os.path.join(data_dir, "cmihbn_td_brain_features.npz")
        if os.path.exists(brain_data_path):
            brain_data = np.load(brain_data_path)
            brain_features = brain_data['features']
            subject_ids = brain_data['subject_ids']
        else:
            print(f"Warning: Brain data not found at {brain_data_path}")
            return {}
        
        # Load behavioral data
        behavioral_data_path = os.path.join(data_dir, "cmihbn_td_behavioral.csv")
        behavioral_data = load_behavioral_data(behavioral_data_path, behavioral_columns)
        
        # Perform correlation analysis
        correlation_results = perform_brain_behavior_correlation(
            brain_features, behavioral_data, subject_ids, behavioral_columns
        )
        
        # Create visualizations
        create_correlation_visualizations(correlation_results, output_dir)
        
        # Save results
        correlation_results.to_csv(
            os.path.join(output_dir, "cmihbn_td_correlations.csv"), index=False
        )
        
        return {
            'correlation_results': correlation_results,
            'n_subjects': len(subject_ids),
            'behavioral_measures': behavioral_columns
        }
        
    except Exception as e:
        print(f"Error analyzing CMIHBN TD cohort: {e}")
        return {}


def analyze_adhd200_td_cohort(data_dir: str,
                            output_dir: str = "results/brain_behavior/adhd200_td") -> Dict[str, any]:
    """
    Analyze ADHD200 TD cohort with site-specific analysis.
    
    Args:
        data_dir (str): Directory containing ADHD200 data
        output_dir (str): Output directory
        
    Returns:
        Dict[str, any]: Analysis results
    """
    print("Analyzing ADHD200 TD cohort...")
    
    # Define behavioral measures for ADHD200
    behavioral_columns = [
        'age', 'sex', 'iq', 'adhd_symptoms', 'anxiety_symptoms', 'depression_symptoms'
    ]
    
    try:
        # Load brain features
        brain_data_path = os.path.join(data_dir, "adhd200_td_brain_features.npz")
        if os.path.exists(brain_data_path):
            brain_data = np.load(brain_data_path)
            brain_features = brain_data['features']
            subject_ids = brain_data['subject_ids']
            site_info = brain_data['sites']
        else:
            print(f"Warning: Brain data not found at {brain_data_path}")
            return {}
        
        # Load behavioral data
        behavioral_data_path = os.path.join(data_dir, "adhd200_td_behavioral.csv")
        behavioral_data = load_behavioral_data(behavioral_data_path, behavioral_columns)
        
        # Perform overall correlation analysis
        overall_results = perform_brain_behavior_correlation(
            brain_features, behavioral_data, subject_ids, behavioral_columns
        )
        
        # Perform site-specific analysis
        site_results = analyze_site_specific_correlations(
            brain_features, behavioral_data, subject_ids, site_info, behavioral_columns
        )
        
        # Create visualizations
        create_correlation_visualizations(overall_results, output_dir)
        
        # Save results
        overall_results.to_csv(
            os.path.join(output_dir, "adhd200_td_overall_correlations.csv"), index=False
        )
        
        # Save site-specific results
        for site, results in site_results.items():
            results.to_csv(
                os.path.join(output_dir, f"adhd200_td_site_{site}_correlations.csv"), 
                index=False
            )
        
        return {
            'overall_correlation_results': overall_results,
            'site_specific_results': site_results,
            'n_subjects': len(subject_ids),
            'n_sites': len(set(site_info)),
            'behavioral_measures': behavioral_columns
        }
        
    except Exception as e:
        print(f"Error analyzing ADHD200 TD cohort: {e}")
        return {}


def main():
    """Main function for brain-behavior correlation analysis."""
    parser = argparse.ArgumentParser(description="Brain-behavior correlation analysis")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing data files")
    parser.add_argument("--cohort", type=str, choices=['cmihbn_td', 'adhd200_td', 'both'],
                       default='both', help="Cohort to analyze")
    parser.add_argument("--output_dir", type=str, default="results/brain_behavior",
                       help="Output directory for results")
    parser.add_argument("--correction_method", type=str, default='fdr_bh',
                       choices=['fdr_bh', 'bonferroni', 'holm'],
                       help="Multiple comparison correction method")
    parser.add_argument("--alpha", type=float, default=0.05,
                       help="Significance level")
    
    args = parser.parse_args()
    
    results = {}
    
    if args.cohort in ['cmihbn_td', 'both']:
        cmihbn_results = analyze_cmihbn_td_cohort(
            args.data_dir, 
            os.path.join(args.output_dir, "cmihbn_td")
        )
        results['cmihbn_td'] = cmihbn_results
    
    if args.cohort in ['adhd200_td', 'both']:
        adhd200_results = analyze_adhd200_td_cohort(
            args.data_dir,
            os.path.join(args.output_dir, "adhd200_td")
        )
        results['adhd200_td'] = adhd200_results
    
    print(f"Brain-behavior correlation analysis completed. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
