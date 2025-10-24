#!/usr/bin/env python3
"""
Network-level analysis using consensus count data and Yeo atlas.

This script performs network-level analysis of feature attributions using
consensus count data and maps features to Yeo 17-network atlas.
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

from feature_utils import (
    compute_consensus_features_across_models,
    map_features_to_networks,
    compute_network_level_scores,
    create_feature_consensus_nifti,
    create_feature_summary_table
)
from plotting_utils import (
    plot_network_analysis,
    plot_feature_importance,
    save_figure
)
from statistical_utils import compute_similarity_metrics


def load_yeo_atlas(yeo_path: str) -> pd.DataFrame:
    """
    Load Yeo 17-network atlas.
    
    Args:
        yeo_path (str): Path to Yeo atlas CSV file
        
    Returns:
        pd.DataFrame: Yeo atlas data
    """
    if not os.path.exists(yeo_path):
        raise FileNotFoundError(f"Yeo atlas file not found: {yeo_path}")
    
    yeo_atlas = pd.read_csv(yeo_path)
    return yeo_atlas


def analyze_network_consensus(ig_dir: str,
                            model_prefix: str,
                            num_models: int,
                            yeo_atlas_path: str,
                            percentile: float = 95.0,
                            output_dir: str = "results/network_analysis") -> Dict[str, any]:
    """
    Perform network-level consensus analysis.
    
    Args:
        ig_dir (str): Directory containing integrated gradients files
        model_prefix (str): Prefix for model files
        num_models (int): Number of models to analyze
        yeo_atlas_path (str): Path to Yeo atlas CSV file
        percentile (float): Percentile threshold for feature selection
        output_dir (str): Output directory for results
        
    Returns:
        Dict[str, any]: Analysis results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Yeo atlas
    yeo_atlas = load_yeo_atlas(yeo_atlas_path)
    
    # Compute consensus features
    print(f"Computing consensus features across {num_models} models...")
    consensus_counts = compute_consensus_features_across_models(
        ig_dir, model_prefix, num_models, percentile
    )
    
    if not consensus_counts:
        print("Warning: No consensus features found!")
        return {}
    
    # Map features to networks
    print("Mapping features to Yeo networks...")
    feature_indices = list(consensus_counts.keys())
    network_mapping = map_features_to_networks(feature_indices, yeo_atlas_path)
    
    # Compute network-level scores
    print("Computing network-level scores...")
    feature_scores = np.array([consensus_counts[i] for i in feature_indices])
    network_scores = compute_network_level_scores(
        feature_scores, network_mapping, aggregation_method='mean'
    )
    
    # Create feature summary table
    print("Creating feature summary table...")
    roi_labels = [f"ROI_{i}" for i in feature_indices]  # Placeholder labels
    feature_summary = create_feature_summary_table(
        consensus_counts, roi_labels, network_mapping,
        output_path=os.path.join(output_dir, "feature_summary.csv")
    )
    
    # Create network summary
    network_summary = pd.DataFrame([
        {'network': network, 'score': score, 'n_features': len(features)}
        for network, (score, features) in zip(
            network_scores.keys(), 
            [(network_scores[net], network_mapping[net]) for net in network_scores.keys()]
        )
    ]).sort_values('score', ascending=False)
    
    network_summary.to_csv(
        os.path.join(output_dir, "network_summary.csv"), index=False
    )
    
    # Create visualizations
    print("Creating visualizations...")
    
    # Network-level plot
    network_fig = plot_network_analysis(
        network_scores,
        title="Network-level Feature Importance",
        save_path=os.path.join(output_dir, "network_analysis.png")
    )
    
    # Feature-level plot
    feature_fig = plot_feature_importance(
        feature_scores,
        roi_labels,
        title="Top Feature Importance",
        top_n=20,
        save_path=os.path.join(output_dir, "feature_importance.png")
    )
    
    # Save results
    results = {
        'consensus_counts': consensus_counts,
        'network_mapping': network_mapping,
        'network_scores': network_scores,
        'feature_summary': feature_summary,
        'network_summary': network_summary,
        'yeo_atlas': yeo_atlas
    }
    
    return results


def compare_networks_across_cohorts(cohort_results: Dict[str, Dict],
                                  output_dir: str = "results/network_analysis") -> pd.DataFrame:
    """
    Compare network-level results across cohorts.
    
    Args:
        cohort_results (Dict[str, Dict]): Results from different cohorts
        output_dir (str): Output directory
        
    Returns:
        pd.DataFrame: Comparison results
    """
    # Extract network scores for each cohort
    cohort_network_scores = {}
    for cohort, results in cohort_results.items():
        if 'network_scores' in results:
            cohort_network_scores[cohort] = results['network_scores']
    
    # Create comparison DataFrame
    all_networks = set()
    for scores in cohort_network_scores.values():
        all_networks.update(scores.keys())
    
    comparison_data = []
    for network in all_networks:
        row = {'network': network}
        for cohort, scores in cohort_network_scores.items():
            row[f'{cohort}_score'] = scores.get(network, 0.0)
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values(
        comparison_df.columns[1], ascending=False  # Sort by first cohort's scores
    )
    
    # Save comparison
    comparison_df.to_csv(
        os.path.join(output_dir, "network_comparison_across_cohorts.csv"), 
        index=False
    )
    
    return comparison_df


def create_network_consensus_nifti(consensus_counts: Dict[int, int],
                                 atlas_nifti_path: str,
                                 output_dir: str = "results/network_analysis") -> None:
    """
    Create NIfTI files with consensus feature counts.
    
    Args:
        consensus_counts (Dict[int, int]): Feature consensus counts
        atlas_nifti_path (str): Path to atlas NIfTI file
        output_dir (str): Output directory
    """
    if not os.path.exists(atlas_nifti_path):
        print(f"Warning: Atlas NIfTI file not found: {atlas_nifti_path}")
        return
    
    # Create normalized consensus map
    create_feature_consensus_nifti(
        consensus_counts,
        atlas_nifti_path,
        os.path.join(output_dir, "consensus_features_normalized.nii.gz"),
        normalize=True
    )
    
    # Create raw consensus map
    create_feature_consensus_nifti(
        consensus_counts,
        atlas_nifti_path,
        os.path.join(output_dir, "consensus_features_raw.nii.gz"),
        normalize=False
    )


def main():
    """Main function for network analysis."""
    parser = argparse.ArgumentParser(description="Network-level analysis using consensus count data")
    parser.add_argument("--ig_dir", type=str, required=True,
                       help="Directory containing integrated gradients files")
    parser.add_argument("--model_prefix", type=str, required=True,
                       help="Prefix for model files")
    parser.add_argument("--num_models", type=int, required=True,
                       help="Number of models to analyze")
    parser.add_argument("--yeo_atlas", type=str, required=True,
                       help="Path to Yeo atlas CSV file")
    parser.add_argument("--atlas_nifti", type=str,
                       help="Path to atlas NIfTI file")
    parser.add_argument("--percentile", type=float, default=95.0,
                       help="Percentile threshold for feature selection")
    parser.add_argument("--output_dir", type=str, default="results/network_analysis",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Perform network analysis
    results = analyze_network_consensus(
        ig_dir=args.ig_dir,
        model_prefix=args.model_prefix,
        num_models=args.num_models,
        yeo_atlas_path=args.yeo_atlas,
        percentile=args.percentile,
        output_dir=args.output_dir
    )
    
    # Create NIfTI files if atlas provided
    if args.atlas_nifti and 'consensus_counts' in results:
        create_network_consensus_nifti(
            results['consensus_counts'],
            args.atlas_nifti,
            args.output_dir
        )
    
    print(f"Network analysis completed. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
