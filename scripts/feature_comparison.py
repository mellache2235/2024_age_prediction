#!/usr/bin/env python3
"""
Refactored feature comparison script with improved structure and PEP8 compliance.

This script compares feature attributions between different cohorts and computes
similarity metrics with proper statistical analysis.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / 'utils'))

from data_utils import detect_roi_columns
from statistical_utils import compute_similarity_metrics
import matplotlib.pyplot as plt


class FeatureComparison:
    """
    Class for comparing feature attributions between cohorts.
    """
    
    def __init__(self, top_fraction: float = 0.5, selection_mode: str = "mean_IG"):
        """
        Initialize FeatureComparison.
        
        Args:
            top_fraction (float): Fraction of top features to select
            selection_mode (str): Selection mode ('mean_IG' or 'rank_based')
        """
        self.top_fraction = top_fraction
        self.selection_mode = selection_mode
        self.non_roi_candidates = {
            "subject_id", "participant_id", "subid", "sub_id", "id",
            "Unnamed: 0", "Unnamed: 0.1", "Unnamed: 0.2"
        }
    
    def load_feature_data(self, file_path: str) -> pd.DataFrame:
        """
        Load feature attribution data from CSV file.
        
        Args:
            file_path (str): Path to CSV file
            
        Returns:
            pd.DataFrame: Feature data
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Feature file not found: {file_path}")
        
        return pd.read_csv(file_path)
    
    def select_top_features_mean_ig(self, df: pd.DataFrame) -> pd.Series:
        """
        Select top features based on mean IG across subjects.
        
        Args:
            df (pd.DataFrame): Feature data
            
        Returns:
            pd.Series: Top features ranked by mean IG
        """
        roi_cols = detect_roi_columns(df, self.non_roi_candidates)
        mean_ig = df[roi_cols].mean(axis=0)
        n_top = max(1, int(np.floor(len(mean_ig) * self.top_fraction)))
        return mean_ig.nlargest(n_top)
    
    def select_top_features_rank_based(self, df: pd.DataFrame) -> pd.Series:
        """
        Select top features based on rank-based approach.
        
        Args:
            df (pd.DataFrame): Feature data
            
        Returns:
            pd.Series: Top features ranked by median percentile rank
        """
        roi_cols = detect_roi_columns(df, self.non_roi_candidates)
        
        # Compute percentile ranks for each subject
        ranks = df[roi_cols].rank(axis=1, method="average", ascending=False)
        n_features = len(roi_cols)
        percentiles = 1.0 - (ranks - 1) / (n_features - 1)
        
        # Compute median percentile rank across subjects
        median_percentiles = percentiles.median(axis=0)
        n_top = max(1, int(np.floor(len(median_percentiles) * self.top_fraction)))
        
        return median_percentiles.nlargest(n_top)
    
    def select_top_features(self, df: pd.DataFrame) -> pd.Series:
        """
        Select top features based on the specified selection mode.
        
        Args:
            df (pd.DataFrame): Feature data
            
        Returns:
            pd.Series: Top features
        """
        if self.selection_mode == "mean_IG":
            return self.select_top_features_mean_ig(df)
        elif self.selection_mode == "rank_based":
            return self.select_top_features_rank_based(df)
        else:
            raise ValueError(f"Unknown selection mode: {self.selection_mode}")
    
    def compute_overlap_metrics(self, features_a: pd.Series, 
                              features_b: pd.Series) -> Dict[str, float]:
        """
        Compute overlap metrics between two feature sets.
        
        Args:
            features_a (pd.Series): First set of features
            features_b (pd.Series): Second set of features
            
        Returns:
            Dict[str, float]: Overlap metrics
        """
        set_a = set(features_a.index)
        set_b = set(features_b.index)
        
        return compute_similarity_metrics(set_a, set_b)
    
    def compute_cosine_similarity(self, features_a: pd.Series, 
                                features_b: pd.Series) -> float:
        """
        Compute cosine similarity between feature vectors.
        
        Args:
            features_a (pd.Series): First feature vector
            features_b (pd.Series): Second feature vector
            
        Returns:
            float: Cosine similarity
        """
        # Get common features
        common_features = set(features_a.index) & set(features_b.index)
        
        if not common_features:
            return 0.0
        
        # Extract values for common features
        values_a = features_a[list(common_features)].values
        values_b = features_b[list(common_features)].values
        
        # Compute cosine similarity
        dot_product = np.dot(values_a, values_b)
        norm_a = np.linalg.norm(values_a)
        norm_b = np.linalg.norm(values_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def compare_cohorts(self, file_a: str, file_b: str, 
                       output_dir: str) -> Dict[str, any]:
        """
        Compare feature attributions between two cohorts.
        
        Args:
            file_a (str): Path to first cohort's feature file
            file_b (str): Path to second cohort's feature file
            output_dir (str): Output directory for results
            
        Returns:
            Dict[str, any]: Comparison results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        print(f"Loading data from {file_a} and {file_b}")
        df_a = self.load_feature_data(file_a)
        df_b = self.load_feature_data(file_b)
        
        # Select top features
        print(f"Selecting top {self.top_fraction*100}% features using {self.selection_mode} mode")
        top_features_a = self.select_top_features(df_a)
        top_features_b = self.select_top_features(df_b)
        
        # Compute overlap metrics
        print("Computing overlap metrics...")
        overlap_metrics = self.compute_overlap_metrics(top_features_a, top_features_b)
        
        # Compute cosine similarity
        cosine_sim = self.compute_cosine_similarity(top_features_a, top_features_b)
        overlap_metrics['cosine_similarity'] = cosine_sim
        
        # Create detailed results
        intersection_features = set(top_features_a.index) & set(top_features_b.index)
        intersection_df = pd.DataFrame({
            'feature': list(intersection_features),
            'cohort_a_score': [top_features_a.get(f, 0) for f in intersection_features],
            'cohort_b_score': [top_features_b.get(f, 0) for f in intersection_features]
        })
        intersection_df = intersection_df.sort_values('cohort_a_score', ascending=False)
        
        # Save results
        results = {
            'overlap_metrics': overlap_metrics,
            'top_features_a': top_features_a,
            'top_features_b': top_features_b,
            'intersection_features': intersection_df
        }
        
        # Save to files
        with open(os.path.join(output_dir, 'overlap_metrics.json'), 'w') as f:
            json.dump(overlap_metrics, f, indent=2)
        
        intersection_df.to_csv(
            os.path.join(output_dir, 'overlap_roi_list.csv'), index=False
        )
        
        top_features_a.to_csv(
            os.path.join(output_dir, 'ranked_IG_A.csv'), header=[self.selection_mode]
        )
        
        top_features_b.to_csv(
            os.path.join(output_dir, 'ranked_IG_B.csv'), header=[self.selection_mode]
        )
        
        # Note: Visualizations are now created using separate plotting scripts
        
        return results
    
        
        # Create summary plot
        metrics = results['overlap_metrics']
        metric_names = ['jaccard', 'dice', 'cosine_similarity']
        metric_values = [metrics.get(name, 0) for name in metric_names]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(metric_names, metric_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=12)
        
        ax.set_ylabel('Similarity Score')
        ax.set_title('Feature Overlap Metrics')
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'overlap_metrics.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function for feature comparison."""
    parser = argparse.ArgumentParser(
        description="Compare feature attributions between cohorts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two cohorts with default settings
  python feature_comparison.py --file_a cohort1.csv --file_b cohort2.csv --output_dir results

  # Use rank-based selection with top 30% features
  python feature_comparison.py --file_a cohort1.csv --file_b cohort2.csv \\
    --output_dir results --selection_mode rank_based --top_fraction 0.3
        """
    )
    
    parser.add_argument("--file_a", type=str, required=True,
                       help="Path to first cohort's feature file")
    parser.add_argument("--file_b", type=str, required=True,
                       help="Path to second cohort's feature file")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for results")
    parser.add_argument("--top_fraction", type=float, default=0.5,
                       help="Fraction of top features to select (default: 0.5)")
    parser.add_argument("--selection_mode", type=str, default="mean_IG",
                       choices=["mean_IG", "rank_based"],
                       help="Feature selection mode (default: mean_IG)")
    
    args = parser.parse_args()
    
    # Create feature comparison object
    comparator = FeatureComparison(
        top_fraction=args.top_fraction,
        selection_mode=args.selection_mode
    )
    
    # Run comparison
    try:
        results = comparator.compare_cohorts(
            args.file_a, args.file_b, args.output_dir
        )
        
        print(f"Feature comparison completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        print(f"Overlap metrics: {results['overlap_metrics']}")
        
    except Exception as e:
        print(f"Error in feature comparison: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
