#!/usr/bin/env python3
"""
Create tables for regions of importance based on consensus count data.

This script generates comprehensive tables summarizing important brain regions
across different datasets and cohorts.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / 'utils'))

from feature_utils import (
    compute_consensus_features_across_models,
    create_feature_summary_table,
    compare_feature_sets_across_cohorts,
    extract_top_features_by_consensus
)
from data_utils import load_roi_labels


def create_consensus_region_table(consensus_counts: Dict[int, int],
                                roi_labels: List[str],
                                network_mapping: Optional[Dict[str, List[int]]] = None,
                                output_path: str = "results/tables/consensus_regions.csv") -> pd.DataFrame:
    """
    Create a table of consensus regions with their importance scores.
    
    Args:
        consensus_counts (Dict[int, int]): Feature consensus counts
        roi_labels (List[str]): ROI labels
        network_mapping (Dict[str, List[int]], optional): Network assignments
        output_path (str): Output path for the table
        
    Returns:
        pd.DataFrame: Consensus regions table
    """
    # Create summary data
    summary_data = []
    
    for feature_idx, count in consensus_counts.items():
        if feature_idx < len(roi_labels):
            row = {
                'roi_index': feature_idx,
                'roi_name': roi_labels[feature_idx],
                'consensus_count': count,
                'consensus_percentage': count / max(consensus_counts.values()) * 100,
                'importance_rank': 0  # Will be set after sorting
            }
            
            # Add network information if available
            if network_mapping:
                for network, features in network_mapping.items():
                    if feature_idx in features:
                        row['network'] = network
                        break
                else:
                    row['network'] = 'Unknown'
            
            summary_data.append(row)
    
    # Create DataFrame and sort by consensus count
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('consensus_count', ascending=False)
    summary_df['importance_rank'] = range(1, len(summary_df) + 1)
    
    # Save table
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    print(f"Consensus regions table saved to: {output_path}")
    
    return summary_df


def create_overlap_analysis_table(cohort_features: Dict[str, List[int]],
                                cohort_names: List[str],
                                roi_labels: List[str],
                                output_path: str = "results/tables/overlap_analysis.csv") -> pd.DataFrame:
    """
    Create a table analyzing overlap between cohorts.
    
    Args:
        cohort_features (Dict[str, List[int]]): Dictionary mapping cohort names to feature lists
        cohort_names (List[str]): Names of cohorts
        roi_labels (List[str]): ROI labels
        output_path (str): Output path for the table
        
    Returns:
        pd.DataFrame: Overlap analysis table
    """
    # Compare feature sets across cohorts
    comparison_df = compare_feature_sets_across_cohorts(cohort_features, cohort_names)
    
    # Add ROI information for intersection features
    if len(comparison_df) > 0:
        # Get all unique features across cohorts
        all_features = set()
        for features in cohort_features.values():
            all_features.update(features)
        
        # Create feature information table
        feature_info = []
        for feature_idx in all_features:
            if feature_idx < len(roi_labels):
                feature_info.append({
                    'roi_index': feature_idx,
                    'roi_name': roi_labels[feature_idx]
                })
        
        feature_info_df = pd.DataFrame(feature_info)
        
        # Save tables
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        comparison_df.to_csv(output_path, index=False)
        feature_info_df.to_csv(output_path.replace('.csv', '_features.csv'), index=False)
        
        print(f"Overlap analysis table saved to: {output_path}")
        print(f"Feature information table saved to: {output_path.replace('.csv', '_features.csv')}")
    
    return comparison_df


def create_network_summary_table(network_scores: Dict[str, float],
                               network_mapping: Dict[str, List[int]],
                               roi_labels: List[str],
                               output_path: str = "results/tables/network_summary.csv") -> pd.DataFrame:
    """
    Create a summary table of network-level analysis.
    
    Args:
        network_scores (Dict[str, float]): Network-level scores
        network_mapping (Dict[str, List[int]]): Network to feature mapping
        roi_labels (List[str]): ROI labels
        output_path (str): Output path for the table
        
    Returns:
        pd.DataFrame: Network summary table
    """
    network_summary = []
    
    for network, score in network_scores.items():
        features = network_mapping.get(network, [])
        
        # Get ROI names for features in this network
        roi_names = [roi_labels[i] for i in features if i < len(roi_labels)]
        
        network_summary.append({
            'network': network,
            'score': score,
            'n_features': len(features),
            'feature_indices': ','.join(map(str, features)),
            'roi_names': '; '.join(roi_names)
        })
    
    # Create DataFrame and sort by score
    summary_df = pd.DataFrame(network_summary)
    summary_df = summary_df.sort_values('score', ascending=False)
    summary_df['network_rank'] = range(1, len(summary_df) + 1)
    
    # Save table
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    print(f"Network summary table saved to: {output_path}")
    
    return summary_df


def create_comprehensive_region_table(consensus_data: Dict[str, Dict],
                                    roi_labels_path: str,
                                    output_dir: str = "results/tables") -> None:
    """
    Create comprehensive region tables for all datasets.
    
    Args:
        consensus_data (Dict[str, Dict]): Consensus data for different datasets
        roi_labels_path (str): Path to ROI labels file
        output_dir (str): Output directory for tables
    """
    # Load ROI labels
    roi_labels = load_roi_labels(roi_labels_path)
    
    # Create individual tables for each dataset
    for dataset_name, data in consensus_data.items():
        print(f"Creating tables for {dataset_name}...")
        
        if 'consensus_counts' not in data:
            print(f"Warning: No consensus counts found for {dataset_name}")
            continue
        
        # Create consensus regions table
        consensus_table = create_consensus_region_table(
            data['consensus_counts'],
            roi_labels,
            data.get('network_mapping'),
            os.path.join(output_dir, f"{dataset_name}_consensus_regions.csv")
        )
        
        # Create network summary table if network mapping available
        if 'network_mapping' in data and 'network_scores' in data:
            network_table = create_network_summary_table(
                data['network_scores'],
                data['network_mapping'],
                roi_labels,
                os.path.join(output_dir, f"{dataset_name}_network_summary.csv")
            )
    
    # Create cross-dataset comparison table
    if len(consensus_data) > 1:
        print("Creating cross-dataset comparison table...")
        
        # Extract feature lists for each dataset
        cohort_features = {}
        for dataset_name, data in consensus_data.items():
            if 'consensus_counts' in data:
                # Get top features (e.g., top 20%)
                top_features = extract_top_features_by_consensus(
                    data['consensus_counts'], top_percentage=0.2
                )
                cohort_features[dataset_name] = top_features
        
        if cohort_features:
            overlap_table = create_overlap_analysis_table(
                cohort_features,
                list(cohort_features.keys()),
                roi_labels,
                os.path.join(output_dir, "cross_dataset_overlap.csv")
            )


def main():
    """Main function for creating region tables."""
    parser = argparse.ArgumentParser(
        description="Create tables for regions of importance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create tables for a single dataset
  python create_region_tables.py \\
    --consensus_file results/consensus_data.npz \\
    --roi_labels /path/to/roi_labels.txt \\
    --output_dir results/tables

  # Create comprehensive tables for multiple datasets
  python create_region_tables.py \\
    --consensus_dir results/consensus/ \\
    --roi_labels /path/to/roi_labels.txt \\
    --output_dir results/tables
        """
    )
    
    parser.add_argument("--consensus_file", type=str,
                       help="Path to single consensus data file")
    parser.add_argument("--consensus_dir", type=str,
                       help="Directory containing multiple consensus data files")
    parser.add_argument("--roi_labels", type=str, required=True,
                       help="Path to ROI labels file")
    parser.add_argument("--output_dir", type=str, default="results/tables",
                       help="Output directory for tables")
    parser.add_argument("--dataset_name", type=str,
                       help="Name of dataset (for single file mode)")
    
    args = parser.parse_args()
    
    if not args.consensus_file and not args.consensus_dir:
        print("Error: Must specify either --consensus_file or --consensus_dir")
        sys.exit(1)
    
    if args.consensus_file:
        # Single file mode
        if not args.dataset_name:
            args.dataset_name = "dataset"
        
        try:
            # Load consensus data
            consensus_data = np.load(args.consensus_file, allow_pickle=True)
            consensus_counts = consensus_data['consensus_counts'].item()
            network_mapping = consensus_data.get('network_mapping', {}).item()
            
            # Create single dataset tables
            roi_labels = load_roi_labels(args.roi_labels)
            
            create_consensus_region_table(
                consensus_counts,
                roi_labels,
                network_mapping,
                os.path.join(args.output_dir, f"{args.dataset_name}_consensus_regions.csv")
            )
            
            if network_mapping:
                network_scores = consensus_data.get('network_scores', {}).item()
                create_network_summary_table(
                    network_scores,
                    network_mapping,
                    roi_labels,
                    os.path.join(args.output_dir, f"{args.dataset_name}_network_summary.csv")
                )
            
            print(f"Tables created successfully for {args.dataset_name}")
            
        except Exception as e:
            print(f"Error processing single file: {e}")
            sys.exit(1)
    
    else:
        # Multiple files mode
        try:
            # Load consensus data from multiple files
            consensus_data = {}
            
            for file_path in Path(args.consensus_dir).glob("*.npz"):
                dataset_name = file_path.stem
                
                try:
                    data = np.load(file_path, allow_pickle=True)
                    consensus_data[dataset_name] = {
                        'consensus_counts': data['consensus_counts'].item(),
                        'network_mapping': data.get('network_mapping', {}).item(),
                        'network_scores': data.get('network_scores', {}).item()
                    }
                except Exception as e:
                    print(f"Warning: Could not load {file_path}: {e}")
            
            if not consensus_data:
                print("Error: No valid consensus data files found")
                sys.exit(1)
            
            # Create comprehensive tables
            create_comprehensive_region_table(
                consensus_data,
                args.roi_labels,
                args.output_dir
            )
            
            print(f"Comprehensive tables created for {len(consensus_data)} datasets")
            
        except Exception as e:
            print(f"Error processing multiple files: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
