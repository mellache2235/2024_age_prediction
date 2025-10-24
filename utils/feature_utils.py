"""
Feature attribution and analysis utilities for age prediction.

This module provides functions for computing feature attributions, consensus analysis,
and network-level feature analysis using brain atlases.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
import os
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def compute_feature_attributions(attr_data: np.ndarray, 
                               labels: np.ndarray,
                               percentile: float = 95.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute feature attributions and select top features based on percentile.
    
    Args:
        attr_data (np.ndarray): Attribution data (subjects x features x timepoints)
        labels (np.ndarray): Labels for grouping
        percentile (float): Percentile threshold for feature selection
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Feature indices and feature scores
    """
    # Compute median across timepoints
    medians = np.median(attr_data, axis=2)
    
    # Compute mean absolute attributions across subjects
    mean_attr = np.mean(np.abs(medians), axis=0)
    
    # Find features above percentile threshold
    threshold = np.percentile(np.abs(mean_attr), percentile)
    feature_indices = np.where(np.abs(mean_attr) >= threshold)[0]
    
    return feature_indices, mean_attr


def compute_consensus_features_across_models(ig_dir: str,
                                           model_prefix: str,
                                           num_models: int,
                                           percentile: float = 95.0,
                                           group_filter: Optional[np.ndarray] = None) -> Dict[int, int]:
    """
    Compute consensus features across multiple models.
    
    Args:
        ig_dir (str): Directory containing integrated gradients files
        model_prefix (str): Prefix for model files
        num_models (int): Number of models to analyze
        percentile (float): Percentile threshold for feature selection
        group_filter (np.ndarray, optional): Boolean array to filter subjects
        
    Returns:
        Dict[int, int]: Dictionary mapping feature indices to consensus counts
    """
    all_feature_indices = []
    
    for k in range(num_models):
        # Load integrated gradients data
        ig_file = os.path.join(ig_dir, f"{model_prefix}_{k}_ig.npz")
        if not os.path.exists(ig_file):
            print(f"Warning: File {ig_file} not found, skipping...")
            continue
            
        ig_data = np.load(ig_file, allow_pickle=True)
        attr_data = ig_data["arr_0"]  # subjects x features x timepoints
        
        # Apply group filter if provided
        if group_filter is not None:
            attr_data = attr_data[group_filter]
        
        # Compute feature attributions
        feature_indices, _ = compute_feature_attributions(attr_data, None, percentile)
        all_feature_indices.extend(feature_indices)
    
    # Count consensus
    feature_counts = Counter(all_feature_indices)
    return dict(feature_counts)


def map_features_to_networks(feature_indices: List[int],
                           yeo_atlas_path: str) -> Dict[str, List[int]]:
    """
    Map feature indices to Yeo network assignments.
    
    Args:
        feature_indices (List[int]): List of feature indices
        yeo_atlas_path (str): Path to Yeo atlas CSV file
        
    Returns:
        Dict[str, List[int]]: Dictionary mapping network names to feature indices
    """
    # Load Yeo atlas
    yeo_atlas = pd.read_csv(yeo_atlas_path)
    
    # Create mapping
    network_mapping = {}
    for idx in feature_indices:
        if idx < len(yeo_atlas):
            network = yeo_atlas.iloc[idx]['Yeo_17network']
            if network not in network_mapping:
                network_mapping[network] = []
            network_mapping[network].append(idx)
    
    return network_mapping


def compute_network_level_scores(feature_scores: np.ndarray,
                               network_mapping: Dict[str, List[int]],
                               aggregation_method: str = 'mean') -> Dict[str, float]:
    """
    Compute network-level scores from feature-level scores.
    
    Args:
        feature_scores (np.ndarray): Feature-level scores
        network_mapping (Dict[str, List[int]]): Network to feature mapping
        aggregation_method (str): Method for aggregating features ('mean', 'sum', 'max')
        
    Returns:
        Dict[str, float]: Network-level scores
    """
    network_scores = {}
    
    for network, feature_indices in network_mapping.items():
        if not feature_indices:
            continue
            
        # Get scores for features in this network
        network_feature_scores = feature_scores[feature_indices]
        
        # Aggregate scores
        if aggregation_method == 'mean':
            network_scores[network] = np.mean(network_feature_scores)
        elif aggregation_method == 'sum':
            network_scores[network] = np.sum(network_feature_scores)
        elif aggregation_method == 'max':
            network_scores[network] = np.max(network_feature_scores)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    
    return network_scores


def create_feature_consensus_nifti(consensus_counts: Dict[int, int],
                                 atlas_nifti_path: str,
                                 output_path: str,
                                 normalize: bool = True) -> None:
    """
    Create NIfTI file with consensus feature counts.
    
    Args:
        consensus_counts (Dict[int, int]): Feature consensus counts
        atlas_nifti_path (str): Path to atlas NIfTI file
        output_path (str): Output path for consensus NIfTI
        normalize (bool): Whether to normalize counts
    """
    try:
        from nilearn import image
    except ImportError:
        print("Warning: nilearn not available, skipping NIfTI creation")
        return
    
    # Load atlas
    atlas_volume = image.load_img(atlas_nifti_path)
    img_data = atlas_volume.get_fdata()
    
    # Create consensus map
    consensus_map = np.zeros_like(img_data)
    max_count = max(consensus_counts.values()) if consensus_counts else 1
    
    for roi_idx, count in consensus_counts.items():
        if normalize:
            normalized_count = count / max_count
        else:
            normalized_count = count
        
        # Find voxels belonging to this ROI
        roi_mask = img_data == (roi_idx + 1)  # ROIs are 1-indexed
        consensus_map[roi_mask] = normalized_count
    
    # Save consensus map
    consensus_img = image.new_img_like(atlas_volume, consensus_map)
    consensus_img.to_filename(output_path)
    print(f"Consensus NIfTI saved to: {output_path}")


def analyze_feature_overlap(features_a: List[int], features_b: List[int]) -> Dict[str, Any]:
    """
    Analyze overlap between two sets of features.
    
    Args:
        features_a (List[int]): First set of features
        features_b (List[int]): Second set of features
        
    Returns:
        Dict[str, Any]: Overlap analysis results
    """
    set_a = set(features_a)
    set_b = set(features_b)
    
    intersection = set_a & set_b
    union = set_a | set_b
    
    # Compute similarity metrics
    jaccard = len(intersection) / len(union) if union else 0.0
    dice = (2 * len(intersection)) / (len(set_a) + len(set_b)) if (len(set_a) + len(set_b)) > 0 else 0.0
    
    return {
        'features_a': list(set_a),
        'features_b': list(set_b),
        'intersection': list(intersection),
        'union': list(union),
        'jaccard_index': jaccard,
        'dice_coefficient': dice,
        'intersection_size': len(intersection),
        'union_size': len(union)
    }


def rank_features_by_importance(feature_scores: np.ndarray,
                              feature_names: Optional[List[str]] = None,
                              top_n: Optional[int] = None) -> pd.DataFrame:
    """
    Rank features by importance scores.
    
    Args:
        feature_scores (np.ndarray): Feature importance scores
        feature_names (List[str], optional): Names of features
        top_n (int, optional): Number of top features to return
        
    Returns:
        pd.DataFrame: Ranked features
    """
    # Sort features by importance
    sorted_indices = np.argsort(feature_scores)[::-1]
    
    if top_n is not None:
        sorted_indices = sorted_indices[:top_n]
    
    # Create results DataFrame
    results = pd.DataFrame({
        'feature_index': sorted_indices,
        'importance_score': feature_scores[sorted_indices],
        'rank': range(1, len(sorted_indices) + 1)
    })
    
    if feature_names is not None:
        results['feature_name'] = [feature_names[i] for i in sorted_indices]
    
    return results


def compute_feature_stability(feature_lists: List[List[int]]) -> Dict[int, float]:
    """
    Compute stability of features across multiple runs.
    
    Args:
        feature_lists (List[List[int]]): List of feature lists from different runs
        
    Returns:
        Dict[int, float]: Dictionary mapping feature indices to stability scores
    """
    # Count occurrences of each feature
    feature_counts = Counter()
    for feature_list in feature_lists:
        for feature in feature_list:
            feature_counts[feature] += 1
    
    # Compute stability scores
    n_runs = len(feature_lists)
    stability_scores = {}
    
    for feature, count in feature_counts.items():
        stability_scores[feature] = count / n_runs
    
    return stability_scores


def create_feature_summary_table(consensus_counts: Dict[int, int],
                               feature_names: List[str],
                               network_mapping: Optional[Dict[str, List[int]]] = None,
                               output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Create a summary table of consensus features.
    
    Args:
        consensus_counts (Dict[int, int]): Feature consensus counts
        feature_names (List[str]): Names of features
        network_mapping (Dict[str, List[int]], optional): Network assignments
        output_path (str, optional): Path to save the table
        
    Returns:
        pd.DataFrame: Summary table
    """
    # Create summary data
    summary_data = []
    
    for feature_idx, count in consensus_counts.items():
        if feature_idx < len(feature_names):
            row = {
                'feature_index': feature_idx,
                'feature_name': feature_names[feature_idx],
                'consensus_count': count,
                'consensus_percentage': count / max(consensus_counts.values()) * 100
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
    
    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('consensus_count', ascending=False)
    
    # Save if path provided
    if output_path:
        summary_df.to_csv(output_path, index=False)
        print(f"Feature summary table saved to: {output_path}")
    
    return summary_df


def compare_feature_sets_across_cohorts(cohort_features: Dict[str, List[int]],
                                      cohort_names: List[str]) -> pd.DataFrame:
    """
    Compare feature sets across multiple cohorts.
    
    Args:
        cohort_features (Dict[str, List[int]]): Dictionary mapping cohort names to feature lists
        cohort_names (List[str]): Names of cohorts to compare
        
    Returns:
        pd.DataFrame: Comparison results
    """
    comparison_results = []
    
    for i, cohort_a in enumerate(cohort_names):
        for j, cohort_b in enumerate(cohort_names):
            if i >= j:  # Avoid duplicate comparisons
                continue
            
            features_a = set(cohort_features[cohort_a])
            features_b = set(cohort_features[cohort_b])
            
            # Compute overlap metrics
            intersection = features_a & features_b
            union = features_a | features_b
            
            jaccard = len(intersection) / len(union) if union else 0.0
            dice = (2 * len(intersection)) / (len(features_a) + len(features_b)) if (len(features_a) + len(features_b)) > 0 else 0.0
            
            comparison_results.append({
                'cohort_a': cohort_a,
                'cohort_b': cohort_b,
                'features_a_count': len(features_a),
                'features_b_count': len(features_b),
                'intersection_count': len(intersection),
                'union_count': len(union),
                'jaccard_index': jaccard,
                'dice_coefficient': dice
            })
    
    return pd.DataFrame(comparison_results)


def extract_top_features_by_consensus(consensus_counts: Dict[int, int],
                                    top_percentage: float = 0.1) -> List[int]:
    """
    Extract top features based on consensus counts.
    
    Args:
        consensus_counts (Dict[int, int]): Feature consensus counts
        top_percentage (float): Percentage of top features to extract
        
    Returns:
        List[int]: List of top feature indices
    """
    if not consensus_counts:
        return []
    
    # Sort features by consensus count
    sorted_features = sorted(consensus_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Select top percentage
    n_top = max(1, int(len(sorted_features) * top_percentage))
    top_features = [feature for feature, count in sorted_features[:n_top]]
    
    return top_features
