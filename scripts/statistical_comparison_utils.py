#!/usr/bin/env python3
"""
Statistical comparison utilities for brain feature maps.

Implements robust statistical tests for comparing feature importance across datasets:
- Cosine similarity with permutation-based p-values
- Spearman correlation on ROI ranks
- ROI-wise two-proportion tests with FDR correction
- Network-level aggregation
- CLR (Aitchison) distance for compositional data
- Jensen-Shannon divergence

Author: Enhanced Statistical Analysis
Date: 2024
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr
from statsmodels.stats.multitest import multipletests
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# GLOBAL SIMILARITY METRICS
# ============================================================================

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Returns: similarity in [0, 1] (after normalization from [-1, 1])
    """
    vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-10)
    vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-10)
    
    cos_sim = np.dot(vec1_norm, vec2_norm)
    
    # Normalize to [0, 1] range
    return (cos_sim + 1) / 2


def cosine_similarity_with_pvalue(
    data1: np.ndarray,
    data2: np.ndarray,
    n_permutations: int = 10000,
    top_half: bool = True
) -> Tuple[float, float]:
    """
    Compute cosine similarity with permutation-based p-value.
    
    Args:
        data1: First dataset (n_subjects × n_rois)
        data2: Second dataset (n_subjects × n_rois)
        n_permutations: Number of permutations for null distribution
        top_half: If True, create top-half indicator matrices
    
    Returns:
        (cosine_similarity, p_value)
    """
    # Convert to top-half indicators if requested
    if top_half:
        # For each subject, mark ROIs in top 50%
        vec1 = (data1 > np.median(data1, axis=1, keepdims=True)).mean(axis=0)
        vec2 = (data2 > np.median(data2, axis=1, keepdims=True)).mean(axis=0)
    else:
        # Use mean across subjects
        vec1 = data1.mean(axis=0)
        vec2 = data2.mean(axis=0)
    
    # Observed cosine similarity
    observed_cos = cosine_similarity(vec1, vec2)
    
    # Pool data for permutation
    pooled = np.vstack([data1, data2])
    n1 = len(data1)
    n_total = len(pooled)
    
    # Generate null distribution
    null_distribution = []
    
    for _ in range(n_permutations):
        # Randomly partition pooled data
        perm_idx = np.random.permutation(n_total)
        perm_data1 = pooled[perm_idx[:n1]]
        perm_data2 = pooled[perm_idx[n1:]]
        
        # Compute indicator vectors for permuted data
        if top_half:
            perm_vec1 = (perm_data1 > np.median(perm_data1, axis=1, keepdims=True)).mean(axis=0)
            perm_vec2 = (perm_data2 > np.median(perm_data2, axis=1, keepdims=True)).mean(axis=0)
        else:
            perm_vec1 = perm_data1.mean(axis=0)
            perm_vec2 = perm_data2.mean(axis=0)
        
        # Compute null cosine
        null_cos = cosine_similarity(perm_vec1, perm_vec2)
        null_distribution.append(null_cos)
    
    # Compute p-value (one-tailed: observed > null)
    null_distribution = np.array(null_distribution)
    p_value = (null_distribution >= observed_cos).sum() / n_permutations
    
    return observed_cos, p_value


def spearman_on_ranks(
    data1: np.ndarray,
    data2: np.ndarray
) -> Tuple[float, float]:
    """
    Compute Spearman correlation on ROI ranks.
    
    Args:
        data1: First dataset (n_subjects × n_rois)
        data2: Second dataset (n_subjects × n_rois)
    
    Returns:
        (spearman_rho, p_value)
    """
    # Mean across subjects
    mean1 = data1.mean(axis=0)
    mean2 = data2.mean(axis=0)
    
    # Compute Spearman correlation
    rho, p_value = stats.spearmanr(mean1, mean2)
    
    return rho, p_value


# ============================================================================
# COMPOSITIONAL DATA ANALYSIS
# ============================================================================

def clr_transform(data: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Centered Log-Ratio (CLR) transformation for compositional data.
    
    CLR(x) = log(x / geometric_mean(x))
    
    Args:
        data: Vector of positive values (will be normalized to sum to 1)
        epsilon: Small constant to avoid log(0)
    
    Returns:
        CLR-transformed data
    """
    # Ensure positive values
    data = np.abs(data) + epsilon
    
    # Normalize to sum to 1 (compositional)
    composition = data / data.sum()
    
    # Geometric mean
    geom_mean = np.exp(np.log(composition).mean())
    
    # CLR transform
    clr = np.log(composition / geom_mean)
    
    return clr


def aitchison_distance(
    data1: np.ndarray,
    data2: np.ndarray
) -> float:
    """
    Compute Aitchison (CLR) distance between two compositional datasets.
    
    Aitchison distance = Euclidean distance in CLR space
    
    Args:
        data1: First dataset (n_subjects × n_rois)
        data2: Second dataset (n_subjects × n_rois)
    
    Returns:
        Aitchison distance
    """
    # Mean across subjects
    mean1 = data1.mean(axis=0)
    mean2 = data2.mean(axis=0)
    
    # CLR transform
    clr1 = clr_transform(mean1)
    clr2 = clr_transform(mean2)
    
    # Euclidean distance in CLR space
    distance = np.linalg.norm(clr1 - clr2)
    
    return distance


def jensen_shannon_divergence(
    data1: np.ndarray,
    data2: np.ndarray
) -> float:
    """
    Compute Jensen-Shannon divergence between normalized ROI distributions.
    
    JSD is a symmetric measure of divergence between probability distributions.
    Range: [0, 1] (using base 2 for normalization)
    
    Args:
        data1: First dataset (n_subjects × n_rois)
        data2: Second dataset (n_subjects × n_rois)
    
    Returns:
        Jensen-Shannon divergence
    """
    # Mean across subjects
    mean1 = data1.mean(axis=0)
    mean2 = data2.mean(axis=0)
    
    # Normalize to probability distributions
    p1 = mean1 / mean1.sum()
    p2 = mean2 / mean2.sum()
    
    # Compute JS divergence (scipy uses base 2 by default)
    jsd = jensenshannon(p1, p2)
    
    return jsd


# ============================================================================
# LOCAL (ROI-WISE) TESTS
# ============================================================================

def roi_wise_proportion_test(
    data1: np.ndarray,
    data2: np.ndarray,
    percentile_threshold: float = 50.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ROI-wise two-proportion z-tests with FDR correction.
    
    Tests if the proportion of subjects with high values (>threshold) differs
    between datasets for each ROI.
    
    Args:
        data1: First dataset (n_subjects × n_rois)
        data2: Second dataset (n_subjects × n_rois)
        percentile_threshold: Percentile to define "high" values
    
    Returns:
        (z_scores, p_values, fdr_corrected_p_values)
    """
    n1, n_rois = data1.shape
    n2 = data2.shape[0]
    
    # Determine thresholds for each ROI (pooled)
    pooled = np.vstack([data1, data2])
    thresholds = np.percentile(pooled, percentile_threshold, axis=0)
    
    # Count subjects above threshold for each ROI
    count1 = (data1 > thresholds).sum(axis=0)
    count2 = (data2 > thresholds).sum(axis=0)
    
    # Proportions
    p1 = count1 / n1
    p2 = count2 / n2
    
    # Pooled proportion
    p_pooled = (count1 + count2) / (n1 + n2)
    
    # Two-proportion z-test for each ROI
    z_scores = np.zeros(n_rois)
    p_values = np.zeros(n_rois)
    
    for i in range(n_rois):
        # Standard error under null hypothesis
        se = np.sqrt(p_pooled[i] * (1 - p_pooled[i]) * (1/n1 + 1/n2))
        
        if se > 0:
            z_scores[i] = (p1[i] - p2[i]) / se
            # Two-tailed test
            p_values[i] = 2 * (1 - stats.norm.cdf(np.abs(z_scores[i])))
        else:
            z_scores[i] = 0
            p_values[i] = 1.0
    
    # FDR correction (Benjamini-Hochberg)
    _, fdr_p_values, _, _ = multipletests(p_values, method='fdr_bh')
    
    return z_scores, p_values, fdr_p_values


def network_level_aggregation(
    z_scores: np.ndarray,
    p_values: np.ndarray,
    roi_to_network: Dict[int, str]
) -> pd.DataFrame:
    """
    Aggregate ROI-wise statistics to network level.
    
    Args:
        z_scores: ROI-wise z-scores
        p_values: ROI-wise p-values (FDR-corrected)
        roi_to_network: Mapping from ROI index to network name
    
    Returns:
        DataFrame with network-level summaries
    """
    # Create DataFrame with ROI results
    results = []
    
    for roi_idx in range(len(z_scores)):
        network = roi_to_network.get(roi_idx, 'Unknown')
        results.append({
            'ROI': roi_idx,
            'Network': network,
            'Z_Score': z_scores[roi_idx],
            'P_Value': p_values[roi_idx],
            'Significant': p_values[roi_idx] < 0.05
        })
    
    df = pd.DataFrame(results)
    
    # Aggregate by network
    network_summary = df.groupby('Network').agg({
        'ROI': 'count',
        'Z_Score': ['mean', 'std', 'min', 'max'],
        'Significant': ['sum', 'mean']
    }).reset_index()
    
    # Flatten column names
    network_summary.columns = [
        'Network', 'N_ROIs', 
        'Mean_Z', 'Std_Z', 'Min_Z', 'Max_Z',
        'N_Significant', 'Proportion_Significant'
    ]
    
    return network_summary


# ============================================================================
# COMPREHENSIVE COMPARISON
# ============================================================================

def compare_datasets(
    data1: np.ndarray,
    data2: np.ndarray,
    name1: str,
    name2: str,
    roi_to_network: Optional[Dict[int, str]] = None,
    n_permutations: int = 10000,
    verbose: bool = True
) -> Dict:
    """
    Comprehensive statistical comparison of two datasets.
    
    Args:
        data1: First dataset (n_subjects × n_rois)
        data2: Second dataset (n_subjects × n_rois)
        name1: Name of first dataset
        name2: Name of second dataset
        roi_to_network: Optional mapping from ROI index to network name
        n_permutations: Number of permutations for cosine p-value
        verbose: Print results
    
    Returns:
        Dictionary with all comparison metrics
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"STATISTICAL COMPARISON: {name1} vs {name2}")
        print(f"{'='*80}")
        print(f"\nData shapes: {data1.shape} vs {data2.shape}")
    
    results = {
        'dataset1': name1,
        'dataset2': name2,
        'n1': len(data1),
        'n2': len(data2),
        'n_rois': data1.shape[1]
    }
    
    # ========================================================================
    # GLOBAL METRICS
    # ========================================================================
    
    if verbose:
        print(f"\n{'─'*80}")
        print("GLOBAL SIMILARITY METRICS")
        print(f"{'─'*80}")
    
    # 1. Cosine similarity with permutation p-value
    if verbose:
        print(f"\n[1/6] Computing cosine similarity (top-half indicators)...")
    cos_sim, cos_p = cosine_similarity_with_pvalue(
        data1, data2, n_permutations=n_permutations, top_half=True
    )
    results['cosine_similarity'] = cos_sim
    results['cosine_p_value'] = cos_p
    
    if verbose:
        print(f"  ✓ Cosine similarity: {cos_sim:.4f} (p = {cos_p:.4f})")
    
    # 2. Spearman on ROI ranks
    if verbose:
        print(f"\n[2/6] Computing Spearman correlation on ROI ranks...")
    spearman_rho, spearman_p = spearman_on_ranks(data1, data2)
    results['spearman_rho'] = spearman_rho
    results['spearman_p_value'] = spearman_p
    
    if verbose:
        print(f"  ✓ Spearman ρ: {spearman_rho:.4f} (p = {spearman_p:.4f})")
    
    # 3. Aitchison (CLR) distance
    if verbose:
        print(f"\n[3/6] Computing Aitchison (CLR) distance...")
    aitchison_dist = aitchison_distance(data1, data2)
    results['aitchison_distance'] = aitchison_dist
    
    if verbose:
        print(f"  ✓ Aitchison distance: {aitchison_dist:.4f}")
    
    # 4. Jensen-Shannon divergence
    if verbose:
        print(f"\n[4/6] Computing Jensen-Shannon divergence...")
    jsd = jensen_shannon_divergence(data1, data2)
    results['jensen_shannon_divergence'] = jsd
    
    if verbose:
        print(f"  ✓ Jensen-Shannon divergence: {jsd:.4f}")
    
    # ========================================================================
    # LOCAL (ROI-WISE) TESTS
    # ========================================================================
    
    if verbose:
        print(f"\n{'─'*80}")
        print("LOCAL (ROI-WISE) TESTS")
        print(f"{'─'*80}")
        print(f"\n[5/6] Running ROI-wise two-proportion tests with FDR correction...")
    
    z_scores, p_values, fdr_p_values = roi_wise_proportion_test(data1, data2)
    
    results['roi_z_scores'] = z_scores
    results['roi_p_values'] = p_values
    results['roi_fdr_p_values'] = fdr_p_values
    results['n_significant_rois'] = (fdr_p_values < 0.05).sum()
    
    if verbose:
        print(f"  ✓ Significant ROIs (FDR < 0.05): {results['n_significant_rois']}/{len(fdr_p_values)}")
        print(f"  ✓ Mean |Z|: {np.abs(z_scores).mean():.3f}")
        print(f"  ✓ Max |Z|: {np.abs(z_scores).max():.3f}")
    
    # ========================================================================
    # NETWORK-LEVEL AGGREGATION
    # ========================================================================
    
    if roi_to_network is not None:
        if verbose:
            print(f"\n[6/6] Aggregating to network level...")
        
        network_summary = network_level_aggregation(
            z_scores, fdr_p_values, roi_to_network
        )
        results['network_summary'] = network_summary
        
        if verbose:
            print(f"\n  Networks with significant differences:")
            sig_networks = network_summary[network_summary['N_Significant'] > 0]
            if len(sig_networks) > 0:
                for _, row in sig_networks.iterrows():
                    print(f"    • {row['Network']}: {int(row['N_Significant'])}/{int(row['N_ROIs'])} ROIs "
                          f"({100*row['Proportion_Significant']:.1f}%), Mean Z = {row['Mean_Z']:.2f}")
            else:
                print(f"    (None)")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    if verbose:
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"\nGlobal Agreement:")
        print(f"  • Cosine similarity:      {cos_sim:.4f} (p = {cos_p:.4f})")
        print(f"  • Spearman ρ (ranks):     {spearman_rho:.4f} (p = {spearman_p:.4f})")
        print(f"\nDistribution Measures:")
        print(f"  • Aitchison distance:     {aitchison_dist:.4f} (lower = more similar)")
        print(f"  • Jensen-Shannon div:     {jsd:.4f} (lower = more similar)")
        print(f"\nLocal Differences:")
        print(f"  • Significant ROIs:       {results['n_significant_rois']}/{len(fdr_p_values)}")
        print(f"  • Mean |Z-score|:         {np.abs(z_scores).mean():.3f}")
        print(f"{'='*80}\n")
    
    return results


def create_comparison_report(
    results: Dict,
    output_path: str
) -> None:
    """
    Create a detailed comparison report.
    
    Args:
        results: Output from compare_datasets()
        output_path: Path to save report CSV
    """
    # Summary metrics
    summary_df = pd.DataFrame([{
        'Dataset_1': results['dataset1'],
        'Dataset_2': results['dataset2'],
        'N_Subjects_1': results['n1'],
        'N_Subjects_2': results['n2'],
        'N_ROIs': results['n_rois'],
        'Cosine_Similarity': results['cosine_similarity'],
        'Cosine_P_Value': results['cosine_p_value'],
        'Spearman_Rho': results['spearman_rho'],
        'Spearman_P_Value': results['spearman_p_value'],
        'Aitchison_Distance': results['aitchison_distance'],
        'Jensen_Shannon_Divergence': results['jensen_shannon_divergence'],
        'N_Significant_ROIs': results['n_significant_rois']
    }])
    
    summary_df.to_csv(output_path, index=False)
    print(f"✓ Saved comparison report: {output_path}")

