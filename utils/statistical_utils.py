"""
Statistical analysis utilities for age prediction analysis.

This module provides functions for statistical testing, multiple comparison correction,
and other statistical analyses commonly used in neuroimaging research.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def benjamini_hochberg_correction(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Benjamini-Hochberg FDR correction to p-values.
    
    Args:
        p_values (np.ndarray): Array of p-values
        alpha (float): Significance level (default: 0.05)
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            corrected_p_values, rejected_hypotheses, p_value_corrected, alpha_sidak
    """
    return multipletests(p_values, alpha=alpha, method='fdr_bh')


def bonferroni_correction(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Bonferroni correction to p-values.
    
    Args:
        p_values (np.ndarray): Array of p-values
        alpha (float): Significance level (default: 0.05)
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            corrected_p_values, rejected_hypotheses, p_value_corrected, alpha_sidak
    """
    return multipletests(p_values, alpha=alpha, method='bonferroni')


def holm_correction(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Holm correction to p-values.
    
    Args:
        p_values (np.ndarray): Array of p-values
        alpha (float): Significance level (default: 0.05)
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            corrected_p_values, rejected_hypotheses, p_value_corrected, alpha_sidak
    """
    return multipletests(p_values, alpha=alpha, method='holm')


def apply_multiple_comparison_correction(p_values: np.ndarray, 
                                       method: str = 'fdr_bh',
                                       alpha: float = 0.05) -> Dict[str, np.ndarray]:
    """
    Apply multiple comparison correction to p-values.
    
    Args:
        p_values (np.ndarray): Array of p-values
        method (str): Correction method ('fdr_bh', 'bonferroni', 'holm')
        alpha (float): Significance level
        
    Returns:
        Dict[str, np.ndarray]: Dictionary containing corrected p-values and significance
    """
    if method == 'fdr_bh':
        corrected_p, rejected, _, _ = benjamini_hochberg_correction(p_values, alpha)
    elif method == 'bonferroni':
        corrected_p, rejected, _, _ = bonferroni_correction(p_values, alpha)
    elif method == 'holm':
        corrected_p, rejected, _, _ = holm_correction(p_values, alpha)
    else:
        raise ValueError(f"Unknown correction method: {method}")
    
    return {
        'corrected_p_values': corrected_p,
        'significant': rejected,
        'original_p_values': p_values
    }


def compute_effect_size(group1: np.ndarray, group2: np.ndarray, 
                       effect_type: str = 'cohens_d') -> float:
    """
    Compute effect size between two groups.
    
    Args:
        group1 (np.ndarray): First group data
        group2 (np.ndarray): Second group data
        effect_type (str): Type of effect size ('cohens_d', 'hedges_g')
        
    Returns:
        float: Effect size
    """
    if effect_type == 'cohens_d':
        # Cohen's d
        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                             (len(group2) - 1) * np.var(group2, ddof=1)) / 
                            (len(group1) + len(group2) - 2))
        effect_size = (np.mean(group1) - np.mean(group2)) / pooled_std
    elif effect_type == 'hedges_g':
        # Hedges' g (bias-corrected Cohen's d)
        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                             (len(group2) - 1) * np.var(group2, ddof=1)) / 
                            (len(group1) + len(group2) - 2))
        correction_factor = 1 - (3 / (4 * (len(group1) + len(group2)) - 9))
        effect_size = ((np.mean(group1) - np.mean(group2)) / pooled_std) * correction_factor
    else:
        raise ValueError(f"Unknown effect size type: {effect_type}")
    
    return effect_size


def bootstrap_confidence_interval(data: np.ndarray, 
                                statistic_func: callable,
                                n_bootstrap: int = 1000,
                                confidence_level: float = 0.95,
                                random_state: Optional[int] = None) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for a statistic.
    
    Args:
        data (np.ndarray): Input data
        statistic_func (callable): Function to compute statistic
        n_bootstrap (int): Number of bootstrap samples
        confidence_level (float): Confidence level (e.g., 0.95 for 95% CI)
        random_state (int, optional): Random seed for reproducibility
        
    Returns:
        Tuple[float, float]: Lower and upper confidence interval bounds
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(statistic_func(bootstrap_sample))
    
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_stats, lower_percentile)
    ci_upper = np.percentile(bootstrap_stats, upper_percentile)
    
    return ci_lower, ci_upper


def permutation_test(group1: np.ndarray, group2: np.ndarray,
                    n_permutations: int = 10000,
                    test_statistic: str = 'mean_diff',
                    random_state: Optional[int] = None) -> Tuple[float, float]:
    """
    Perform permutation test between two groups.
    
    Args:
        group1 (np.ndarray): First group data
        group2 (np.ndarray): Second group data
        n_permutations (int): Number of permutations
        test_statistic (str): Test statistic ('mean_diff', 't_statistic')
        random_state (int, optional): Random seed for reproducibility
        
    Returns:
        Tuple[float, float]: Test statistic and p-value
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Combine data
    combined_data = np.concatenate([group1, group2])
    n1, n2 = len(group1), len(group2)
    
    # Compute observed test statistic
    if test_statistic == 'mean_diff':
        observed_stat = np.mean(group1) - np.mean(group2)
    elif test_statistic == 't_statistic':
        observed_stat, _ = stats.ttest_ind(group1, group2)
    else:
        raise ValueError(f"Unknown test statistic: {test_statistic}")
    
    # Permutation test
    permuted_stats = []
    for _ in range(n_permutations):
        # Shuffle combined data
        shuffled_data = np.random.permutation(combined_data)
        perm_group1 = shuffled_data[:n1]
        perm_group2 = shuffled_data[n1:]
        
        # Compute test statistic for permuted data
        if test_statistic == 'mean_diff':
            perm_stat = np.mean(perm_group1) - np.mean(perm_group2)
        elif test_statistic == 't_statistic':
            perm_stat, _ = stats.ttest_ind(perm_group1, perm_group2)
        
        permuted_stats.append(perm_stat)
    
    # Compute p-value
    if test_statistic == 'mean_diff':
        p_value = np.mean(np.abs(permuted_stats) >= np.abs(observed_stat))
    else:  # t_statistic
        p_value = np.mean(np.abs(permuted_stats) >= np.abs(observed_stat))
    
    return observed_stat, p_value


def correlation_analysis(x: np.ndarray, y: np.ndarray,
                        method: str = 'pearson',
                        confidence_level: float = 0.95) -> Dict[str, float]:
    """
    Perform correlation analysis between two variables.
    
    Args:
        x (np.ndarray): First variable
        y (np.ndarray): Second variable
        method (str): Correlation method ('pearson', 'spearman', 'kendall')
        confidence_level (float): Confidence level for confidence interval
        
    Returns:
        Dict[str, float]: Dictionary containing correlation results
    """
    if method == 'pearson':
        corr, p_value = stats.pearsonr(x, y)
    elif method == 'spearman':
        corr, p_value = stats.spearmanr(x, y)
    elif method == 'kendall':
        corr, p_value = stats.kendalltau(x, y)
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    
    # Compute confidence interval using Fisher's z-transformation
    if method == 'pearson':
        n = len(x)
        z = np.arctanh(corr)
        se = 1 / np.sqrt(n - 3)
        alpha = 1 - confidence_level
        z_crit = stats.norm.ppf(1 - alpha/2)
        z_lower = z - z_crit * se
        z_upper = z + z_crit * se
        ci_lower = np.tanh(z_lower)
        ci_upper = np.tanh(z_upper)
    else:
        # For non-parametric correlations, use bootstrap
        ci_lower, ci_upper = bootstrap_confidence_interval(
            np.column_stack([x, y]), 
            lambda data: stats.spearmanr(data[:, 0], data[:, 1])[0] if method == 'spearman' 
                        else stats.kendalltau(data[:, 0], data[:, 1])[0],
            confidence_level=confidence_level
        )
    
    return {
        'correlation': corr,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'method': method
    }


def group_comparison_analysis(group1: np.ndarray, group2: np.ndarray,
                            test_type: str = 'ttest',
                            equal_var: bool = True) -> Dict[str, float]:
    """
    Perform group comparison analysis.
    
    Args:
        group1 (np.ndarray): First group data
        group2 (np.ndarray): Second group data
        test_type (str): Type of test ('ttest', 'mannwhitney', 'permutation')
        equal_var (bool): Whether to assume equal variances (for t-test)
        
    Returns:
        Dict[str, float]: Dictionary containing test results
    """
    if test_type == 'ttest':
        if equal_var:
            statistic, p_value = stats.ttest_ind(group1, group2)
        else:
            statistic, p_value = stats.ttest_ind(group1, group2, equal_var=False)
    elif test_type == 'mannwhitney':
        statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    elif test_type == 'permutation':
        statistic, p_value = permutation_test(group1, group2)
    else:
        raise ValueError(f"Unknown test type: {test_type}")
    
    # Compute effect size
    effect_size = compute_effect_size(group1, group2)
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'effect_size': effect_size,
        'group1_mean': np.mean(group1),
        'group1_std': np.std(group1),
        'group2_mean': np.mean(group2),
        'group2_std': np.std(group2),
        'test_type': test_type
    }


def multiple_correlation_analysis(data: pd.DataFrame, 
                                target_variable: str,
                                predictor_variables: List[str],
                                method: str = 'pearson',
                                correction_method: str = 'fdr_bh',
                                alpha: float = 0.05) -> pd.DataFrame:
    """
    Perform multiple correlation analysis with multiple comparison correction.
    
    Args:
        data (pd.DataFrame): Input data
        target_variable (str): Name of target variable
        predictor_variables (List[str]): List of predictor variable names
        method (str): Correlation method
        correction_method (str): Multiple comparison correction method
        alpha (float): Significance level
        
    Returns:
        pd.DataFrame: Results of correlation analysis
    """
    results = []
    
    for predictor in predictor_variables:
        # Remove missing values
        valid_data = data[[target_variable, predictor]].dropna()
        
        if len(valid_data) < 3:  # Need at least 3 observations
            continue
        
        x = valid_data[target_variable].values
        y = valid_data[predictor].values
        
        # Compute correlation
        corr_results = correlation_analysis(x, y, method=method)
        
        results.append({
            'predictor': predictor,
            'correlation': corr_results['correlation'],
            'p_value': corr_results['p_value'],
            'ci_lower': corr_results['ci_lower'],
            'ci_upper': corr_results['ci_upper'],
            'n_observations': len(valid_data)
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Apply multiple comparison correction
    if len(results_df) > 1:
        correction_results = apply_multiple_comparison_correction(
            results_df['p_value'].values, 
            method=correction_method, 
            alpha=alpha
        )
        results_df['corrected_p_value'] = correction_results['corrected_p_values']
        results_df['significant'] = correction_results['significant']
    else:
        results_df['corrected_p_value'] = results_df['p_value']
        results_df['significant'] = results_df['p_value'] < alpha
    
    return results_df.sort_values('p_value')


def compute_consensus_features(feature_lists: List[List[int]], 
                             threshold: float = 0.5) -> Dict[int, float]:
    """
    Compute consensus features across multiple models/folds.
    
    Args:
        feature_lists (List[List[int]]): List of feature lists from different models
        threshold (float): Minimum proportion of models that must include a feature
        
    Returns:
        Dict[int, float]: Dictionary mapping feature indices to consensus scores
    """
    # Flatten all feature lists
    all_features = []
    for feature_list in feature_lists:
        all_features.extend(feature_list)
    
    # Count occurrences
    feature_counts = {}
    for feature in all_features:
        feature_counts[feature] = feature_counts.get(feature, 0) + 1
    
    # Compute consensus scores
    n_models = len(feature_lists)
    consensus_features = {}
    
    for feature, count in feature_counts.items():
        consensus_score = count / n_models
        if consensus_score >= threshold:
            consensus_features[feature] = consensus_score
    
    return consensus_features


def compute_similarity_metrics(set1: set, set2: set) -> Dict[str, float]:
    """
    Compute similarity metrics between two sets of features.
    
    Args:
        set1 (set): First set of features
        set2 (set): Second set of features
        
    Returns:
        Dict[str, float]: Dictionary containing similarity metrics
    """
    intersection = set1 & set2
    union = set1 | set2
    
    # Jaccard index
    jaccard = len(intersection) / len(union) if union else 0.0
    
    # Dice coefficient
    dice = (2 * len(intersection)) / (len(set1) + len(set2)) if (len(set1) + len(set2)) > 0 else 0.0
    
    # Overlap coefficient
    overlap = len(intersection) / min(len(set1), len(set2)) if min(len(set1), len(set2)) > 0 else 0.0
    
    return {
        'jaccard': jaccard,
        'dice': dice,
        'overlap': overlap,
        'intersection_size': len(intersection),
        'union_size': len(union)
    }
