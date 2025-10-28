#!/usr/bin/env python3
"""
Cosine similarity analysis of brain feature maps across cohorts.

Compares TD, ADHD, and ASD cohorts using cosine similarity on IG scores.

Usage:
    python cosine_similarity_analysis.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import seaborn as sns

# Set Arial font
font_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/clustering_analysis/arial.ttf'
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()

# Add utils to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
from logging_utils import (print_section_header, print_step, print_success, 
                           print_warning, print_error, print_info, print_completion)

# ============================================================================
# DATASET CONFIGURATIONS
# ============================================================================

BASE_DIR = Path('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/integrated_gradients')
OUTPUT_DIR = Path('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/cosine_similarity')

# TD Cohorts (pooled)
TD_DATASETS = {
    'HCP-Dev': BASE_DIR / 'hcp_dev_wIDS_features_IG_convnet_regressor_most_updated.csv',
    'NKI': BASE_DIR / 'nki_cog_dev_wIDS_features_IG_convnet_regressor_single_model_fold_0.csv',
    'CMI-HBN TD': BASE_DIR / 'cmihbn_td_features_all_sites_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS_single_model_predictions.csv',
    'ADHD200 TD': BASE_DIR / 'adhd200_td_features_all_sites_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS_single_model_predictions.csv'
}

# ADHD Cohorts
ADHD_DATASETS = {
    'ADHD200 ADHD': BASE_DIR / 'adhd200_adhd_features_all_sites_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS_single_model_predictions.csv',
    'CMI-HBN ADHD': BASE_DIR / 'cmihbn_adhd_no_cutoffs_features_all_sites_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS_single_model_predictions.csv'
}

# ASD Cohorts
ASD_DATASETS = {
    'ABIDE ASD': BASE_DIR / 'abide_asd_features_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS.csv',
    'Stanford ASD': BASE_DIR / 'stanford_asd_features_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS.csv'
}


def load_ig_scores(csv_path):
    """Load IG scores from CSV file."""
    df = pd.read_csv(csv_path)
    
    # Get subject IDs
    if 'subject_id' in df.columns:
        subject_ids = df['subject_id'].values
    elif 'Unnamed: 0' in df.columns:
        subject_ids = df['Unnamed: 0'].values
    else:
        subject_ids = df.index.values
    
    # Get ROI columns (exclude subject_id column)
    roi_cols = [col for col in df.columns if col not in ['subject_id', 'Unnamed: 0']]
    
    # Extract IG matrix
    ig_matrix = df[roi_cols].values
    
    return ig_matrix, roi_cols, subject_ids


def compute_mean_feature_map(datasets_dict):
    """
    Compute mean feature map for each dataset separately and overall.
    
    Args:
        datasets_dict: Dictionary of {name: path} for datasets
        
    Returns:
        Dictionary of mean features per dataset, and pooled mean
    """
    dataset_means = {}
    all_features = []
    
    for name, path in datasets_dict.items():
        if not path.exists():
            print_warning(f"File not found: {path}")
            continue
            
        ig_matrix, roi_cols, subject_ids = load_ig_scores(path)
        
        # Compute mean for this dataset
        dataset_mean = np.mean(ig_matrix, axis=0)
        dataset_means[name] = dataset_mean
        
        all_features.append(ig_matrix)
        print_info(f"{name}: {len(subject_ids)} subjects, {len(roi_cols)} ROIs")
    
    if not all_features:
        raise ValueError("No valid datasets found")
    
    # Concatenate all subjects
    all_features = np.vstack(all_features)
    
    # Compute pooled mean across all subjects
    pooled_mean = np.mean(all_features, axis=0)
    
    print_success(f"Pooled: {all_features.shape[0]} total subjects")
    
    return dataset_means, pooled_mean, all_features


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    return 1 - cosine(vec1, vec2)


def compute_pairwise_similarities(features1, features2):
    """
    Compute pairwise cosine similarities between two sets of feature vectors.
    
    Args:
        features1: Array of shape (n_subjects1, n_features)
        features2: Array of shape (n_subjects2, n_features)
        
    Returns:
        Array of similarities
    """
    similarities = []
    
    for feat1 in features1:
        for feat2 in features2:
            sim = cosine_similarity(feat1, feat2)
            similarities.append(sim)
    
    return np.array(similarities)


def main():
    """Main analysis function."""
    print_section_header("COSINE SIMILARITY ANALYSIS - TD vs ADHD vs ASD")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # ====================================================================
        # STEP 1: Load and compute mean feature maps
        # ====================================================================
        print_step("Loading TD cohorts", "HCP-Dev, NKI, CMI-HBN TD, ADHD200 TD")
        td_dataset_means, td_pooled_mean, td_all = compute_mean_feature_map(TD_DATASETS)
        
        print_step("Loading ADHD cohorts", "ADHD200 ADHD, CMI-HBN ADHD")
        adhd_dataset_means, adhd_pooled_mean, adhd_all = compute_mean_feature_map(ADHD_DATASETS)
        
        print_step("Loading ASD cohorts", "ABIDE ASD, Stanford ASD")
        asd_dataset_means, asd_pooled_mean, asd_all = compute_mean_feature_map(ASD_DATASETS)
        
        # ====================================================================
        # STEP 2: Compute cosine similarities between pooled means
        # ====================================================================
        print_step("Computing cosine similarities", "Between pooled group means")
        
        sim_td_adhd_pooled = cosine_similarity(td_pooled_mean, adhd_pooled_mean)
        sim_td_asd_pooled = cosine_similarity(td_pooled_mean, asd_pooled_mean)
        sim_adhd_asd_pooled = cosine_similarity(adhd_pooled_mean, asd_pooled_mean)
        
        print_info(f"TD vs ADHD (pooled): {sim_td_adhd_pooled:.4f}")
        print_info(f"TD vs ASD (pooled): {sim_td_asd_pooled:.4f}")
        print_info(f"ADHD vs ASD (pooled): {sim_adhd_asd_pooled:.4f}")
        
        # ====================================================================
        # STEP 2b: Compute pairwise similarities between individual datasets
        # ====================================================================
        print_step("Computing pairwise similarities", "Between individual datasets")
        
        # TD vs ADHD (all pairwise combinations)
        td_adhd_sims = []
        for td_name, td_vec in td_dataset_means.items():
            for adhd_name, adhd_vec in adhd_dataset_means.items():
                sim = cosine_similarity(td_vec, adhd_vec)
                td_adhd_sims.append(sim)
                print_info(f"  {td_name} vs {adhd_name}: {sim:.4f}")
        
        # TD vs ASD (all pairwise combinations)
        td_asd_sims = []
        for td_name, td_vec in td_dataset_means.items():
            for asd_name, asd_vec in asd_dataset_means.items():
                sim = cosine_similarity(td_vec, asd_vec)
                td_asd_sims.append(sim)
                print_info(f"  {td_name} vs {asd_name}: {sim:.4f}")
        
        # ADHD vs ASD (all pairwise combinations)
        adhd_asd_sims = []
        for adhd_name, adhd_vec in adhd_dataset_means.items():
            for asd_name, asd_vec in asd_dataset_means.items():
                sim = cosine_similarity(adhd_vec, asd_vec)
                adhd_asd_sims.append(sim)
                print_info(f"  {adhd_name} vs {asd_name}: {sim:.4f}")
        
        # Compute ranges
        td_adhd_range = (np.min(td_adhd_sims), np.max(td_adhd_sims))
        td_asd_range = (np.min(td_asd_sims), np.max(td_asd_sims))
        adhd_asd_range = (np.min(adhd_asd_sims), np.max(adhd_asd_sims))
        
        print_info(f"\nTD vs ADHD range: {td_adhd_range[0]:.4f} to {td_adhd_range[1]:.4f}")
        print_info(f"TD vs ASD range: {td_asd_range[0]:.4f} to {td_asd_range[1]:.4f}")
        print_info(f"ADHD vs ASD range: {adhd_asd_range[0]:.4f} to {adhd_asd_range[1]:.4f}")
        
        # ====================================================================
        # STEP 3: Compute within-group and between-group similarities
        # ====================================================================
        print_step("Computing within-group similarities", "For statistical comparison")
        
        # Sample 1000 random pairs for computational efficiency
        n_samples = min(1000, td_all.shape[0])
        
        # Within-group similarities
        td_indices = np.random.choice(td_all.shape[0], n_samples, replace=False)
        adhd_indices = np.random.choice(adhd_all.shape[0], min(n_samples, adhd_all.shape[0]), replace=False)
        asd_indices = np.random.choice(asd_all.shape[0], min(n_samples, asd_all.shape[0]), replace=False)
        
        within_td = []
        for i in range(len(td_indices)):
            for j in range(i+1, min(i+10, len(td_indices))):  # Compare with 10 neighbors
                sim = cosine_similarity(td_all[td_indices[i]], td_all[td_indices[j]])
                within_td.append(sim)
        
        within_adhd = []
        for i in range(len(adhd_indices)):
            for j in range(i+1, min(i+10, len(adhd_indices))):
                sim = cosine_similarity(adhd_all[adhd_indices[i]], adhd_all[adhd_indices[j]])
                within_adhd.append(sim)
        
        within_asd = []
        for i in range(len(asd_indices)):
            for j in range(i+1, min(i+10, len(asd_indices))):
                sim = cosine_similarity(asd_all[asd_indices[i]], asd_all[asd_indices[j]])
                within_asd.append(sim)
        
        # Between-group similarities (sample)
        between_td_adhd = []
        for i in range(min(100, len(td_indices))):
            for j in range(min(100, len(adhd_indices))):
                sim = cosine_similarity(td_all[td_indices[i]], adhd_all[adhd_indices[j]])
                between_td_adhd.append(sim)
        
        between_td_asd = []
        for i in range(min(100, len(td_indices))):
            for j in range(min(100, len(asd_indices))):
                sim = cosine_similarity(td_all[td_indices[i]], asd_all[asd_indices[j]])
                between_td_asd.append(sim)
        
        print_info(f"Within TD: {np.mean(within_td):.4f} ± {np.std(within_td):.4f}")
        print_info(f"Within ADHD: {np.mean(within_adhd):.4f} ± {np.std(within_adhd):.4f}")
        print_info(f"Within ASD: {np.mean(within_asd):.4f} ± {np.std(within_asd):.4f}")
        print_info(f"Between TD-ADHD: {np.mean(between_td_adhd):.4f} ± {np.std(between_td_adhd):.4f}")
        print_info(f"Between TD-ASD: {np.mean(between_td_asd):.4f} ± {np.std(between_td_asd):.4f}")
        
        # ====================================================================
        # STEP 4: Save results
        # ====================================================================
        print_step("Saving results", "CSV and summary")
        
        # Create summary DataFrame
        results_df = pd.DataFrame({
            'Comparison': [
                'TD vs ADHD (pooled means)',
                'TD vs ASD (pooled means)',
                'ADHD vs ASD (pooled means)',
                'TD vs ADHD (range across datasets)',
                'TD vs ASD (range across datasets)',
                'ADHD vs ASD (range across datasets)',
                'Within TD',
                'Within ADHD',
                'Within ASD',
                'Between TD-ADHD (subjects)',
                'Between TD-ASD (subjects)'
            ],
            'Cosine_Similarity': [
                sim_td_adhd_pooled,
                sim_td_asd_pooled,
                sim_adhd_asd_pooled,
                np.mean(td_adhd_sims),
                np.mean(td_asd_sims),
                np.mean(adhd_asd_sims),
                np.mean(within_td),
                np.mean(within_adhd),
                np.mean(within_asd),
                np.mean(between_td_adhd),
                np.mean(between_td_asd)
            ],
            'Min': [
                np.nan,
                np.nan,
                np.nan,
                td_adhd_range[0],
                td_asd_range[0],
                adhd_asd_range[0],
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan
            ],
            'Max': [
                np.nan,
                np.nan,
                np.nan,
                td_adhd_range[1],
                td_asd_range[1],
                adhd_asd_range[1],
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan
            ],
            'Std': [
                np.nan,
                np.nan,
                np.nan,
                np.std(td_adhd_sims),
                np.std(td_asd_sims),
                np.std(adhd_asd_sims),
                np.std(within_td),
                np.std(within_adhd),
                np.std(within_asd),
                np.std(between_td_adhd),
                np.std(between_td_asd)
            ]
        })
        
        csv_path = OUTPUT_DIR / 'cosine_similarity_results.csv'
        results_df.to_csv(csv_path, index=False)
        print_success(f"Results saved: {csv_path}")
        
        # ====================================================================
        # STEP 5: Create visualization
        # ====================================================================
        print_step("Creating visualization", "Similarity matrix heatmap")
        
        # Create similarity matrix (using pooled means)
        sim_matrix = np.array([
            [1.0, sim_td_adhd_pooled, sim_td_asd_pooled],
            [sim_td_adhd_pooled, 1.0, sim_adhd_asd_pooled],
            [sim_td_asd_pooled, sim_adhd_asd_pooled, 1.0]
        ])
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
        
        sns.heatmap(sim_matrix, 
                    annot=True, 
                    fmt='.4f',
                    cmap='RdYlGn',
                    vmin=0.9, vmax=1.0,
                    cbar_kws={'label': 'Cosine Similarity'},
                    linewidths=2,
                    linecolor='black',
                    xticklabels=['TD\n(pooled)', 'ADHD\n(pooled)', 'ASD\n(pooled)'],
                    yticklabels=['TD\n(pooled)', 'ADHD\n(pooled)', 'ASD\n(pooled)'],
                    ax=ax,
                    square=True)
        
        ax.set_title('Cosine Similarity Between Cohort Mean Feature Maps', 
                     fontsize=16, fontweight='bold', pad=20)
        
        # Customize
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=12)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
        
        plt.tight_layout()
        
        plot_path = OUTPUT_DIR / 'cosine_similarity_heatmap.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print_success(f"Heatmap saved: {plot_path}")
        
        # ====================================================================
        # STEP 6: Print summary report
        # ====================================================================
        print_section_header("COSINE SIMILARITY ANALYSIS RESULTS")
        
        print("\n" + "="*80)
        print("POOLED MEAN SIMILARITIES (between group averages)")
        print("="*80)
        print(f"TD vs ADHD:    {sim_td_adhd_pooled:.4f}")
        print(f"TD vs ASD:     {sim_td_asd_pooled:.4f}")
        print(f"ADHD vs ASD:   {sim_adhd_asd_pooled:.4f}")
        
        print("\n" + "="*80)
        print("PAIRWISE DATASET SIMILARITIES (range across individual datasets)")
        print("="*80)
        print(f"TD vs ADHD:    {td_adhd_range[0]:.4f} to {td_adhd_range[1]:.4f} (mean: {np.mean(td_adhd_sims):.4f})")
        print(f"TD vs ASD:     {td_asd_range[0]:.4f} to {td_asd_range[1]:.4f} (mean: {np.mean(td_asd_sims):.4f})")
        print(f"ADHD vs ASD:   {adhd_asd_range[0]:.4f} to {adhd_asd_range[1]:.4f} (mean: {np.mean(adhd_asd_sims):.4f})")
        
        print("\n" + "="*80)
        print("INTERPRETATION")
        print("="*80)
        print(f"Cosine similarity ranges from -1 (opposite) to 1 (identical)")
        print(f"Values close to 1 indicate high similarity in feature patterns")
        print(f"\nThese regions demonstrated remarkable consistency across both")
        print(f"the HCP-Development and three clinical cohorts, with cosine")
        print(f"similarity between feature importance maps ranging from")
        print(f"{td_adhd_range[0]:.3f} to {td_adhd_range[1]:.3f} (TD vs ADHD) and")
        print(f"{td_asd_range[0]:.3f} to {td_asd_range[1]:.3f} (TD vs ASD).")
        
        print("\n" + "="*80)
        print("SUBJECT-LEVEL STATISTICS (within and between groups)")
        print("="*80)
        print(f"Within TD:         {np.mean(within_td):.4f} ± {np.std(within_td):.4f}")
        print(f"Within ADHD:       {np.mean(within_adhd):.4f} ± {np.std(within_adhd):.4f}")
        print(f"Within ASD:        {np.mean(within_asd):.4f} ± {np.std(within_asd):.4f}")
        print(f"Between TD-ADHD:   {np.mean(between_td_adhd):.4f} ± {np.std(between_td_adhd):.4f}")
        print(f"Between TD-ASD:    {np.mean(between_td_asd):.4f} ± {np.std(between_td_asd):.4f}")
        
        print("\n" + "="*80)
        print("OUTPUT FILES")
        print("="*80)
        print(f"Results CSV:  {csv_path}")
        print(f"Heatmap PNG:  {plot_path}")
        print("="*80 + "\n")
        
        print_completion("Cosine similarity analysis completed successfully!")
        
    except Exception as e:
        print_error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
