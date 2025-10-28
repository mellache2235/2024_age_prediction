#!/usr/bin/env python3
"""
Run comprehensive statistical comparisons between brain feature maps.

Compares TD, ADHD, and ASD cohorts using:
- Cosine similarity with permutation p-values
- Spearman correlation on ROI ranks  
- Aitchison (CLR) distance
- Jensen-Shannon divergence
- ROI-wise two-proportion tests with FDR
- Network-level aggregation

Usage:
    python run_statistical_comparisons.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from statistical_comparison_utils import compare_datasets, create_comparison_report


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path("/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/integrated_gradients")
OUTPUT_DIR = Path("/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/statistical_comparisons")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset configurations
DATASETS = {
    # TD Cohorts
    'NKI TD': BASE_DIR / 'nki_cog_dev_wIDS_features_IG_convnet_regressor_single_model_fold_0.csv',
    'ADHD200 TD': BASE_DIR / 'adhd200_td_features_all_sites_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS_single_model_predictions.csv',
    'CMI-HBN TD': BASE_DIR / 'cmihbn_td_features_all_sites_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS_single_model_predictions.csv',
    
    # ADHD Cohorts
    'ADHD200 ADHD': BASE_DIR / 'adhd200_adhd_features_all_sites_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS_single_model_predictions.csv',
    'CMI-HBN ADHD': BASE_DIR / 'cmihbn_adhd_no_cutoffs_features_all_sites_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS_single_model_predictions.csv',
    
    # ASD Cohorts
    'ABIDE ASD': BASE_DIR / 'abide_asd_features_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS.csv',
    'Stanford ASD': BASE_DIR / 'stanford_asd_features_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS.csv'
}

# Comparisons to run (name1, name2, group)
COMPARISONS = [
    # Within TD
    ('NKI TD', 'ADHD200 TD', 'Within_TD'),
    ('NKI TD', 'CMI-HBN TD', 'Within_TD'),
    ('ADHD200 TD', 'CMI-HBN TD', 'Within_TD'),
    
    # Within ADHD
    ('ADHD200 ADHD', 'CMI-HBN ADHD', 'Within_ADHD'),
    
    # Within ASD
    ('ABIDE ASD', 'Stanford ASD', 'Within_ASD'),
    
    # TD vs ADHD (same dataset)
    ('ADHD200 TD', 'ADHD200 ADHD', 'TD_vs_ADHD'),
    ('CMI-HBN TD', 'CMI-HBN ADHD', 'TD_vs_ADHD'),
    
    # Cross-cohort comparisons
    ('ADHD200 TD', 'ABIDE ASD', 'TD_vs_ASD'),
    ('CMI-HBN TD', 'ABIDE ASD', 'TD_vs_ASD'),
    ('ADHD200 ADHD', 'ABIDE ASD', 'ADHD_vs_ASD'),
    ('CMI-HBN ADHD', 'ABIDE ASD', 'ADHD_vs_ASD')
]

# Network mapping (simplified - user should provide full mapping)
# This is a placeholder - load from Yeo atlas
YEO_ATLAS_PATH = "/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/feature_attribution/csv_files/subregion_func_network_Yeo_updated_yz.csv"


# ============================================================================
# DATA LOADING
# ============================================================================

def load_ig_data(csv_path: Path) -> np.ndarray:
    """Load IG scores as numpy array (subjects × ROIs)."""
    df = pd.read_csv(csv_path)
    
    # Get ROI columns (all columns except subject_id)
    roi_cols = [c for c in df.columns if c not in ['subject_id', 'Unnamed: 0', 'id', 'ID']]
    
    if len(roi_cols) == 0:
        raise ValueError(f"No ROI columns found in {csv_path.name}")
    
    # Extract as numpy array
    ig_matrix = df[roi_cols].values
    
    print(f"  Loaded {csv_path.name}: {ig_matrix.shape}")
    
    return ig_matrix


def load_network_mapping(atlas_path: str) -> dict:
    """Load ROI to network mapping from Yeo atlas."""
    try:
        atlas_df = pd.read_csv(atlas_path)
        
        # Create mapping from ROI index to network
        roi_to_network = {}
        
        if 'Yeo_17network' in atlas_df.columns:
            for idx, network_id in enumerate(atlas_df['Yeo_17network'].values):
                roi_to_network[idx] = f"Network_{int(network_id)}"
        
        print(f"  Loaded network mapping: {len(roi_to_network)} ROIs")
        return roi_to_network
        
    except Exception as e:
        print(f"  Warning: Could not load network mapping: {e}")
        return None


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all comparisons."""
    print("\n" + "="*80)
    print("COMPREHENSIVE STATISTICAL COMPARISONS OF BRAIN FEATURE MAPS")
    print("="*80)
    
    # Load network mapping
    print("\nLoading network mapping...")
    roi_to_network = load_network_mapping(YEO_ATLAS_PATH)
    
    # Load all datasets
    print("\nLoading datasets...")
    datasets = {}
    for name, path in DATASETS.items():
        try:
            datasets[name] = load_ig_data(path)
        except Exception as e:
            print(f"  ⚠ Could not load {name}: {e}")
    
    print(f"\n✓ Loaded {len(datasets)} datasets")
    
    # Run comparisons
    print(f"\n{'='*80}")
    print(f"RUNNING {len(COMPARISONS)} COMPARISONS")
    print(f"{'='*80}")
    
    all_results = []
    
    for i, (name1, name2, group) in enumerate(COMPARISONS, 1):
        print(f"\n[{i}/{len(COMPARISONS)}] Comparing: {name1} vs {name2} ({group})")
        print("-" * 80)
        
        # Check if both datasets available
        if name1 not in datasets or name2 not in datasets:
            print(f"  ⚠ Skipping: One or both datasets not available")
            continue
        
        # Run comparison
        try:
            results = compare_datasets(
                datasets[name1],
                datasets[name2],
                name1,
                name2,
                roi_to_network=roi_to_network,
                n_permutations=10000,
                verbose=True
            )
            
            # Add group info
            results['comparison_group'] = group
            
            # Save detailed results
            safe_name1 = name1.replace(' ', '_').replace('-', '_')
            safe_name2 = name2.replace(' ', '_').replace('-', '_')
            
            output_file = OUTPUT_DIR / f"{safe_name1}_vs_{safe_name2}.csv"
            create_comparison_report(results, str(output_file))
            
            # Save network summary if available
            if 'network_summary' in results:
                network_file = OUTPUT_DIR / f"{safe_name1}_vs_{safe_name2}_networks.csv"
                results['network_summary'].to_csv(network_file, index=False)
                print(f"✓ Saved network summary: {network_file.name}")
            
            # Collect for summary
            all_results.append({
                'Dataset_1': name1,
                'Dataset_2': name2,
                'Group': group,
                'Cosine_Similarity': results['cosine_similarity'],
                'Cosine_P_Value': results['cosine_p_value'],
                'Spearman_Rho': results['spearman_rho'],
                'Spearman_P_Value': results['spearman_p_value'],
                'Aitchison_Distance': results['aitchison_distance'],
                'Jensen_Shannon_Div': results['jensen_shannon_divergence'],
                'N_Significant_ROIs': results['n_significant_rois'],
                'N_ROIs': results['n_rois']
            })
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Create master summary
    print(f"\n{'='*80}")
    print("CREATING MASTER SUMMARY")
    print(f"{'='*80}\n")
    
    summary_df = pd.DataFrame(all_results)
    summary_file = OUTPUT_DIR / "master_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    
    print(f"✓ Saved master summary: {summary_file}")
    
    # Print summary by group
    print(f"\n{'='*80}")
    print("SUMMARY BY GROUP")
    print(f"{'='*80}\n")
    
    for group in summary_df['Group'].unique():
        group_data = summary_df[summary_df['Group'] == group]
        
        print(f"\n{group}:")
        print(f"  Mean Cosine Similarity:   {group_data['Cosine_Similarity'].mean():.4f} ± {group_data['Cosine_Similarity'].std():.4f}")
        print(f"  Mean Spearman ρ:          {group_data['Spearman_Rho'].mean():.4f} ± {group_data['Spearman_Rho'].std():.4f}")
        print(f"  Mean Aitchison Distance:  {group_data['Aitchison_Distance'].mean():.4f} ± {group_data['Aitchison_Distance'].std():.4f}")
        print(f"  Mean JS Divergence:       {group_data['Jensen_Shannon_Div'].mean():.4f} ± {group_data['Jensen_Shannon_Div'].std():.4f}")
        print(f"  Mean Significant ROIs:    {group_data['N_Significant_ROIs'].mean():.1f} / {group_data['N_ROIs'].iloc[0]}")
    
    print(f"\n{'='*80}")
    print("COMPLETE!")
    print(f"{'='*80}")
    print(f"\n✓ All results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

