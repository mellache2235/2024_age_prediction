#!/usr/bin/env python3
"""
Lean, optimized brain-behavior analysis.

Usage:
    python run_optimized_brain_behavior.py --dataset nki
    python run_optimized_brain_behavior.py --dataset adhd200_td
    python run_optimized_brain_behavior.py --dataset cmihbn_td
    python run_optimized_brain_behavior.py --all
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from brain_behavior_utils import run_brain_behavior_analysis

# ============================================================================
# DATASET CONFIGURATIONS
# ============================================================================

BASE_IG_DIR = Path("/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/integrated_gradients")
BASE_BEH_DIR = Path("/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/FLUX/assessment_data")
BASE_OUTPUT_DIR = Path("/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior_optimized")

DATASETS = {
    'nki': {
        'name': 'NKI-RS TD',
        'ig_csv': BASE_IG_DIR / "nki_cog_dev_wIDS_features_IG_convnet_regressor_single_model_fold_0.csv",
        'behavioral_csv': BASE_BEH_DIR / "nki_behavioral_merged.csv",  # User needs to create merged file
        'behavioral_cols': None,  # Auto-detect
        'max_measures': None  # Analyze all
    },
    
    'adhd200_td': {
        'name': 'ADHD200 TD',
        'ig_csv': BASE_IG_DIR / "adhd200_td_features_all_sites_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS_single_model_predictions.csv",
        'behavioral_csv': "/oak/stanford/groups/menon/deriveddata/public/adhd200/restfmri/timeseries/group_level/brainnetome/normz/adhd200_run-rest_brainnetome_mean_regMov-6param_wmcsf_dt1_bpf008-09_normz_246ROIs.pklz",  # Special handling needed
        'behavioral_cols': ['Hyper_Impulsive', 'Inattentive'],
        'max_measures': None
    },
    
    'cmihbn_td': {
        'name': 'CMI-HBN TD',
        'ig_csv': BASE_IG_DIR / "cmihbn_td_features_all_sites_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS_single_model_predictions.csv",
        'behavioral_csv': "/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/prepare_data/adhd/C3SR.csv",
        'behavioral_cols': None,
        'max_measures': None
    }
}


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Optimized brain-behavior analysis with hyperparameter tuning"
    )
    parser.add_argument(
        '--dataset', '-d',
        choices=list(DATASETS.keys()) + ['all'],
        default='nki',
        help="Dataset to analyze"
    )
    parser.add_argument(
        '--max-measures', '-m',
        type=int,
        default=None,
        help="Maximum number of behavioral measures to analyze (for testing)"
    )
    
    args = parser.parse_args()
    
    # Determine which datasets to process
    if args.dataset == 'all':
        datasets_to_process = list(DATASETS.keys())
    else:
        datasets_to_process = [args.dataset]
    
    # Process each dataset
    for dataset_key in datasets_to_process:
        config = DATASETS[dataset_key]
        
        # Update max_measures if specified
        if args.max_measures:
            config['max_measures'] = args.max_measures
        
        # Run analysis
        output_dir = BASE_OUTPUT_DIR / dataset_key
        
        try:
            summary_df = run_brain_behavior_analysis(
                ig_csv=str(config['ig_csv']),
                behavioral_csv=str(config['behavioral_csv']),
                output_dir=str(output_dir),
                dataset_name=config['name'],
                behavioral_cols=config['behavioral_cols'],
                max_measures=config['max_measures']
            )
            
            print(f"\n✅ {config['name']} complete!")
            print(f"   Best ρ: {summary_df['Spearman_Rho'].max():.3f}")
            print(f"   Results: {output_dir}")
            
        except Exception as e:
            print(f"\n❌ {config['name']} failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

