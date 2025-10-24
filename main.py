#!/usr/bin/env python3
"""
Main entry point for the age prediction analysis pipeline.

This script provides an overview of the modular analysis pipeline.
Each step can be run independently using the specific scripts.

Available Steps:
1. Train brain age model: python scripts/train_brain_age_model.py
2. Test on external data: python scripts/test_external_dataset.py --dataset <dataset>
3. Compute IG: python scripts/compute_integrated_gradients.py --dataset <dataset> --fold 0
4. Brain-behavior analysis: python scripts/brain_behavior_analysis.py --dataset <dataset>

Usage:
    python main.py  # Shows this help message
"""

import os
import sys
import yaml
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    """Show available analysis steps."""
    print("=" * 80)
    print("AGE PREDICTION ANALYSIS PIPELINE")
    print("=" * 80)
    print()
    print("This pipeline consists of modular steps that can be run independently:")
    print()
    print("1. TRAIN BRAIN AGE MODEL")
    print("   python scripts/train_brain_age_model.py")
    print("   - Trains ConvNet models on HCP-Dev data with nested CV")
    print("   - Saves trained models to results/brain_age_models/")
    print()
    print("2. TEST ON EXTERNAL DATASETS")
    print("   python scripts/test_external_dataset.py --dataset <dataset>")
    print("   Available datasets: nki_rs_td, adhd200_adhd, cmihbn_adhd, adhd200_td, cmihbn_td, abide_asd, stanford_asd")
    print("   - Tests trained models on external datasets")
    print("   - Applies bias correction using TD cohorts")
    print("   - Saves results to results/external_testing/<dataset>/")
    print()
    print("3. COMPUTE INTEGRATED GRADIENTS")
    print("   python scripts/compute_integrated_gradients.py --dataset <dataset> --fold 0")
    print("   Available datasets: nki_rs_td, adhd200_adhd, cmihbn_adhd, adhd200_td, cmihbn_td, abide_asd, stanford_asd")
    print("   - Computes IG scores for specified dataset using trained model")
    print("   - Saves IG scores to results/integrated_gradients/<dataset>/")
    print()
    print("4. BRAIN-BEHAVIOR ANALYSIS")
    print("   python scripts/brain_behavior_analysis.py --dataset <dataset>")
    print("   Available datasets: nki_rs_td, adhd200_adhd, cmihbn_adhd, adhd200_td, cmihbn_td, abide_asd, stanford_asd")
    print("   - Runs PCA on IG scores and correlates with behavioral measures")
    print("   - Applies FDR correction")
    print("   - Saves results to results/brain_behavior/<dataset>/")
    print()
    print("5. ADDITIONAL ANALYSES")
    print("   python scripts/cosine_similarity_analysis.py --discovery_csv <file> --nki_csv <file> --cmihbn_csv <file> --adhd200_csv <file>  # Cosine similarity between discovery and validation cohorts")
    print("   python scripts/network_analysis_yeo.py --count_csv <file> --yeo_atlas <file>  # Network analysis using Yeo atlas")
    print("   python scripts/generate_count_data.py --ig_csv <file> --output <file>  # Generate count data from IG scores")
    print("   python scripts/feature_comparison.py        # Feature overlap analysis")
    print("   python scripts/create_region_tables.py      # Region importance tables")
    print()
    print("=" * 80)
    print("EXAMPLE WORKFLOW:")
    print("=" * 80)
    print("1. python scripts/train_brain_age_model.py")
    print("2. python scripts/test_external_dataset.py --dataset nki_rs_td")
    print("3. python scripts/compute_integrated_gradients.py --dataset nki_rs_td --fold 0")
    print("4. python scripts/brain_behavior_analysis.py --dataset nki_rs_td")
    print()
    print("For more details, see README.md")

if __name__ == "__main__":
    main()