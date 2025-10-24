#!/usr/bin/env python3
"""
Setup script to create the complete directory structure for storing analysis results.

This script creates all necessary directories for:
- Brain age prediction results
- Integrated gradients and feature attribution
- Count data and consensus analysis
- Network analysis and plots
- Brain-behavior correlation analysis
- Region tables and summaries
- Cosine similarity analysis
- All visualization outputs
"""

import os
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_directory_structure(base_dir: str = "results"):
    """
    Create the complete directory structure for storing analysis results.
    
    Args:
        base_dir (str): Base directory for results (default: "results")
    """
    
    # Define the complete directory structure
    directories = [
        # Brain Age Prediction Results
        f"{base_dir}/brain_age_prediction",
        f"{base_dir}/brain_age_prediction/models",
        f"{base_dir}/brain_age_prediction/predictions",
        f"{base_dir}/brain_age_prediction/bias_correction",
        f"{base_dir}/brain_age_prediction/comprehensive_analysis",
        
        # Integrated Gradients and Feature Attribution
        f"{base_dir}/integrated_gradients",
        f"{base_dir}/integrated_gradients/dev",
        f"{base_dir}/integrated_gradients/nki_rs_td",
        f"{base_dir}/integrated_gradients/adhd200_td",
        f"{base_dir}/integrated_gradients/adhd200_adhd",
        f"{base_dir}/integrated_gradients/cmihbn_td",
        f"{base_dir}/integrated_gradients/cmihbn_adhd",
        f"{base_dir}/integrated_gradients/abide_asd",
        f"{base_dir}/integrated_gradients/stanford_asd",
        
        # Count Data and Consensus Analysis
        f"{base_dir}/count_data",
        f"{base_dir}/consensus_analysis",
        f"{base_dir}/consensus_analysis/individual_datasets",
        f"{base_dir}/consensus_analysis/shared_analysis",
        
        # Network Analysis
        f"{base_dir}/network_analysis",
        f"{base_dir}/network_analysis/yeo_network_plots",
        f"{base_dir}/network_analysis/cosine_similarity",
        f"{base_dir}/network_analysis/network_summaries",
        
        # Brain-Behavior Correlation Analysis
        f"{base_dir}/brain_behavior_analysis",
        f"{base_dir}/brain_behavior_analysis/td_cohorts",
        f"{base_dir}/brain_behavior_analysis/adhd_cohorts",
        f"{base_dir}/brain_behavior_analysis/asd_cohorts",
        f"{base_dir}/brain_behavior_analysis/correlation_matrices",
        f"{base_dir}/brain_behavior_analysis/fdr_corrected",
        
        # Region Tables and Summaries
        f"{base_dir}/region_tables",
        f"{base_dir}/region_tables/individual_datasets",
        f"{base_dir}/region_tables/shared_regions",
        f"{base_dir}/region_tables/overlap_analysis",
        
        # Visualization Outputs
        f"{base_dir}/figures",
        f"{base_dir}/figures/brain_age_correlations",
        f"{base_dir}/figures/brain_visualization",
        f"{base_dir}/figures/network_analysis",
        f"{base_dir}/figures/brain_behavior_analysis",
        f"{base_dir}/figures/3d_brain_surfaces",
        f"{base_dir}/figures/nifti_visualizations",
        f"{base_dir}/figures/polar_bar_plots",
        f"{base_dir}/figures/scatter_plots",
        f"{base_dir}/figures/correlation_plots",
        
        # Statistical Analysis Results
        f"{base_dir}/statistical_analysis",
        f"{base_dir}/statistical_analysis/significance_testing",
        f"{base_dir}/statistical_analysis/effect_sizes",
        f"{base_dir}/statistical_analysis/group_comparisons",
        
        # Model Performance and Metrics
        f"{base_dir}/model_performance",
        f"{base_dir}/model_performance/cross_validation",
        f"{base_dir}/model_performance/external_validation",
        f"{base_dir}/model_performance/bias_analysis",
        
        # Data Processing and Intermediate Results
        f"{base_dir}/data_processing",
        f"{base_dir}/data_processing/standardized_data",
        f"{base_dir}/data_processing/feature_engineering",
        f"{base_dir}/data_processing/quality_control",
        
        # Reports and Summaries
        f"{base_dir}/reports",
        f"{base_dir}/reports/executive_summaries",
        f"{base_dir}/reports/detailed_analyses",
        f"{base_dir}/reports/method_comparisons",
        
        # Temporary and Working Directories
        f"{base_dir}/temp",
        f"{base_dir}/temp/processing",
        f"{base_dir}/temp/visualization",
        f"{base_dir}/temp/analysis",
        
        # Archive and Backup
        f"{base_dir}/archive",
        f"{base_dir}/archive/old_results",
        f"{base_dir}/archive/backup_models",
    ]
    
    # Create all directories
    created_count = 0
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            created_count += 1
            logging.info(f"Created directory: {directory}")
        except Exception as e:
            logging.error(f"Failed to create directory {directory}: {e}")
    
    logging.info(f"Successfully created {created_count} directories in {base_dir}/")
    
    # Create README files in key directories
    create_readme_files(base_dir)
    
    return created_count


def create_readme_files(base_dir: str):
    """Create README files in key directories to document their purpose."""
    
    readme_content = {
        f"{base_dir}/brain_age_prediction/README.md": """
# Brain Age Prediction Results

This directory contains results from brain age prediction analysis:

- `models/`: Trained PyTorch Lightning models and checkpoints
- `predictions/`: Age predictions for all datasets
- `bias_correction/`: Bias correction parameters and corrected predictions
- `comprehensive_analysis/`: Overall analysis results and summaries
""",
        
        f"{base_dir}/integrated_gradients/README.md": """
# Integrated Gradients Results

This directory contains feature attribution results using Integrated Gradients:

- Individual dataset folders contain IG scores for each subject
- Files are organized by dataset (dev, nki_rs_td, adhd200_td, etc.)
- Each dataset folder contains CSV files with ROI-level IG scores
""",
        
        f"{base_dir}/count_data/README.md": """
# Count Data Results

This directory contains consensus count data:

- CSV files with count data for each dataset
- Count represents how many times each ROI was in top X% of features
- Used for network analysis and region importance tables
""",
        
        f"{base_dir}/network_analysis/README.md": """
# Network Analysis Results

This directory contains network-level analysis results:

- `yeo_network_plots/`: Polar bar plots grouped by Yeo networks
- `cosine_similarity/`: Cosine similarity analysis between cohorts
- `network_summaries/`: Network-level summaries and statistics
""",
        
        f"{base_dir}/brain_behavior_analysis/README.md": """
# Brain-Behavior Correlation Analysis

This directory contains brain-behavior correlation results:

- `td_cohorts/`: Typically developing cohort analyses
- `adhd_cohorts/`: ADHD cohort analyses
- `asd_cohorts/`: ASD cohort analyses
- `correlation_matrices/`: Correlation matrices and statistics
- `fdr_corrected/`: FDR-corrected results
""",
        
        f"{base_dir}/figures/README.md": """
# Visualization Outputs

This directory contains all generated figures:

- `brain_age_correlations/`: Scatter plots of predicted vs actual age
- `brain_visualization/`: 3D brain surface plots and NIfTI visualizations
- `network_analysis/`: Network-level plots and polar bar charts
- `brain_behavior_analysis/`: Brain-behavior correlation plots
- `3d_brain_surfaces/`: 3D brain surface visualizations
- `nifti_visualizations/`: NIfTI image visualizations
- `polar_bar_plots/`: Polar bar plots for network analysis
- `scatter_plots/`: Scatter plots for correlations
- `correlation_plots/`: Brain-behavior correlation plots
""",
        
        f"{base_dir}/region_tables/README.md": """
# Region Tables and Summaries

This directory contains region importance tables:

- `individual_datasets/`: Top regions for each dataset
- `shared_regions/`: Regions shared across multiple datasets
- `overlap_analysis/`: Overlap analysis between cohorts
""",
    }
    
    for file_path, content in readme_content.items():
        try:
            with open(file_path, 'w') as f:
                f.write(content.strip())
            logging.info(f"Created README: {file_path}")
        except Exception as e:
            logging.error(f"Failed to create README {file_path}: {e}")


def main():
    """Main function to set up results directories."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create directory structure for analysis results")
    parser.add_argument("--base_dir", type=str, default="results",
                       help="Base directory for results (default: results)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create directory structure
    created_count = create_directory_structure(args.base_dir)
    
    print(f"\n✅ Successfully created {created_count} directories!")
    print(f"📁 Results will be stored in: {args.base_dir}/")
    print(f"📋 Check the README files in key directories for usage information.")
    
    # Print summary of key directories
    print(f"\n📊 Key result directories:")
    print(f"   🧠 Brain age prediction: {args.base_dir}/brain_age_prediction/")
    print(f"   🔍 Feature attribution: {args.base_dir}/integrated_gradients/")
    print(f"   📈 Network analysis: {args.base_dir}/network_analysis/")
    print(f"   🧬 Brain-behavior: {args.base_dir}/brain_behavior_analysis/")
    print(f"   📊 Figures: {args.base_dir}/figures/")
    print(f"   📋 Region tables: {args.base_dir}/region_tables/")


if __name__ == "__main__":
    main()
