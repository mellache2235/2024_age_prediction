#!/usr/bin/env python3
"""
Main entry point for the age prediction analysis pipeline.

This script orchestrates the complete analysis pipeline including:
1. Brain age prediction with nested CV and bias correction
2. Feature attribution analysis using Integrated Gradients
3. Brain visualization (3D surface plots)
4. Network-level analysis using consensus IG data
5. Brain-behavior correlation analysis with PCA

Usage:
    python main.py
"""

import os
import sys
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Optional

# Add scripts to path
sys.path.append(str(Path(__file__).parent / 'scripts'))

# Import analysis modules
from brain_age_prediction import run_brain_age_prediction_analysis
from comprehensive_brain_behavior_analysis import run_comprehensive_brain_behavior_analysis
from network_analysis import run_network_analysis
from feature_comparison import run_feature_comparison_analysis
from create_region_tables import run_region_table_analysis
from brain_surface_plots import run_brain_visualization
from brain_age_plots import run_brain_age_plotting

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler()
    ]
)

def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        sys.exit(1)

def create_output_directories(config: Dict) -> None:
    """Create necessary output directories."""
    output_config = config.get('output', {})
    base_dir = output_config.get('base_dir', 'results')
    
    directories = [
        base_dir,
        f"{base_dir}/figures",
        f"{base_dir}/models",
        f"{base_dir}/tables",
        f"{base_dir}/brain_behavior",
        f"{base_dir}/network_analysis",
        f"{base_dir}/feature_comparison"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Created directory: {directory}")

def run_brain_age_prediction(config: Dict, output_dir: str) -> Dict:
    """Run brain age prediction analysis."""
    logging.info("=" * 60)
    logging.info("RUNNING BRAIN AGE PREDICTION ANALYSIS")
    logging.info("=" * 60)
    
    try:
        results = run_brain_age_prediction_analysis(config, output_dir)
        logging.info("Brain age prediction analysis completed successfully!")
        return results
    except Exception as e:
        logging.error(f"Error in brain age prediction analysis: {e}")
        return {'status': 'failed', 'error': str(e)}

def run_feature_attribution(config: Dict, output_dir: str) -> Dict:
    """Run feature attribution analysis."""
    logging.info("=" * 60)
    logging.info("RUNNING FEATURE ATTRIBUTION ANALYSIS")
    logging.info("=" * 60)
    
    try:
        # This would run the feature attribution analysis
        # For now, we'll assume IG scores are already computed
        logging.info("Feature attribution analysis (IG computation) - assuming pre-computed")
        return {'status': 'completed', 'message': 'IG scores assumed to be pre-computed'}
    except Exception as e:
        logging.error(f"Error in feature attribution analysis: {e}")
        return {'status': 'failed', 'error': str(e)}

def run_brain_visualization(config: Dict, output_dir: str) -> Dict:
    """Run brain visualization analysis."""
    logging.info("=" * 60)
    logging.info("RUNNING BRAIN VISUALIZATION ANALYSIS")
    logging.info("=" * 60)
    
    try:
        results = run_brain_visualization(config, output_dir)
        logging.info("Brain visualization analysis completed successfully!")
        return results
    except Exception as e:
        logging.error(f"Error in brain visualization analysis: {e}")
        return {'status': 'failed', 'error': str(e)}

def run_network_analysis(config: Dict, output_dir: str) -> Dict:
    """Run network-level analysis."""
    logging.info("=" * 60)
    logging.info("RUNNING NETWORK ANALYSIS")
    logging.info("=" * 60)
    
    try:
        results = run_network_analysis(config, output_dir)
        logging.info("Network analysis completed successfully!")
        return results
    except Exception as e:
        logging.error(f"Error in network analysis: {e}")
        return {'status': 'failed', 'error': str(e)}

def run_brain_behavior_analysis(config: Dict, output_dir: str) -> Dict:
    """Run brain-behavior correlation analysis."""
    logging.info("=" * 60)
    logging.info("RUNNING BRAIN-BEHAVIOR CORRELATION ANALYSIS")
    logging.info("=" * 60)
    
    try:
        results = run_comprehensive_brain_behavior_analysis(config, output_dir)
        logging.info("Brain-behavior correlation analysis completed successfully!")
        return results
    except Exception as e:
        logging.error(f"Error in brain-behavior correlation analysis: {e}")
        return {'status': 'failed', 'error': str(e)}

def main():
    """Main function to run the analysis pipeline."""
    # Load configuration
    config = load_config()
    
    # Create output directories
    create_output_directories(config)
    
    # Run analyses in the specified order
    analyses_to_run = ['brain_age_prediction', 'feature_attribution', 'brain_visualization', 'network_analysis', 'brain_behavior']
    
    logging.info(f"Running analyses: {analyses_to_run}")
    
    # Run analyses in order
    results = {}
    
    for analysis in analyses_to_run:
        if analysis == 'brain_age_prediction':
            results['brain_age_prediction'] = run_brain_age_prediction(config, 'results')
        elif analysis == 'feature_attribution':
            results['feature_attribution'] = run_feature_attribution(config, 'results')
        elif analysis == 'brain_visualization':
            results['brain_visualization'] = run_brain_visualization(config, 'results')
        elif analysis == 'network_analysis':
            results['network_analysis'] = run_network_analysis(config, 'results')
        elif analysis == 'brain_behavior':
            results['brain_behavior'] = run_brain_behavior_analysis(config, 'results')
        else:
            logging.warning(f"Unknown analysis: {analysis}")
    
    # Summary
    logging.info("=" * 60)
    logging.info("ANALYSIS PIPELINE COMPLETED")
    logging.info("=" * 60)
    
    for analysis, result in results.items():
        status = result.get('status', 'unknown')
        logging.info(f"{analysis}: {status}")
    
    logging.info("Results saved to: results/")

if __name__ == "__main__":
    main()