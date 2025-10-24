#!/usr/bin/env python3
"""
Comprehensive cosine similarity analysis using count data.

This script computes cosine similarity for multiple comparison types:
1. Discovery vs Validation TD cohorts (HCP-Dev vs NKI-RS TD, CMI-HBN TD, ADHD-200 TD)
2. Within-condition comparisons (ADHD200 ADHD vs CMI-HBN ADHD, ABIDE ASD vs Stanford ASD)
3. Pooled condition comparisons (Pooled ADHD vs Pooled ASD)
4. Cross-condition comparisons (TD vs ADHD, TD vs ASD)

Usage:
    python scripts/cosine_similarity_analysis.py --analysis_type all --data_dir <count_data_directory>
    python scripts/cosine_similarity_analysis.py --analysis_type discovery_validation --discovery_csv <hcp_dev.csv> --validation_csvs <nki.csv> <cmihbn.csv> <adhd200.csv>
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path: str = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_count_data_paths(config: Dict) -> Dict[str, str]:
    """Get count data file paths from config."""
    return config.get('network_analysis', {}).get('count_data', {})

def load_count_data(file_path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load count data from CSV or Excel file.
    
    Args:
        file_path (str): Path to count data file (CSV or Excel)
        
    Returns:
        Tuple[np.ndarray, List[str]]: Attribution values and region names
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Count data file not found: {file_path}")
    
    # Read file based on extension
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    
    logging.info(f"Columns in {file_path}: {list(df.columns)}")
    
    # Handle different column name formats
    if 'Count' in df.columns and 'Region ID' in df.columns:
        # Use the actual column names from your Excel files
        attributions = df['Count'].values
        regions = df['Region ID'].astype(str).tolist()
        logging.info(f"Using 'Count' and 'Region ID' columns")
    elif 'attribution' in df.columns and 'region' in df.columns:
        # Standard format
        attributions = df['attribution'].values
        regions = df['region'].tolist()
        logging.info(f"Using 'attribution' and 'region' columns")
    else:
        # Try to rename columns if they're in different order
        if len(df.columns) >= 2:
            df.columns = ['attribution', 'region'] + list(df.columns[2:])
            attributions = df['attribution'].values
            regions = df['region'].tolist()
            logging.info(f"Renamed columns to 'attribution' and 'region'")
        else:
            raise ValueError(f"File must have 'Count'/'attribution' and 'Region ID'/'region' columns. Found: {df.columns.tolist()}")
    
    # Add detailed debugging
    logging.info(f"Loaded {len(attributions)} regions from {file_path}")
    logging.info(f"Attribution stats: min={attributions.min():.4f}, max={attributions.max():.4f}, mean={attributions.mean():.4f}, std={attributions.std():.4f}")
    logging.info(f"Non-zero attributions: {(attributions > 0).sum()}/{len(attributions)}")
    logging.info(f"Sample attributions: {attributions[:10].tolist()}")
    logging.info(f"Sample regions: {regions[:10]}")
    logging.info(f"Region types: {[type(r) for r in regions[:5]]}")
    
    return attributions, regions

def align_region_data(discovery_data: Tuple[np.ndarray, List[str]], 
                     validation_data: Tuple[np.ndarray, List[str]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align region data between discovery and validation cohorts.
    
    Args:
        discovery_data: (attributions, regions) for discovery cohort
        validation_data: (attributions, regions) for validation cohort
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Aligned attribution arrays
    """
    disc_attr, disc_regions = discovery_data
    val_attr, val_regions = validation_data
    
    # Add detailed debugging
    logging.info(f"Discovery regions: {len(disc_regions)} total, sample: {disc_regions[:5]}")
    logging.info(f"Validation regions: {len(val_regions)} total, sample: {val_regions[:5]}")
    logging.info(f"Discovery attr stats: min={disc_attr.min():.4f}, max={disc_attr.max():.4f}, mean={disc_attr.mean():.4f}")
    logging.info(f"Validation attr stats: min={val_attr.min():.4f}, max={val_attr.max():.4f}, mean={val_attr.mean():.4f}")
    
    # Find common regions
    disc_regions_set = set(disc_regions)
    val_regions_set = set(val_regions)
    common_regions = disc_regions_set & val_regions_set
    
    logging.info(f"Discovery unique regions: {len(disc_regions_set)}")
    logging.info(f"Validation unique regions: {len(val_regions_set)}")
    logging.info(f"Found {len(common_regions)} common regions between cohorts")
    
    if len(common_regions) == 0:
        # Try to find the issue
        logging.error("No common regions found! This suggests a region ID mismatch.")
        logging.error(f"Discovery region types: {[type(r) for r in disc_regions[:5]]}")
        logging.error(f"Validation region types: {[type(r) for r in val_regions[:5]]}")
        logging.error(f"Sample discovery regions: {disc_regions[:10]}")
        logging.error(f"Sample validation regions: {val_regions[:10]}")
        raise ValueError("No common regions found between discovery and validation cohorts")
    
    # Create aligned arrays
    disc_aligned = []
    val_aligned = []
    
    for region in sorted(common_regions):
        disc_idx = disc_regions.index(region)
        val_idx = val_regions.index(region)
        disc_aligned.append(disc_attr[disc_idx])
        val_aligned.append(val_attr[val_idx])
    
    disc_aligned = np.array(disc_aligned)
    val_aligned = np.array(val_aligned)
    
    logging.info(f"Aligned data stats - Discovery: min={disc_aligned.min():.4f}, max={disc_aligned.max():.4f}, mean={disc_aligned.mean():.4f}")
    logging.info(f"Aligned data stats - Validation: min={val_aligned.min():.4f}, max={val_aligned.max():.4f}, mean={val_aligned.mean():.4f}")
    logging.info(f"Aligned non-zero - Discovery: {(disc_aligned > 0).sum()}/{len(disc_aligned)}")
    logging.info(f"Aligned non-zero - Validation: {(val_aligned > 0).sum()}/{len(val_aligned)}")
    
    return disc_aligned, val_aligned

def compute_cosine_similarity(discovery_attr: np.ndarray, validation_attr: np.ndarray) -> Dict:
    """
    Compute multiple similarity metrics between two attribution vectors.
    
    Args:
        discovery_attr: Discovery cohort attribution vector
        validation_attr: Validation cohort attribution vector
        
    Returns:
        Dict: Dictionary containing various similarity metrics
    """
    # Add debugging information
    logging.info(f"Discovery attr stats: min={discovery_attr.min():.4f}, max={discovery_attr.max():.4f}, mean={discovery_attr.mean():.4f}, std={discovery_attr.std():.4f}")
    logging.info(f"Validation attr stats: min={validation_attr.min():.4f}, max={validation_attr.max():.4f}, mean={validation_attr.mean():.4f}, std={validation_attr.std():.4f}")
    logging.info(f"Discovery non-zero: {(discovery_attr > 0).sum()}/{len(discovery_attr)}")
    logging.info(f"Validation non-zero: {(validation_attr > 0).sum()}/{len(validation_attr)}")
    
    # Reshape for sklearn cosine_similarity
    disc_reshaped = discovery_attr.reshape(1, -1)
    val_reshaped = validation_attr.reshape(1, -1)
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(disc_reshaped, val_reshaped)[0, 0]
    
    # Compute Pearson correlation
    from scipy.stats import pearsonr
    pearson_r, pearson_p = pearsonr(discovery_attr, validation_attr)
    
    # Compute Spearman correlation
    from scipy.stats import spearmanr
    spearman_r, spearman_p = spearmanr(discovery_attr, validation_attr)
    
    # Compute Jaccard similarity (for binary vectors - top features)
    # Convert to binary: top 10% of features
    top_10_percent = int(len(discovery_attr) * 0.1)
    disc_top_indices = set(np.argsort(discovery_attr)[-top_10_percent:])
    val_top_indices = set(np.argsort(validation_attr)[-top_10_percent:])
    jaccard_sim = len(disc_top_indices & val_top_indices) / len(disc_top_indices | val_top_indices)
    
    # Compute overlap of top 50 features
    top_50 = 50
    disc_top50_indices = set(np.argsort(discovery_attr)[-top_50:])
    val_top50_indices = set(np.argsort(validation_attr)[-top_50:])
    overlap_50 = len(disc_top50_indices & val_top50_indices) / top_50
    
    results = {
        'cosine_similarity': cosine_sim,
        'pearson_correlation': pearson_r,
        'pearson_p_value': pearson_p,
        'spearman_correlation': spearman_r,
        'spearman_p_value': spearman_p,
        'jaccard_similarity_top10pct': jaccard_sim,
        'overlap_top50_features': overlap_50
    }
    
    logging.info(f"Similarity metrics: Cosine={cosine_sim:.4f}, Pearson={pearson_r:.4f}, Spearman={spearman_r:.4f}, Jaccard={jaccard_sim:.4f}, Top50_overlap={overlap_50:.4f}")
    
    return results

def analyze_cosine_similarities(discovery_csv: str, validation_csvs: List[str], 
                              validation_names: List[str]) -> Dict:
    """
    Analyze cosine similarities between discovery and validation cohorts.
    
    Args:
        discovery_csv: Path to discovery cohort count data CSV
        validation_csvs: List of paths to validation cohort count data CSVs
        validation_names: List of names for validation cohorts
        
    Returns:
        Dict: Analysis results
    """
    logging.info("Loading discovery cohort data...")
    discovery_data = load_count_data(discovery_csv)
    
    similarities = {}
    detailed_results = {}
    
    for val_csv, val_name in zip(validation_csvs, validation_names):
        logging.info(f"Processing validation cohort: {val_name}")
        
        try:
            # Load validation data
            validation_data = load_count_data(val_csv)
            
            # Align region data
            disc_aligned, val_aligned = align_region_data(discovery_data, validation_data)
            
            # Compute similarity metrics
            similarity_metrics = compute_cosine_similarity(disc_aligned, val_aligned)
            similarities[val_name] = similarity_metrics['cosine_similarity']
            
            # Store detailed results
            detailed_results[val_name] = {
                'similarity_metrics': similarity_metrics,
                'n_common_regions': len(disc_aligned),
                'discovery_mean': np.mean(disc_aligned),
                'validation_mean': np.mean(val_aligned),
                'discovery_std': np.std(disc_aligned),
                'validation_std': np.std(val_aligned)
            }
            
            logging.info(f"Cosine similarity with {val_name}: {similarity_metrics['cosine_similarity']:.4f}")
            
        except Exception as e:
            logging.error(f"Error processing {val_name}: {e}")
            similarities[val_name] = None
            detailed_results[val_name] = {'error': str(e)}
    
    # Compute summary statistics
    valid_similarities = [sim for sim in similarities.values() if sim is not None]
    
    if valid_similarities:
        summary = {
            'mean_similarity': np.mean(valid_similarities),
            'std_similarity': np.std(valid_similarities),
            'min_similarity': np.min(valid_similarities),
            'max_similarity': np.max(valid_similarities),
            'range_similarity': np.max(valid_similarities) - np.min(valid_similarities),
            'n_cohorts': len(valid_similarities)
        }
    else:
        summary = {'error': 'No valid similarities computed'}
    
    results = {
        'discovery_cohort': 'HCP-Dev',
        'validation_cohorts': validation_names,
        'similarities': similarities,
        'detailed_results': detailed_results,
        'summary': summary
    }
    
    return results

def run_discovery_validation_analysis(discovery_csv: str, validation_csvs: List[str], 
                                    validation_names: List[str]) -> Dict:
    """Run discovery vs validation analysis."""
    logging.info("Running discovery vs validation analysis...")
    return analyze_cosine_similarities(discovery_csv, validation_csvs, validation_names)

def run_within_condition_analysis(config: Dict) -> Dict:
    """Run within-condition comparisons (ADHD vs TD within same dataset, ASD vs ASD)."""
    logging.info("Running within-condition analysis...")
    
    # Get count data paths from config
    count_data_paths = get_count_data_paths(config)
    
    results = {}
    
    # ADHD200: ADHD vs TD within same dataset
    adhd200_adhd_csv = count_data_paths.get('adhd200_adhd')
    adhd200_td_csv = count_data_paths.get('adhd200_td')
    
    if adhd200_adhd_csv and adhd200_td_csv and os.path.exists(adhd200_adhd_csv) and os.path.exists(adhd200_td_csv):
        adhd200_results = analyze_cosine_similarities(
            adhd200_adhd_csv, [adhd200_td_csv], ['ADHD200_TD']
        )
        results['ADHD200_within_dataset'] = adhd200_results
    else:
        logging.warning("ADHD200 count data files not found")
        results['ADHD200_within_dataset'] = {'error': 'Files not found'}
    
    # CMI-HBN: ADHD vs TD within same dataset
    cmihbn_adhd_csv = count_data_paths.get('cmihbn_adhd')
    cmihbn_td_csv = count_data_paths.get('cmihbn_td')
    
    if cmihbn_adhd_csv and cmihbn_td_csv and os.path.exists(cmihbn_adhd_csv) and os.path.exists(cmihbn_td_csv):
        cmihbn_results = analyze_cosine_similarities(
            cmihbn_adhd_csv, [cmihbn_td_csv], ['CMIHBN_TD']
        )
        results['CMIHBN_within_dataset'] = cmihbn_results
    else:
        logging.warning("CMI-HBN count data files not found")
        results['CMIHBN_within_dataset'] = {'error': 'Files not found'}
    
    # ASD: ABIDE vs Stanford (both ASD datasets)
    abide_asd_csv = count_data_paths.get('abide_asd')
    stanford_asd_csv = count_data_paths.get('stanford_asd')
    
    if abide_asd_csv and stanford_asd_csv and os.path.exists(abide_asd_csv) and os.path.exists(stanford_asd_csv):
        asd_results = analyze_cosine_similarities(
            abide_asd_csv, [stanford_asd_csv], ['Stanford_ASD']
        )
        results['ASD_within_condition'] = asd_results
    else:
        logging.warning("ASD count data files not found")
        results['ASD_within_condition'] = {'error': 'Files not found'}
    
    return results

def run_pooled_condition_analysis(config: Dict) -> Dict:
    """Run pooled condition comparisons."""
    logging.info("Running pooled condition analysis...")
    
    # Get count data paths from config
    count_data_paths = get_count_data_paths(config)
    
    # Pooled ADHD (average of ADHD200 ADHD and CMI-HBN ADHD)
    adhd200_adhd_csv = count_data_paths.get('adhd200_adhd')
    cmihbn_adhd_csv = count_data_paths.get('cmihbn_adhd')
    
    # Pooled ASD (average of ABIDE ASD and Stanford ASD)
    abide_asd_csv = count_data_paths.get('abide_asd')
    stanford_asd_csv = count_data_paths.get('stanford_asd')
    
    results = {}
    
    # Create pooled ADHD
    if os.path.exists(adhd200_adhd_csv) and os.path.exists(cmihbn_adhd_csv):
        pooled_adhd = create_pooled_data([adhd200_adhd_csv, cmihbn_adhd_csv], "Pooled_ADHD")
        results['pooled_adhd_data'] = pooled_adhd
    else:
        logging.warning("ADHD count data files not found for pooling")
        pooled_adhd = None
    
    # Create pooled ASD
    if os.path.exists(abide_asd_csv) and os.path.exists(stanford_asd_csv):
        pooled_asd = create_pooled_data([abide_asd_csv, stanford_asd_csv], "Pooled_ASD")
        results['pooled_asd_data'] = pooled_asd
    else:
        logging.warning("ASD count data files not found for pooling")
        pooled_asd = None
    
    # Compare pooled ADHD vs pooled ASD
    if pooled_adhd is not None and pooled_asd is not None:
        pooled_comparison = analyze_cosine_similarities(
            pooled_adhd, [pooled_asd], ['Pooled_ASD']
        )
        results['pooled_adhd_vs_asd'] = pooled_comparison
    else:
        results['pooled_adhd_vs_asd'] = {'error': 'Pooled data not available'}
    
    return results

def run_cross_condition_analysis(config: Dict) -> Dict:
    """Run cross-condition comparisons (TD vs ADHD, TD vs ASD)."""
    logging.info("Running cross-condition analysis...")
    
    # Get count data paths from config
    count_data_paths = get_count_data_paths(config)
    
    # TD cohorts
    nki_td_csv = count_data_paths.get('nki')
    cmihbn_td_csv = count_data_paths.get('cmihbn_td')
    adhd200_td_csv = count_data_paths.get('adhd200_td')
    
    # ADHD cohorts
    adhd200_adhd_csv = count_data_paths.get('adhd200_adhd')
    cmihbn_adhd_csv = count_data_paths.get('cmihbn_adhd')
    
    # ASD cohorts
    abide_asd_csv = count_data_paths.get('abide_asd')
    stanford_asd_csv = count_data_paths.get('stanford_asd')
    
    results = {}
    
    # Create pooled TD
    td_files = [f for f in [nki_td_csv, cmihbn_td_csv, adhd200_td_csv] if os.path.exists(f)]
    if len(td_files) >= 2:
        pooled_td = create_pooled_data(td_files, "Pooled_TD")
        results['pooled_td_data'] = pooled_td
    else:
        logging.warning("Insufficient TD count data files for pooling")
        pooled_td = None
    
    # Create pooled ADHD
    adhd_files = [f for f in [adhd200_adhd_csv, cmihbn_adhd_csv] if os.path.exists(f)]
    if len(adhd_files) >= 1:
        pooled_adhd = create_pooled_data(adhd_files, "Pooled_ADHD")
        results['pooled_adhd_data'] = pooled_adhd
    else:
        logging.warning("ADHD count data files not found")
        pooled_adhd = None
    
    # Create pooled ASD
    asd_files = [f for f in [abide_asd_csv, stanford_asd_csv] if os.path.exists(f)]
    if len(asd_files) >= 1:
        pooled_asd = create_pooled_data(asd_files, "Pooled_ASD")
        results['pooled_asd_data'] = pooled_asd
    else:
        logging.warning("ASD count data files not found")
        pooled_asd = None
    
    # TD vs ADHD
    if pooled_td is not None and pooled_adhd is not None:
        td_vs_adhd = analyze_cosine_similarities(
            pooled_td, [pooled_adhd], ['Pooled_ADHD']
        )
        results['td_vs_adhd'] = td_vs_adhd
    else:
        results['td_vs_adhd'] = {'error': 'Pooled data not available'}
    
    # TD vs ASD
    if pooled_td is not None and pooled_asd is not None:
        td_vs_asd = analyze_cosine_similarities(
            pooled_td, [pooled_asd], ['Pooled_ASD']
        )
        results['td_vs_asd'] = td_vs_asd
    else:
        results['td_vs_asd'] = {'error': 'Pooled data not available'}
    
    return results

def create_pooled_data(file_paths: List[str], name: str) -> str:
    """Create pooled data by averaging across multiple files (CSV or Excel)."""
    logging.info(f"Creating pooled data for {name} from {len(file_paths)} files...")
    
    all_data = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            logging.warning(f"File not found: {file_path}")
            continue
        
        # Read file based on extension
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
            # Handle Excel column names
            if 'Count' in df.columns and 'Region ID' in df.columns:
                df = df.rename(columns={'Count': 'attribution', 'Region ID': 'region'})
        else:
            df = pd.read_csv(file_path)
        
        all_data.append(df)
    
    if not all_data:
        raise ValueError(f"No valid data files found for {name}")
    
    # Align regions across all datasets
    common_regions = set(all_data[0]['region'].astype(str))
    for df in all_data[1:]:
        common_regions = common_regions & set(df['region'].astype(str))
    
    logging.info(f"Found {len(common_regions)} common regions for pooling")
    
    # Create pooled data
    pooled_attributions = []
    for region in sorted(common_regions):
        region_attrs = []
        for df in all_data:
            region_attr = df[df['region'].astype(str) == region]['attribution'].iloc[0]
            region_attrs.append(region_attr)
        pooled_attr = np.mean(region_attrs)
        pooled_attributions.append(pooled_attr)
    
    # Create pooled DataFrame
    pooled_df = pd.DataFrame({
        'attribution': pooled_attributions,
        'region': sorted(common_regions)
    })
    
    # Save pooled data
    pooled_file = f"pooled_{name.lower()}_count_data.csv"
    pooled_df.to_csv(pooled_file, index=False)
    logging.info(f"Pooled data saved to: {pooled_file}")
    
    return pooled_file

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Comprehensive cosine similarity analysis")
    parser.add_argument("--analysis_type", type=str, 
                       choices=['all', 'discovery_validation', 'within_condition', 'pooled_condition', 'cross_condition'],
                       default='all', help="Type of analysis to run")
    parser.add_argument("--config", type=str, default="/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--data_dir", type=str, help="Directory containing count data CSV files (deprecated, use config)")
    parser.add_argument("--discovery_csv", type=str, help="Path to discovery cohort count data CSV")
    parser.add_argument("--nki_csv", type=str, help="Path to NKI-RS TD count data CSV")
    parser.add_argument("--cmihbn_csv", type=str, help="Path to CMI-HBN TD count data CSV")
    parser.add_argument("--adhd200_csv", type=str, help="Path to ADHD-200 TD count data CSV")
    parser.add_argument("--output_dir", type=str, default="results/cosine_similarity",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # Run analyses based on type
    if args.analysis_type in ['all', 'discovery_validation']:
        # Get count data paths from config
        count_data_paths = get_count_data_paths(config)
        
        # Use config paths or command line arguments
        discovery_csv = args.discovery_csv or count_data_paths.get('dev')
        nki_csv = args.nki_csv or count_data_paths.get('nki')
        cmihbn_csv = args.cmihbn_csv or count_data_paths.get('cmihbn_td')
        adhd200_csv = args.adhd200_csv or count_data_paths.get('adhd200_td')
        
        if discovery_csv and nki_csv and cmihbn_csv and adhd200_csv and all(os.path.exists(f) for f in [discovery_csv, nki_csv, cmihbn_csv, adhd200_csv]):
            validation_csvs = [nki_csv, cmihbn_csv, adhd200_csv]
            validation_names = ['NKI-RS_TD', 'CMI-HBN_TD', 'ADHD-200_TD']
            all_results['discovery_validation'] = run_discovery_validation_analysis(
                discovery_csv, validation_csvs, validation_names
            )
        else:
            logging.warning("Discovery validation analysis skipped - missing required files")
    
    if args.analysis_type in ['all', 'within_condition']:
        all_results['within_condition'] = run_within_condition_analysis(config)
    
    if args.analysis_type in ['all', 'pooled_condition']:
        all_results['pooled_condition'] = run_pooled_condition_analysis(config)
    
    if args.analysis_type in ['all', 'cross_condition']:
        all_results['cross_condition'] = run_cross_condition_analysis(config)
    
    # Save results
    import json
    results_file = output_dir / "comprehensive_cosine_similarity_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*80)
    print("COMPREHENSIVE COSINE SIMILARITY ANALYSIS RESULTS")
    print("="*80)
    
    for analysis_type, results in all_results.items():
        print(f"\n{analysis_type.upper().replace('_', ' ')} ANALYSIS:")
        print("-" * 40)
        
        if isinstance(results, dict) and 'error' not in results:
            if 'summary' in results and 'error' not in results['summary']:
                summary = results['summary']
                print(f"  Mean Similarity: {summary['mean_similarity']:.4f}")
                print(f"  Range: {summary['min_similarity']:.4f} - {summary['max_similarity']:.4f}")
            elif 'similarities' in results:
                for cohort, sim in results['similarities'].items():
                    if sim is not None:
                        print(f"  {cohort}: {sim:.4f}")
        else:
            print(f"  Error: {results.get('error', 'Unknown error')}")
    
    print(f"\nDetailed results saved to: {results_file}")
    logging.info("Comprehensive cosine similarity analysis completed!")

if __name__ == "__main__":
    main()
