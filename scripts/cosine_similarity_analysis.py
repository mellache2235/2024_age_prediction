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
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_count_data(csv_file: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load count data from CSV file.
    
    Args:
        csv_file (str): Path to count data CSV file
        
    Returns:
        Tuple[np.ndarray, List[str]]: Attribution values and region names
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Count data file not found: {csv_file}")
    
    df = pd.read_csv(csv_file)
    
    # Ensure we have the expected columns
    if 'attribution' not in df.columns or 'region' not in df.columns:
        # Try to rename columns if they're in different order
        if len(df.columns) >= 2:
            df.columns = ['attribution', 'region'] + list(df.columns[2:])
        else:
            raise ValueError(f"CSV file must have 'attribution' and 'region' columns. Found: {df.columns.tolist()}")
    
    attributions = df['attribution'].values
    regions = df['region'].tolist()
    
    logging.info(f"Loaded {len(attributions)} regions from {csv_file}")
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
    
    # Find common regions
    common_regions = set(disc_regions) & set(val_regions)
    logging.info(f"Found {len(common_regions)} common regions between cohorts")
    
    if len(common_regions) == 0:
        raise ValueError("No common regions found between discovery and validation cohorts")
    
    # Create aligned arrays
    disc_aligned = []
    val_aligned = []
    
    for region in sorted(common_regions):
        disc_idx = disc_regions.index(region)
        val_idx = val_regions.index(region)
        disc_aligned.append(disc_attr[disc_idx])
        val_aligned.append(val_attr[val_idx])
    
    return np.array(disc_aligned), np.array(val_aligned)

def compute_cosine_similarity(discovery_attr: np.ndarray, validation_attr: np.ndarray) -> float:
    """
    Compute cosine similarity between two attribution vectors.
    
    Args:
        discovery_attr: Discovery cohort attribution vector
        validation_attr: Validation cohort attribution vector
        
    Returns:
        float: Cosine similarity score
    """
    # Reshape for sklearn cosine_similarity
    disc_reshaped = discovery_attr.reshape(1, -1)
    val_reshaped = validation_attr.reshape(1, -1)
    
    # Compute cosine similarity
    similarity = cosine_similarity(disc_reshaped, val_reshaped)[0, 0]
    
    return similarity

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
            
            # Compute cosine similarity
            similarity = compute_cosine_similarity(disc_aligned, val_aligned)
            similarities[val_name] = similarity
            
            # Store detailed results
            detailed_results[val_name] = {
                'similarity': similarity,
                'n_common_regions': len(disc_aligned),
                'discovery_mean': np.mean(disc_aligned),
                'validation_mean': np.mean(val_aligned),
                'discovery_std': np.std(disc_aligned),
                'validation_std': np.std(val_aligned)
            }
            
            logging.info(f"Cosine similarity with {val_name}: {similarity:.4f}")
            
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

def run_within_condition_analysis(data_dir: str) -> Dict:
    """Run within-condition comparisons."""
    logging.info("Running within-condition analysis...")
    
    # ADHD comparisons
    adhd200_adhd_csv = os.path.join(data_dir, "adhd200_adhd_count_data.csv")
    cmihbn_adhd_csv = os.path.join(data_dir, "cmihbn_adhd_count_data.csv")
    
    # ASD comparisons
    abide_asd_csv = os.path.join(data_dir, "abide_asd_count_data.csv")
    stanford_asd_csv = os.path.join(data_dir, "stanford_asd_count_data.csv")
    
    results = {}
    
    # ADHD within-condition
    if os.path.exists(adhd200_adhd_csv) and os.path.exists(cmihbn_adhd_csv):
        adhd_results = analyze_cosine_similarities(
            adhd200_adhd_csv, [cmihbn_adhd_csv], ['CMI-HBN_ADHD']
        )
        results['ADHD_within_condition'] = adhd_results
    else:
        logging.warning("ADHD count data files not found")
        results['ADHD_within_condition'] = {'error': 'Files not found'}
    
    # ASD within-condition
    if os.path.exists(abide_asd_csv) and os.path.exists(stanford_asd_csv):
        asd_results = analyze_cosine_similarities(
            abide_asd_csv, [stanford_asd_csv], ['Stanford_ASD']
        )
        results['ASD_within_condition'] = asd_results
    else:
        logging.warning("ASD count data files not found")
        results['ASD_within_condition'] = {'error': 'Files not found'}
    
    return results

def run_pooled_condition_analysis(data_dir: str) -> Dict:
    """Run pooled condition comparisons."""
    logging.info("Running pooled condition analysis...")
    
    # Pooled ADHD (average of ADHD200 ADHD and CMI-HBN ADHD)
    adhd200_adhd_csv = os.path.join(data_dir, "adhd200_adhd_count_data.csv")
    cmihbn_adhd_csv = os.path.join(data_dir, "cmihbn_adhd_count_data.csv")
    
    # Pooled ASD (average of ABIDE ASD and Stanford ASD)
    abide_asd_csv = os.path.join(data_dir, "abide_asd_count_data.csv")
    stanford_asd_csv = os.path.join(data_dir, "stanford_asd_count_data.csv")
    
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

def run_cross_condition_analysis(data_dir: str) -> Dict:
    """Run cross-condition comparisons (TD vs ADHD, TD vs ASD)."""
    logging.info("Running cross-condition analysis...")
    
    # TD cohorts
    nki_td_csv = os.path.join(data_dir, "nki_rs_td_count_data.csv")
    cmihbn_td_csv = os.path.join(data_dir, "cmihbn_td_count_data.csv")
    adhd200_td_csv = os.path.join(data_dir, "adhd200_td_count_data.csv")
    
    # ADHD cohorts
    adhd200_adhd_csv = os.path.join(data_dir, "adhd200_adhd_count_data.csv")
    cmihbn_adhd_csv = os.path.join(data_dir, "cmihbn_adhd_count_data.csv")
    
    # ASD cohorts
    abide_asd_csv = os.path.join(data_dir, "abide_asd_count_data.csv")
    stanford_asd_csv = os.path.join(data_dir, "stanford_asd_count_data.csv")
    
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

def create_pooled_data(csv_files: List[str], name: str) -> str:
    """Create pooled data by averaging across multiple CSV files."""
    logging.info(f"Creating pooled data for {name} from {len(csv_files)} files...")
    
    all_data = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        all_data.append(df)
    
    # Align regions across all datasets
    common_regions = set(all_data[0]['region'])
    for df in all_data[1:]:
        common_regions = common_regions & set(df['region'])
    
    logging.info(f"Found {len(common_regions)} common regions for pooling")
    
    # Create pooled data
    pooled_attributions = []
    for region in sorted(common_regions):
        region_attrs = []
        for df in all_data:
            region_attr = df[df['region'] == region]['attribution'].iloc[0]
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
    parser.add_argument("--data_dir", type=str, help="Directory containing count data CSV files")
    parser.add_argument("--discovery_csv", type=str, help="Path to discovery cohort count data CSV")
    parser.add_argument("--nki_csv", type=str, help="Path to NKI-RS TD count data CSV")
    parser.add_argument("--cmihbn_csv", type=str, help="Path to CMI-HBN TD count data CSV")
    parser.add_argument("--adhd200_csv", type=str, help="Path to ADHD-200 TD count data CSV")
    parser.add_argument("--output_dir", type=str, default="results/cosine_similarity",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # Run analyses based on type
    if args.analysis_type in ['all', 'discovery_validation']:
        if args.discovery_csv and args.nki_csv and args.cmihbn_csv and args.adhd200_csv:
            validation_csvs = [args.nki_csv, args.cmihbn_csv, args.adhd200_csv]
            validation_names = ['NKI-RS_TD', 'CMI-HBN_TD', 'ADHD-200_TD']
            all_results['discovery_validation'] = run_discovery_validation_analysis(
                args.discovery_csv, validation_csvs, validation_names
            )
        else:
            logging.warning("Discovery validation analysis skipped - missing required CSV files")
    
    if args.analysis_type in ['all', 'within_condition']:
        if args.data_dir:
            all_results['within_condition'] = run_within_condition_analysis(args.data_dir)
        else:
            logging.warning("Within condition analysis skipped - data_dir not provided")
    
    if args.analysis_type in ['all', 'pooled_condition']:
        if args.data_dir:
            all_results['pooled_condition'] = run_pooled_condition_analysis(args.data_dir)
        else:
            logging.warning("Pooled condition analysis skipped - data_dir not provided")
    
    if args.analysis_type in ['all', 'cross_condition']:
        if args.data_dir:
            all_results['cross_condition'] = run_cross_condition_analysis(args.data_dir)
        else:
            logging.warning("Cross condition analysis skipped - data_dir not provided")
    
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
