#!/usr/bin/env python3
"""
Analyze shared TD regions to find regions with counts > 450.

This script reads the TD count data files and identifies regions that have
minimum counts greater than 450 across all TD cohorts.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Try to import tabulate for pretty printing
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    logging.warning("tabulate not found. Install with: pip install tabulate")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def analyze_shared_td_regions(count_data_dir: str, min_count_threshold: int = 450):
    """
    Analyze shared TD regions and find those with minimum counts > threshold.
    
    Args:
        count_data_dir (str): Directory containing count data CSV files
        min_count_threshold (int): Minimum count threshold (default: 450)
    """
    # TD datasets (core 4 datasets)
    td_datasets = ['dev', 'nki', 'adhd200_td', 'cmihbn_td']
    
    # Load count data for each TD dataset
    all_regions = {}
    
    for dataset in td_datasets:
        csv_path = os.path.join(count_data_dir, f"{dataset}_count_data.csv")
        
        if not os.path.exists(csv_path):
            logging.warning(f"Count data file not found: {csv_path}")
            continue
        
        logging.info(f"Loading {dataset} count data...")
        count_data = pd.read_csv(csv_path)
        
        # Get top 50 regions
        top_regions = count_data.nlargest(50, 'Count')
        
        for _, row in top_regions.iterrows():
            # Use Region ID as the key for matching across datasets
            region_id = row.get('Region ID', row.get('(ID) Region Label', 'Unknown'))
            
            if region_id not in all_regions:
                all_regions[region_id] = {
                    'counts': [],
                    'datasets': [],
                    'brain_region': row.get('Brain Regions', row.get('Gyrus', 'Unknown')),
                    'subdivision': row.get('Subdivision', row.get('Region Alias', 'Unknown')),
                    'region_label': row.get('(ID) Region Label', region_id)
                }
            
            all_regions[region_id]['counts'].append(row['Count'])
            all_regions[region_id]['datasets'].append(dataset)
    
    # Filter regions that appear in all 4 datasets
    shared_regions = {region_id: data for region_id, data in all_regions.items() 
                     if len(data['datasets']) == 4}
    
    logging.info(f"\nTotal regions appearing in all 4 TD datasets: {len(shared_regions)}")
    
    # Calculate minimum counts and filter by threshold
    high_count_regions = []
    
    for region_id, data in shared_regions.items():
        min_count = np.min(data['counts'])
        max_count = np.max(data['counts'])
        mean_count = np.mean(data['counts'])
        
        if min_count > min_count_threshold:
            high_count_regions.append({
                'Region ID': region_id,
                'Brain Regions': data['brain_region'],
                'Subdivision': data['subdivision'],
                '(ID) Region Label': data['region_label'],
                'Min Count': int(min_count),
                'Max Count': int(max_count),
                'Mean Count': round(mean_count, 1),
                'Datasets': ', '.join(data['datasets']),
                'Individual Counts': ', '.join([f"{d}:{int(c)}" for d, c in zip(data['datasets'], data['counts'])])
            })
    
    # Sort by minimum count (descending)
    high_count_regions = sorted(high_count_regions, key=lambda x: x['Min Count'], reverse=True)
    
    # Create DataFrame
    results_df = pd.DataFrame(high_count_regions)
    
    # Print results
    print("\n" + "="*100)
    print(f"  SHARED TD REGIONS WITH MINIMUM COUNT > {min_count_threshold}")
    print("="*100 + "\n")
    
    if len(results_df) > 0:
        print(f"✓ Found {len(results_df)} regions with minimum count > {min_count_threshold}\n")
        
        # Print summary table using tabulate if available
        if HAS_TABULATE:
            # Summary table
            summary_data = []
            for _, row in results_df.iterrows():
                summary_data.append([
                    row['Brain Regions'][:35] + '...' if len(str(row['Brain Regions'])) > 35 else row['Brain Regions'],
                    str(row['Subdivision'])[:12] + '...' if len(str(row['Subdivision'])) > 12 else str(row['Subdivision']),
                    row['Min Count'],
                    row['Mean Count'],
                    row['Max Count']
                ])
            
            print(tabulate(summary_data, 
                          headers=['Brain Region', 'Subdivision', 'Min', 'Mean', 'Max'],
                          tablefmt='grid',
                          numalign='right'))
            
            # Detailed breakdown
            print("\n\n" + "="*100)
            print("  DETAILED BREAKDOWN")
            print("="*100 + "\n")
            
            for i, (_, row) in enumerate(results_df.iterrows(), 1):
                print(f"\n{i}. {row['(ID) Region Label']}")
                print("   " + "-"*80)
                
                detail_data = [
                    ['Brain Region', row['Brain Regions']],
                    ['Subdivision', row['Subdivision']],
                    ['Minimum Count', row['Min Count']],
                    ['Mean Count', row['Mean Count']],
                    ['Max Count', row['Max Count']],
                ]
                
                print(tabulate(detail_data, tablefmt='plain', colalign=('left', 'left')))
                
                # Individual counts per dataset
                counts_by_dataset = []
                for dataset, count in zip(['HCP-Dev', 'NKI', 'ADHD200 TD', 'CMI-HBN TD'], 
                                         [c.split(':')[1] for c in row['Individual Counts'].split(', ')]):
                    counts_by_dataset.append([dataset, count])
                
                print("\n   Individual Counts:")
                print(tabulate(counts_by_dataset, 
                              headers=['Dataset', 'Count'],
                              tablefmt='simple',
                              numalign='right'))
        else:
            # Fallback to simple formatting
            print("\n" + "="*120)
            print(f"{'Brain Region':<40} {'Subdivision':<15} {'Min Count':<12} {'Mean Count':<12} {'Max Count':<12}")
            print("="*120)
            
            for _, row in results_df.iterrows():
                print(f"{row['Brain Regions']:<40} {str(row['Subdivision']):<15} {row['Min Count']:<12} {row['Mean Count']:<12} {row['Max Count']:<12}")
            
            print("="*120)
            
            # Print detailed breakdown
            print("\n\nDETAILED BREAKDOWN:")
            print("="*120)
            for _, row in results_df.iterrows():
                print(f"\n{row['(ID) Region Label']}")
                print(f"  Brain Region: {row['Brain Regions']}")
                print(f"  Subdivision: {row['Subdivision']}")
                print(f"  Minimum Count: {row['Min Count']}")
                print(f"  Individual Counts: {row['Individual Counts']}")
            print("="*120)
        
        # Save to CSV
        output_path = os.path.join(count_data_dir, f"td_shared_regions_min_count_gt_{min_count_threshold}.csv")
        results_df.to_csv(output_path, index=False)
        
        # Print summary statistics
        print("\n\n" + "="*100)
        print("  SUMMARY STATISTICS")
        print("="*100 + "\n")
        
        summary_stats = [
            ['Total Regions Found', len(results_df)],
            ['Highest Min Count', results_df['Min Count'].max()],
            ['Lowest Min Count', results_df['Min Count'].min()],
            ['Average Min Count', round(results_df['Min Count'].mean(), 1)],
            ['Median Min Count', results_df['Min Count'].median()],
        ]
        
        if HAS_TABULATE:
            print(tabulate(summary_stats, tablefmt='simple', colalign=('left', 'right')))
        else:
            for stat, value in summary_stats:
                print(f"{stat:<25} {value:>10}")
        
        print("\n" + "="*100)
        print(f"✓ Results saved to: {output_path}")
        print("="*100 + "\n")
        
    else:
        print(f"✗ No regions found with minimum count > {min_count_threshold}\n")
    
    return results_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze shared TD regions to find those with high minimum counts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find TD regions with minimum count > 450
  python analyze_td_shared_regions.py --count_data_dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/count_data --threshold 450
  
  # Find TD regions with minimum count > 400
  python analyze_td_shared_regions.py --count_data_dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/count_data --threshold 400
        """
    )
    
    parser.add_argument('--count_data_dir', type=str, 
                       default='/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/count_data',
                       help='Directory containing count data CSV files')
    parser.add_argument('--threshold', type=int, default=450,
                       help='Minimum count threshold (default: 450)')
    
    args = parser.parse_args()
    
    # Run analysis
    results = analyze_shared_td_regions(args.count_data_dir, args.threshold)
    
    if len(results) > 0:
        logging.info(f"\n✅ Analysis complete! Found {len(results)} regions with minimum count > {args.threshold}")
    else:
        logging.info(f"\n⚠️  No regions found with minimum count > {args.threshold}")

