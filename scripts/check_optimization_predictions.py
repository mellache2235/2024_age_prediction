#!/usr/bin/env python3
"""
Check integrity of optimization predictions.

Verifies that:
1. Predictions exist for all measures
2. Actual vs predicted correlations match reported values
3. No data integrity issues (NaN, inf, duplicates)
4. Model performance is consistent

Usage:
    python check_optimization_predictions.py --cohort stanford_asd
    python check_optimization_predictions.py --all

Author: Brain-Behavior Optimization Team
Date: 2024
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import argparse
import warnings
warnings.filterwarnings('ignore')

# Add to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from logging_utils import (print_section_header, print_step, print_success, 
                           print_warning, print_error, print_info, print_completion)

# ============================================================================
# COHORT CONFIGURATIONS
# ============================================================================

COHORTS = {
    'abide_asd': {
        'name': 'ABIDE ASD',
        'results_dir': '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/abide_asd_optimized'
    },
    'stanford_asd': {
        'name': 'Stanford ASD',
        'results_dir': '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/stanford_asd_optimized'
    },
    'adhd200_td': {
        'name': 'ADHD200 TD',
        'results_dir': '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/adhd200_td_optimized'
    },
    'adhd200_adhd': {
        'name': 'ADHD200 ADHD',
        'results_dir': '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/adhd200_adhd_optimized'
    },
    'cmihbn_td': {
        'name': 'CMI-HBN TD',
        'results_dir': '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/cmihbn_td_optimized'
    },
    'cmihbn_adhd': {
        'name': 'CMI-HBN ADHD',
        'results_dir': '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/cmihbn_adhd_optimized'
    },
    'nki_rs_td': {
        'name': 'NKI-RS TD',
        'results_dir': '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/nki_rs_td_optimized'
    }
}

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def check_data_integrity(data, name):
    """Check for NaN, Inf, and other data issues."""
    issues = []
    
    # Check for NaN
    nan_count = np.isnan(data).sum()
    if nan_count > 0:
        issues.append(f"  ⚠ {nan_count} NaN values")
    
    # Check for Inf
    inf_count = np.isinf(data).sum()
    if inf_count > 0:
        issues.append(f"  ⚠ {inf_count} Inf values")
    
    # Check for constant values
    if np.std(data[~np.isnan(data)]) == 0:
        issues.append(f"  ⚠ Constant values (no variance)")
    
    if issues:
        print_warning(f"{name} has issues:")
        for issue in issues:
            print(issue)
        return False
    else:
        print_success(f"{name}: No integrity issues")
        return True


def validate_predictions(pred_file, summary_row, measure_name):
    """
    Validate a single prediction file.
    
    Args:
        pred_file: Path to predictions CSV
        summary_row: Row from optimization_summary.csv
        measure_name: Name of behavioral measure
    
    Returns:
        dict with validation results
    """
    print()
    print_step(f"Validating: {measure_name}", "")
    
    results = {
        'measure': measure_name,
        'file_exists': False,
        'data_integrity': False,
        'correlation_match': False,
        'reported_rho': None,
        'computed_rho': None,
        'rho_difference': None,
        'n_subjects': None,
        'issues': []
    }
    
    # Check if file exists
    if not pred_file.exists():
        print_error(f"Prediction file not found: {pred_file.name}")
        results['issues'].append("File not found")
        return results
    
    results['file_exists'] = True
    print_success(f"Found: {pred_file.name}")
    
    try:
        # Load predictions
        pred_df = pd.read_csv(pred_file)
        
        # Check required columns
        required_cols = ['Actual', 'Predicted']
        missing_cols = [col for col in required_cols if col not in pred_df.columns]
        if missing_cols:
            print_error(f"Missing columns: {missing_cols}")
            results['issues'].append(f"Missing columns: {missing_cols}")
            return results
        
        actual = pred_df['Actual'].values
        predicted = pred_df['Predicted'].values
        
        results['n_subjects'] = len(actual)
        print_info(f"N subjects: {len(actual)}", 0)
        
        # Check data integrity
        actual_ok = check_data_integrity(actual, "Actual values")
        predicted_ok = check_data_integrity(predicted, "Predicted values")
        results['data_integrity'] = actual_ok and predicted_ok
        
        if not results['data_integrity']:
            results['issues'].append("Data integrity issues detected")
            return results
        
        # Compute correlation
        computed_rho, computed_p = spearmanr(actual, predicted)
        results['computed_rho'] = computed_rho
        
        # Get reported correlation from summary
        if summary_row is not None and 'Final_Spearman' in summary_row:
            reported_rho = summary_row['Final_Spearman']
            results['reported_rho'] = reported_rho
            
            # Compare
            rho_diff = abs(computed_rho - reported_rho)
            results['rho_difference'] = rho_diff
            
            # Allow small numerical differences (1e-3)
            if rho_diff < 0.001:
                results['correlation_match'] = True
                print_success(f"Correlation match: ρ = {computed_rho:.4f} (reported: {reported_rho:.4f})")
            else:
                results['correlation_match'] = False
                print_warning(f"Correlation mismatch!")
                print(f"    Computed:  ρ = {computed_rho:.4f}")
                print(f"    Reported:  ρ = {reported_rho:.4f}")
                print(f"    Difference: {rho_diff:.4f}")
                results['issues'].append(f"Correlation mismatch: diff = {rho_diff:.4f}")
        else:
            print_warning("No reported correlation found in summary")
            print_info(f"Computed: ρ = {computed_rho:.4f}, p = {computed_p:.4f}", 0)
        
        # Check residuals if available
        if 'Residual' in pred_df.columns:
            residuals = pred_df['Residual'].values
            computed_residuals = actual - predicted
            residual_match = np.allclose(residuals, computed_residuals, atol=1e-6)
            
            if residual_match:
                print_success("Residuals correctly computed")
            else:
                print_warning("Residual computation mismatch")
                results['issues'].append("Residual mismatch")
        
        # Summary
        if results['correlation_match'] and results['data_integrity']:
            print_success(f"✓ All checks passed for {measure_name}")
        
    except Exception as e:
        print_error(f"Error validating {measure_name}: {str(e)}")
        results['issues'].append(str(e))
    
    return results


# ============================================================================
# MAIN VALIDATION
# ============================================================================

def validate_cohort(cohort_key):
    """Validate all predictions for a cohort."""
    config = COHORTS[cohort_key]
    
    print_section_header(f"PREDICTION INTEGRITY CHECK - {config['name'].upper()}")
    
    results_dir = Path(config['results_dir'])
    
    if not results_dir.exists():
        print_error(f"Results directory not found: {results_dir}")
        return False
    
    # Load optimization summary
    summary_path = results_dir / 'optimization_summary.csv'
    if not summary_path.exists():
        print_error(f"Optimization summary not found: {summary_path}")
        return False
    
    summary_df = pd.read_csv(summary_path)
    print_info(f"Found optimization summary with {len(summary_df)} measures")
    
    # Find all prediction files
    pred_files = list(results_dir.glob("predictions_*.csv"))
    print_info(f"Found {len(pred_files)} prediction files")
    print()
    
    if len(pred_files) == 0:
        print_warning("No prediction files found!")
        return False
    
    # Validate each prediction file
    all_results = []
    
    for pred_file in sorted(pred_files):
        # Extract measure name from filename
        # Format: predictions_{measure}_{method}.csv
        filename = pred_file.stem
        parts = filename.replace('predictions_', '').split('_')
        
        # Try to match with summary
        summary_row = None
        measure_name = None
        
        # Try to find matching row in summary
        for idx, row in summary_df.iterrows():
            safe_measure = row['Measure'].replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
            if safe_measure in filename:
                summary_row = row
                measure_name = row['Measure']
                break
        
        if measure_name is None:
            measure_name = filename.replace('predictions_', '')
        
        # Validate
        result = validate_predictions(pred_file, summary_row, measure_name)
        all_results.append(result)
    
    # Summary statistics
    print()
    print_section_header("VALIDATION SUMMARY")
    
    total = len(all_results)
    files_exist = sum(1 for r in all_results if r['file_exists'])
    data_ok = sum(1 for r in all_results if r['data_integrity'])
    corr_match = sum(1 for r in all_results if r['correlation_match'])
    has_issues = sum(1 for r in all_results if len(r['issues']) > 0)
    
    print_info(f"Total prediction files: {total}")
    print_info(f"  Files found: {files_exist}/{total}")
    print_info(f"  Data integrity OK: {data_ok}/{total}")
    print_info(f"  Correlation match: {corr_match}/{total}")
    print_info(f"  Files with issues: {has_issues}/{total}")
    
    if has_issues > 0:
        print()
        print_warning("Issues found in the following measures:")
        for r in all_results:
            if len(r['issues']) > 0:
                print(f"  • {r['measure']}")
                for issue in r['issues']:
                    print(f"      - {issue}")
    
    # Save validation report
    print()
    print_step("Saving validation report", "")
    
    report_df = pd.DataFrame([
        {
            'Measure': r['measure'],
            'File_Exists': r['file_exists'],
            'Data_Integrity_OK': r['data_integrity'],
            'Correlation_Match': r['correlation_match'],
            'Reported_Rho': r['reported_rho'],
            'Computed_Rho': r['computed_rho'],
            'Rho_Difference': r['rho_difference'],
            'N_Subjects': r['n_subjects'],
            'Issues': '; '.join(r['issues']) if r['issues'] else 'None'
        }
        for r in all_results
    ])
    
    report_path = results_dir / 'validation_report.csv'
    report_df.to_csv(report_path, index=False)
    print_success(f"Validation report saved: {report_path.name}")
    
    # Overall success
    all_ok = (has_issues == 0) and (files_exist == total) and (data_ok == total)
    
    print()
    if all_ok:
        print_completion(f"✓ All predictions validated successfully for {config['name']}!")
    else:
        print_warning(f"Some validation issues found for {config['name']}")
    
    return all_ok


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check integrity of optimization predictions"
    )
    parser.add_argument(
        '--cohort', '-c',
        choices=list(COHORTS.keys()),
        help="Cohort to validate"
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help="Validate all cohorts"
    )
    
    args = parser.parse_args()
    
    if not args.cohort and not args.all:
        parser.error("Must specify either --cohort or --all")
    
    # Determine which cohorts to process
    if args.all:
        cohorts_to_process = list(COHORTS.keys())
    else:
        cohorts_to_process = [args.cohort]
    
    # Process each cohort
    results = {}
    for cohort_key in cohorts_to_process:
        success = validate_cohort(cohort_key)
        results[cohort_key] = success
        print("\n" + "="*100 + "\n")
    
    # Overall summary
    print_section_header("OVERALL SUMMARY")
    for cohort_key, success in results.items():
        status = "✅ VALID" if success else "⚠️  ISSUES FOUND"
        print(f"  {COHORTS[cohort_key]['name']:.<50} {status}")
    print()


if __name__ == "__main__":
    main()
