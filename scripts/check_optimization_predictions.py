#!/usr/bin/env python3
"""
Check optimization predictions for integrity.

This script reads the predictions CSV files created during optimization
and provides diagnostic information to verify models are working correctly.

Usage:
    python check_optimization_predictions.py --cohort stanford_asd
    python check_optimization_predictions.py --cohort abide_asd --measure ados_total
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Add to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from logging_utils import print_section_header, print_step, print_success, print_warning, print_info, print_error

BASE_DIR = Path("/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior")


def check_predictions(cohort_dir, measure_name=None):
    """Check prediction files for integrity issues."""
    
    # Find all prediction files
    if measure_name:
        safe_name = measure_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
        pred_files = [cohort_dir / f"predictions_{safe_name}.csv"]
    else:
        pred_files = list(cohort_dir.glob("predictions_*.csv"))
    
    if not pred_files:
        print_error("No prediction files found!")
        return
    
    print_info(f"Found {len(pred_files)} prediction files", 0)
    print()
    
    issues_found = []
    
    for pred_file in pred_files:
        measure = pred_file.stem.replace('predictions_', '').replace('_', ' ')
        
        print_step(f"Checking: {measure}", "")
        
        # Load predictions
        df = pd.read_csv(pred_file)
        
        actual = df['Actual'].values
        predicted = df['Predicted'].values
        residual = df['Residual'].values
        
        # Basic statistics
        print(f"\n  Actual values:")
        print(f"    N = {len(actual)}")
        print(f"    Mean = {actual.mean():.2f}")
        print(f"    Std = {actual.std():.2f}")
        print(f"    Range = [{actual.min():.2f}, {actual.max():.2f}]")
        print(f"    Unique values = {len(np.unique(actual))}")
        
        print(f"\n  Predicted values:")
        print(f"    Mean = {predicted.mean():.2f}")
        print(f"    Std = {predicted.std():.2f}")
        print(f"    Range = [{predicted.min():.2f}, {predicted.max():.2f}]")
        print(f"    Unique values = {len(np.unique(predicted))}")
        
        print(f"\n  Residuals:")
        print(f"    Mean = {residual.mean():.2f} (should be ~0)")
        print(f"    Std = {residual.std():.2f}")
        print(f"    Range = [{residual.min():.2f}, {residual.max():.2f}]")
        
        # Check for issues
        issues = []
        
        # 1. Constant predictions
        if len(np.unique(predicted)) == 1:
            issues.append("❌ CONSTANT PREDICTIONS - model predicting same value for all samples!")
        elif predicted.std() < 0.01:
            issues.append("⚠️  Very low prediction variance (nearly constant)")
        
        # 2. Mean shift
        mean_diff = abs(predicted.mean() - actual.mean())
        if mean_diff > 2 * actual.std():
            issues.append(f"⚠️  Large mean shift: predicted mean is {mean_diff:.2f} away from actual")
        
        # 3. Range mismatch
        pred_range = predicted.max() - predicted.min()
        actual_range = actual.max() - actual.min()
        if pred_range < 0.1 * actual_range:
            issues.append(f"⚠️  Prediction range ({pred_range:.2f}) much smaller than actual ({actual_range:.2f})")
        
        # 4. Extreme residuals
        extreme_residuals = np.abs(residual) > 3 * residual.std()
        n_extreme = extreme_residuals.sum()
        if n_extreme > len(residual) * 0.1:
            issues.append(f"⚠️  Many extreme residuals: {n_extreme}/{len(residual)} ({100*n_extreme/len(residual):.1f}%)")
        
        # 5. Systematic bias
        if abs(residual.mean()) > 0.5 * actual.std():
            issues.append(f"⚠️  Systematic bias: mean residual = {residual.mean():.2f}")
        
        # Print issues
        if issues:
            print(f"\n  🚨 ISSUES DETECTED:")
            for issue in issues:
                print(f"    {issue}")
            issues_found.append((measure, issues))
        else:
            print(f"\n  ✅ No issues detected")
        
        # Show sample predictions
        print(f"\n  Sample predictions (first 5):")
        print(f"    {'Actual':<10} {'Predicted':<10} {'Residual':<10}")
        for i in range(min(5, len(df))):
            print(f"    {df['Actual'].iloc[i]:<10.2f} {df['Predicted'].iloc[i]:<10.2f} {df['Residual'].iloc[i]:<10.2f}")
        
        print("\n" + "-"*80 + "\n")
    
    # Summary
    print_section_header("SUMMARY")
    print_info(f"Total measures checked: {len(pred_files)}", 0)
    print_info(f"Measures with issues: {len(issues_found)}", 0)
    
    if issues_found:
        print()
        print("  Measures with issues:")
        for measure, issues in issues_found:
            print(f"    ❌ {measure}:")
            for issue in issues:
                print(f"       {issue}")
    else:
        print()
        print("  ✅ All measures look good!")


def main():
    parser = argparse.ArgumentParser(
        description="Check optimization predictions for integrity"
    )
    parser.add_argument(
        '--cohort', '-c',
        required=True,
        help="Cohort name (e.g., stanford_asd)"
    )
    parser.add_argument(
        '--measure', '-m',
        help="Specific measure to check (optional, checks all if not specified)"
    )
    
    args = parser.parse_args()
    
    cohort_dir = BASE_DIR / f"{args.cohort}_optimized"
    
    print_section_header(f"PREDICTION INTEGRITY CHECK - {args.cohort.upper()}")
    print()
    print_info(f"Directory: {cohort_dir}", 0)
    print()
    
    if not cohort_dir.exists():
        print_error(f"Directory not found: {cohort_dir}")
        print_info("Have you run optimization for this cohort?", 0)
        return
    
    try:
        check_predictions(cohort_dir, args.measure)
    except Exception as e:
        print_error(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

