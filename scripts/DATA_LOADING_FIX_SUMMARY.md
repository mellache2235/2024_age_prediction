# Data Loading Fix Summary

## Issue

The universal optimized script (`run_all_cohorts_brain_behavior_optimized.py`) was failing to merge IG and behavioral data with 0 common subjects for:
- ABIDE ASD
- ADHD200 TD
- ADHD200 ADHD
- CMI-HBN TD
- CMI-HBN ADHD

## Root Cause

The original data loading functions used generic logic that didn't match the specific data formats:
1. **ABIDE**: Uses `pd.read_pickle()` directly (not gzip), filters for specific sites and ASD label
2. **ADHD200**: Uses `pd.read_pickle()` (not gzip dict), filters for TD/ADHD label, NYU site only for TD
3. **CMI-HBN**: Uses `pd.read_pickle()` with run1 files, merges with C3SR CSV

## Solution

Replaced all data loading functions with **exact logic from enhanced scripts**:

### 1. `load_abide_behavioral(config)`
- ✅ Uses `pd.read_pickle()` instead of gzip
- ✅ Filters for 246ROIs.pklz files
- ✅ Filters by site list
- ✅ Filters for ASD subjects (label == '1')
- ✅ Filters for age <= 21
- ✅ Finds ADOS columns (ados_total, ados_comm, ados_social)

### 2. `load_pklz_behavioral(config)`
For ADHD200 cohorts:
- ✅ Uses `pd.read_pickle()` (DataFrame format)
- ✅ Filters: TR != 2.5, no 'pending' labels, mean_fd < 0.5
- ✅ Filters for TD (label=0) or ADHD (label=1)
- ✅ For TD: NYU site only (scale consistency)
- ✅ Extracts behavioral columns (Hyper/Impulsive, Inattentive)
- ✅ Handles nested arrays in behavioral data

### 3. `load_c3sr_behavioral(config)`
For CMI-HBN cohorts:
- ✅ Loads run1 .pklz files with `pd.read_pickle()`
- ✅ Filters for TD (label=0) or ADHD (label=1)
- ✅ Filters mean_fd < 0.5
- ✅ Merges with C3SR CSV
- ✅ Truncates C3SR IDs to first 12 characters
- ✅ Finds C3SR T-score columns

## Additional Improvements

### 1. Integrity Checking
Added to `evaluate_best_model()`:
```python
# Diagnostic info
print("  Predictions summary:")
print(f"    Actual:    mean={y.mean():.2f}, std={y.std():.2f}")
print(f"    Predicted: mean={y_pred.mean():.2f}, std={y_pred.std():.2f}")

# Check for problems
if y_pred.std() < 0.01:
    print_warning("⚠️  Predictions are nearly constant!")
```

### 2. Prediction Saving
Added to `analyze_single_measure()`:
```python
# Save actual vs predicted values for integrity check
predictions_df = pd.DataFrame({
    'Actual': eval_results['y_actual'],
    'Predicted': eval_results['y_pred'],
    'Residual': eval_results['y_actual'] - eval_results['y_pred']
})
predictions_df.to_csv(Path(output_dir) / f"predictions_{safe_name}.csv", index=False)
```

### 3. Warning Suppression
Updated `spearman_scorer()` in `optimized_brain_behavior_core.py`:
```python
# Handle constant predictions gracefully
if len(np.unique(y_pred)) == 1:
    return 0.0  # Don't raise warning, just return 0
```

## Testing

Run tests for each cohort:

```bash
# Test ABIDE ASD
python run_all_cohorts_brain_behavior_optimized.py --cohort abide_asd --max-measures 2

# Test ADHD200 TD
python run_all_cohorts_brain_behavior_optimized.py --cohort adhd200_td --max-measures 2

# Test ADHD200 ADHD
python run_all_cohorts_brain_behavior_optimized.py --cohort adhd200_adhd --max-measures 2

# Test CMI-HBN TD
python run_all_cohorts_brain_behavior_optimized.py --cohort cmihbn_td --max-measures 2

# Test CMI-HBN ADHD
python run_all_cohorts_brain_behavior_optimized.py --cohort cmihbn_adhd --max-measures 2
```

Expected output:
```
✓ Merged: [N>0] subjects with both IG and behavioral data
```

## Files Modified

1. `run_all_cohorts_brain_behavior_optimized.py` - Fixed all data loading
2. `optimized_brain_behavior_core.py` - Fixed warning handling
3. `run_stanford_asd_brain_behavior_optimized.py` - Added integrity checks

## New Files Created

1. `check_optimization_predictions.py` - Diagnostic tool to verify predictions
2. `create_optimization_summary_figure.py` - Publication figure generator

## Status

✅ All cohorts now use exact same data loading logic as working enhanced scripts  
✅ Integrity checking added  
✅ Warnings handled gracefully  
✅ Predictions saved for verification

**Ready to test!**

---

**Date**: October 2024  
**Issue**: Data loading mismatch  
**Resolution**: Match enhanced script logic exactly

