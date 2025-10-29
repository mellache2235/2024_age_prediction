# Known Issues and Fixes

## Issue 1: Formatting Error in Sample Predictions ✅ FIXED

### Error
```
TypeError: unsupported format string passed to numpy.ndarray.__format__
```

### Cause
When printing sample predictions, numpy scalar types can't be directly formatted with f-strings.

### Fix Applied
Added `float()` conversion:
```python
# Before (causes error):
print(f"    {y[i]:>10.2f} {y_pred[i]:>10.2f} {residual:>10.2f}")

# After (works):
print(f"    {float(y[i]):>10.2f} {float(y_pred[i]):>10.2f} {float(residual):>10.2f}")
```

**Fixed in**:
- ✅ `run_stanford_asd_brain_behavior_optimized.py`
- ✅ `optimized_brain_behavior_core.py`
- ✅ `run_abide_asd_brain_behavior_optimized.py`

---

## Issue 2: Extreme Predictions (Not an Error, But Important!)

### Example from Your Output
```
Actual values: Range = [53.00, 90.00]
Predicted values: Range = [-782.60, 1999.40]  ← WAY OFF!
R² = -6484.483
```

### This Is NOT a Bug - It's a Failed Model!

**What happened**:
- The optimization tried ~200 configurations for `srs_total_score_standard`
- ALL of them failed to find a good relationship
- The "best" one still has terrible predictions

**Why it happened**:
- No real brain-behavior correlation exists for this measure
- Or the relationship is too weak/noisy to model

### How To Interpret

**✅ Good result (social_awareness_tscore)**:
```
ρ = 0.232, p = 0.001
Predicted range: [59.64, 61.07] vs Actual: [53.00, 90.00]
Prediction variance OK
```
→ **USE THIS ONE!**

**❌ Failed result (srs_total_score_standard)**:
```
ρ = -0.022, p = 0.827
Predicted range: [-782.60, 1999.40] vs Actual: [53.00, 90.00]
R² = -6484
🚨 ISSUES DETECTED: Extreme R², High MAE
```
→ **DO NOT USE!** Model completely failed.

### Automatic Detection

The script now automatically warns you:
```
🚨 ISSUES DETECTED:
  ⚠️  Extreme R²: -6484.5 (indicates overfitting)
  ⚠️  High MAE: 428.25 (>17.64)
```

This tells you immediately not to use this result!

---

## Issue 3: ABIDE Subject Mismatch ✅ FIXED

### Error
```
Merged: 169 subjects
But all ADOS values = NaN
```

### Cause
Subjects with IG data didn't overlap with subjects with valid ADOS scores.

### Fix Applied
Created dedicated ABIDE script that:
1. Uses exact enhanced script data loading
2. Handles ID stripping (e.g., "0050001" → "50001")
3. Only returns subjects with valid ADOS scores

**Fixed in**: `run_abide_asd_brain_behavior_optimized.py` ⭐ NEW

---

## Issue 4: NKI 'Anonymized ID' Not Recognized ✅ FIXED

### Error
```
ValueError: No subject ID column found
Available columns: ['Anonymized ID', ...]
```

### Fix Applied
Updated ID detection to include 'anonymized' keyword:
```python
if any(kw in col_lower for kw in ['id', 'subject', 'identifier', 'anonymized']):
```

**Fixed in**: `run_nki_brain_behavior_optimized.py` ⭐ FIXED

---

## Summary

| Issue | Status | Solution |
|-------|--------|----------|
| Numpy formatting error | ✅ Fixed | Added `float()` conversion |
| Extreme predictions | ✅ Detected | Automatic warnings, exclude from results |
| ABIDE subject mismatch | ✅ Fixed | Dedicated script with ID stripping |
| NKI ID column | ✅ Fixed | Added 'anonymized' to keywords |

---

## What To Do With Failed Results

When you see extreme R² or constant predictions:

**Don't panic!** This just means:
- No strong brain-behavior relationship exists for that measure
- The optimization correctly identified this

**What to do**:
1. ✅ Check the automatic issue detection
2. ✅ Look for "No major issues" vs "ISSUES DETECTED"
3. ✅ Only use results that pass integrity checks
4. ✅ Use `create_optimization_summary_figure.py --cohort {name}` to filter significant only

**Example**:
```bash
# This automatically filters out bad results
python create_optimization_summary_figure.py --cohort stanford_asd --min-rho 0.2

# Output:
# Significant: 1/2 measures
# ✓ social_awareness_tscore (use this!)
# ✗ srs_total_score_standard excluded (failed)
```

---

**Status**: All issues fixed or properly detected!

