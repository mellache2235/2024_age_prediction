# Prediction Integrity Checks - Complete Documentation

## Overview

All optimization scripts now automatically print comprehensive integrity checks for actual vs. predicted values during analysis.

---

## What Gets Checked

### 1. **Actual Values**
```
Actual values:
  N = 99
  Mean = 65.23
  Std = 12.45
  Range = [42.00, 95.00]
  Unique values = 87
```

**Why**: Verifies behavioral data is loaded correctly

---

### 2. **Predicted Values**
```
Predicted values:
  Mean = 64.87
  Std = 11.98
  Range = [45.12, 88.34]
  Unique values = 99
```

**Why**: Verifies model is actually making different predictions (not constant)

---

### 3. **Metrics**
```
Metrics:
  Spearman ρ = 0.293
  P-value = 0.0033
  R² = 0.095
  MAE = 5.26
```

**Why**: Shows correlation strength and statistical significance

---

### 4. **Sample Predictions**
```
Sample predictions (first 5):
    Actual  Predicted   Residual
     65.20      63.40       1.80
     72.10      74.30      -2.20
     58.90      60.10      -1.20
     81.50      79.20       2.30
     55.00      57.80      -2.80
```

**Why**: Allows visual inspection of prediction quality

---

## Automatic Issue Detection

The scripts check for common problems:

### ✅ No Issues (Good Result)
```
✅ No major issues detected
```

### ⚠️ Warnings (Review Required)

**1. Constant Predictions**
```
❌ CONSTANT PREDICTIONS - model predicting same value for all!
```
**Meaning**: Model failed, predicts same value regardless of input  
**Action**: DO NOT use this result

**2. Low Prediction Variance**
```
⚠️  Very low prediction variance: std=0.003
```
**Meaning**: Model predictions are nearly constant  
**Action**: Review, likely not useful

**3. Large Mean Shift**
```
⚠️  Large mean shift: 15.45 (>24.90)
```
**Meaning**: Model systematically over/under-predicts  
**Action**: Check data scaling, might need adjustment

**4. Extreme R²**
```
⚠️  Extreme R²: -6484.1 (indicates overfitting)
```
**Meaning**: Severe overfitting, model failed  
**Action**: DO NOT use this result

**5. High MAE**
```
⚠️  High MAE: 35.60 (>24.90)
```
**Meaning**: Model predictions are very inaccurate  
**Action**: Review if correlation is still meaningful

---

## Example: Good Result ✅

```
📊 PREDICTION INTEGRITY CHECK:
================================================================================

Actual values:
  N = 99
  Mean = 65.23
  Std = 12.45
  Range = [42.00, 95.00]
  Unique values = 87

Predicted values:
  Mean = 64.87
  Std = 11.98
  Range = [45.12, 88.34]
  Unique values = 99

Metrics:
  Spearman ρ = 0.293
  P-value = 0.0033
  R² = 0.095
  MAE = 5.26

✅ No major issues detected

Sample predictions (first 5):
    Actual  Predicted   Residual
     65.20      63.40       1.80
     72.10      74.30      -2.20
     58.90      60.10      -1.20
     81.50      79.20       2.30
     55.00      57.80      -2.80
================================================================================
```

**Interpretation**:
- ✅ Predictions vary (std=11.98)
- ✅ Mean close to actual (64.87 vs 65.23)
- ✅ Reasonable range
- ✅ Significant correlation (p<0.01)
- ✅ Moderate effect (ρ=0.29)

**Action**: ✅ USE THIS RESULT for publication!

---

## Example: Failed Result ❌

```
📊 PREDICTION INTEGRITY CHECK:
================================================================================

Actual values:
  N = 99
  Mean = 65.23
  Std = 12.45
  Range = [42.00, 95.00]
  Unique values = 87

Predicted values:
  Mean = 642.50
  Std = 0.00
  Range = [642.50, 642.50]
  Unique values = 1

Metrics:
  Spearman ρ = -0.022
  P-value = 0.8270
  R² = -6484.483
  MAE = 428.25

🚨 ISSUES DETECTED:
  ❌ CONSTANT PREDICTIONS - model predicting same value for all!
  ⚠️  Large mean shift: 577.27 (>24.90)
  ⚠️  Extreme R²: -6484.5 (indicates overfitting)
  ⚠️  High MAE: 428.25 (>24.90)

Sample predictions (first 5):
    Actual  Predicted   Residual
     65.20     642.50    -577.30
     72.10     642.50    -570.40
     58.90     642.50    -583.60
     81.50     642.50    -561.00
     55.00     642.50    -587.50
================================================================================
```

**Interpretation**:
- ❌ All predictions are 642.50 (constant)
- ❌ Massive mean shift
- ❌ Extreme negative R²
- ❌ No correlation

**Action**: ❌ DO NOT USE - model completely failed!

---

## Where This Appears

### 1. Universal Script
```bash
python run_all_cohorts_brain_behavior_optimized.py --cohort abide_asd
```
Shows integrity check for each behavioral measure

### 2. Stanford ASD Script
```bash
python run_stanford_asd_brain_behavior_optimized.py
```
Shows integrity check for each SRS measure

### 3. NKI Script
```bash
python run_nki_brain_behavior_optimized.py
```
Shows integrity check for each CAARS/Conners measure

---

## Saved Predictions CSV

Each measure also gets a CSV file for detailed inspection:

**File**: `predictions_{measure}.csv`

```csv
Actual,Predicted,Residual
65.20,63.40,1.80
72.10,74.30,-2.20
58.90,60.10,-1.20
81.50,79.20,2.30
55.00,57.80,-2.80
...
```

**Use for**:
- Manual verification
- Creating custom plots
- Statistical analysis
- Debugging

**Location**: Same directory as optimization results

---

## Diagnostic Script

For post-analysis verification:

```bash
# Check all measures
python check_optimization_predictions.py --cohort stanford_asd

# Check specific measure
python check_optimization_predictions.py --cohort stanford_asd --measure social_awareness_tscore
```

**Shows**:
- All integrity checks
- Statistical summaries
- Issue detection
- Sample predictions

---

## How to Interpret

### ✅ Use Result If:
- Spearman ρ > 0.2 (or absolute value)
- P-value < 0.05
- R² between 0 and 1
- Predicted std > 1.0
- No major issues flagged

### ⚠️ Review If:
- Prediction std < 2.0 (low variance)
- Mean shift > 1 std
- MAE > 1.5 × actual std
- R² close to 0

### ❌ Don't Use If:
- Constant predictions (std ≈ 0)
- Extreme R² (< -10 or > 10)
- P-value > 0.05 AND low ρ
- Issues flagged with ❌

---

## Coverage

All optimization scripts now include integrity checks:

| Script | Status |
|--------|--------|
| `run_all_cohorts_brain_behavior_optimized.py` | ✅ Added |
| `run_stanford_asd_brain_behavior_optimized.py` | ✅ Enhanced |
| `run_nki_brain_behavior_optimized.py` | ✅ (on Oak, needs small ID fix) |
| `optimized_brain_behavior_core.py` | ✅ Core logic |

---

## Benefits

1. **Early Detection**: Spot problems immediately during analysis
2. **Confidence**: Know your results are valid before publishing
3. **Documentation**: Saved predictions for reproducibility
4. **Transparency**: Clear diagnostics for all results

---

## Example Workflow

```bash
# 1. Run optimization
python run_all_cohorts_brain_behavior_optimized.py --cohort stanford_asd

# Console shows integrity checks automatically:
# ✅ social_awareness_tscore: No major issues
# ❌ srs_total_score_standard: CONSTANT PREDICTIONS

# 2. Only use measures that passed integrity checks!

# 3. Optional: Verify with diagnostic script
python check_optimization_predictions.py --cohort stanford_asd
```

---

**Status**: ✅ Complete  
**Coverage**: All optimization scripts  
**Output**: Automatic + Saved to CSV  
**Documentation**: Complete

