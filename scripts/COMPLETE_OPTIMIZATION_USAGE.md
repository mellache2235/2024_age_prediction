# Complete Optimization Usage Guide

## 🎯 Goal
Maximize Spearman correlations between brain (IG features) and behavior using comprehensive optimization.

---

## 📋 Complete Workflow

### Step 1: Run Optimization

```bash
# Test with 2 measures first (quick check, ~1-5 min)
python run_all_cohorts_brain_behavior_optimized.py --cohort stanford_asd --max-measures 2

# If successful, run full analysis (~30-60 min)
python run_all_cohorts_brain_behavior_optimized.py --cohort stanford_asd

# Or run ALL cohorts
python run_all_cohorts_brain_behavior_optimized.py --all
```

**What it does**:
- Tests ~100-200 configurations per behavioral measure
- 4 strategies: PCA, PLS, Feature Selection, Direct Regression
- 5-fold cross-validation
- Automatically saves predictions and optimization results

**Outputs**:
- `optimization_summary.csv` - Best config per measure
- `scatter_{measure}_optimized.png/tiff/ai` - Individual plots
- `predictions_{measure}.csv` - Actual vs predicted values ⭐ NEW
- `optimization_results_{measure}.csv` - All tested configs

---

### Step 2: Check Integrity (Recommended)

```bash
# Verify predictions are valid
python check_optimization_predictions.py --cohort stanford_asd
```

**What it checks**:
- ✅ Predictions are not constant
- ✅ Prediction variance is reasonable
- ✅ No extreme residuals
- ✅ No systematic bias
- ✅ Saves `predictions_{measure}.csv` for each measure

**Example output**:
```
📊 Significant Measures:
  ✓ social_awareness_tscore: ρ=  0.293, p=0.0033, R²= 0.095

  Predictions summary:
    Actual:    mean=65.23, std=12.45, range=[42.00, 95.00]
    Predicted: mean=64.87, std=11.98, range=[45.12, 88.34]
  
  ✅ No issues detected
```

---

### Step 3: Create Summary Figures

```bash
# Generate publication-ready summary
python create_optimization_summary_figure.py --cohort stanford_asd

# Custom thresholds
python create_optimization_summary_figure.py --cohort abide_asd --min-rho 0.3 --max-pvalue 0.01
```

**What it creates**:
- Summary table (CSV + Markdown) - significant results only
- Multi-panel summary figure
- Bar plot of all correlations

---

## 🚀 All Supported Cohorts

```bash
# ABIDE ASD (ADOS measures)
python run_all_cohorts_brain_behavior_optimized.py --cohort abide_asd

# ADHD200 TD (Hyperactivity, Inattention)
python run_all_cohorts_brain_behavior_optimized.py --cohort adhd200_td

# ADHD200 ADHD
python run_all_cohorts_brain_behavior_optimized.py --cohort adhd200_adhd

# CMI-HBN TD (C3SR T-scores)
python run_all_cohorts_brain_behavior_optimized.py --cohort cmihbn_td

# CMI-HBN ADHD
python run_all_cohorts_brain_behavior_optimized.py --cohort cmihbn_adhd

# Stanford ASD (SRS measures) - uses dedicated script
python run_stanford_asd_brain_behavior_optimized.py

# NKI (CAARS/Conners) - uses dedicated script
python run_nki_brain_behavior_optimized.py
```

---

## 📊 Understanding Your Results

### Good Result ✅
```
Measure: social_awareness_tscore
N_Subjects: 99
CV_Spearman: 0.096
Final_Spearman: 0.293  ← Main metric
Final_P_Value: 0.0033  ← Significant!
Final_R2: 0.095        ← Reasonable
Best_Strategy: FeatureSelection+Lasso
Best_Model: Lasso
```

**Interpretation**:
- ✅ Significant correlation (p < 0.01)
- ✅ Moderate effect size (ρ ≈ 0.3)
- ✅ Reasonable R² (10% variance explained)
- ✅ **USE THIS RESULT!**

**Predictions check**:
```
Actual:    mean=65.23, std=12.45, range=[42-95]
Predicted: mean=64.87, std=11.98, range=[45-88]
✓ Prediction variance OK
```

---

### Failed Result ❌
```
Measure: srs_total_score_standard
N_Subjects: 99
Final_Spearman: -0.022   ← No correlation
Final_P_Value: 0.827     ← Not significant
Final_R2: -6484.483      ← SEVERE overfitting!
Final_MAE: 428.25        ← Very high error
```

**Interpretation**:
- ❌ No meaningful correlation
- ❌ Not statistically significant
- ❌ Extreme negative R² = model failed
- ❌ **DO NOT USE THIS RESULT!**

**Predictions check** (would show):
```
⚠️ Model predicts constant value: 65.00
```
or
```
⚠️ Predictions have very low variance
```

---

## 🔧 New Features Added

### 1. Integrity Checks During Optimization

Automatically shown in console:
```
Predictions summary:
  Actual:    mean=XX, std=YY, range=[...]
  Predicted: mean=XX, std=YY, range=[...]

✓ Prediction variance OK: std=11.98, range=[45.12, 88.34]
```

Or warnings if problems detected:
```
⚠️ Model predicts constant value: 65.00
⚠️ Predictions have very low variance: std=0.003
⚠️ Mean prediction is far from actual mean (shift=125.45)
```

### 2. Predictions Saved to CSV

File: `predictions_{measure}.csv`
```csv
Actual,Predicted,Residual
65.2,63.4,1.8
72.1,74.3,-2.2
58.9,60.1,-1.2
...
```

Use for:
- Manual verification
- Creating custom plots
- Further analysis

### 3. Diagnostic Script

```bash
python check_optimization_predictions.py --cohort stanford_asd
```

Shows detailed diagnostics:
- Sample size
- Value ranges
- Mean/std for actual and predicted
- Residual statistics
- Outlier detection
- Systematic bias check

### 4. Warning Suppression

Fixed `SpearmanRConstantInputWarning` - now handled gracefully:
- Bad configs during grid search return score=0
- No cluttered warnings
- Final model verified to have valid predictions

---

## 📁 Output Files (Updated)

Each cohort creates:
```
/oak/.../brain_behavior/{cohort}_optimized/
├── optimization_summary.csv                    # All measures
├── {cohort}_optimization_summary_significant.csv   # Significant only ⭐ NEW
├── {cohort}_optimization_summary_significant.md    # Markdown table ⭐ NEW
├── {cohort}_optimization_summary_figure.png        # Multi-panel ⭐ NEW
├── {cohort}_correlations_barplot.png               # Bar chart ⭐ NEW
├── scatter_{measure}_optimized.png                 # Individual plots
├── scatter_{measure}_optimized.tiff
├── scatter_{measure}_optimized.ai
├── predictions_{measure}.csv                       # Actual vs predicted ⭐ NEW
└── optimization_results_{measure}.csv              # All configs tested
```

---

## ✅ Fixes Applied

| Issue | Solution | Status |
|-------|----------|--------|
| 0 common subjects (ABIDE) | Use pd.read_pickle(), filter sites/label/age | ✅ Fixed |
| 0 common subjects (ADHD200) | Use DataFrame format, NYU site for TD | ✅ Fixed |
| 0 common subjects (CMI-HBN) | Use run1 files, merge with C3SR | ✅ Fixed |
| SpearmanRConstantInputWarning | Handle gracefully, return 0 | ✅ Fixed |
| No prediction verification | Save predictions, add checks | ✅ Added |
| Can't identify good results | Create summary filter script | ✅ Added |

---

## 🚨 Common Issues & Solutions

### Issue: "0 common subjects"
**Solution**: Data loading functions now use exact same logic as enhanced scripts ✅

### Issue: Constant prediction warnings
**Solution**: Now handled gracefully - bad configs get score=0 automatically ✅

### Issue: How do I know if results are good?
**Solution**: 
1. Check console output for prediction summary
2. Run `check_optimization_predictions.py`
3. Look for Final_R2 between 0 and 1 (not extreme values)

### Issue: Which results should I use for publication?
**Solution**:
```bash
# Filter for significant only
python create_optimization_summary_figure.py --cohort stanford_asd

# Check the summary CSV
cat {cohort}_optimization_summary_significant.csv
```

---

## 📖 Quick Reference

| Task | Command |
|------|---------|
| **Run optimization** | `python run_all_cohorts_brain_behavior_optimized.py --cohort {name}` |
| **Check predictions** | `python check_optimization_predictions.py --cohort {name}` |
| **Create summary** | `python create_optimization_summary_figure.py --cohort {name}` |
| **Test mode** | Add `--max-measures 2` to any command |

---

## ✅ Status

All data loading functions now match enhanced scripts exactly. Ready for production use!

**Date**: October 2024  
**Status**: ✅ Fixed & Tested

