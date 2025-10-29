# Quick Start: Optimized Brain-Behavior Analysis - All Cohorts

## 🚀 Running Optimization for Each Cohort

### ✅ Cohorts 1-5: Run Directly (No Extra Steps)

```bash
# ABIDE ASD
python run_all_cohorts_brain_behavior_optimized.py --cohort abide_asd

# ADHD200 TD
python run_all_cohorts_brain_behavior_optimized.py --cohort adhd200_td

# ADHD200 ADHD
python run_all_cohorts_brain_behavior_optimized.py --cohort adhd200_adhd

# CMI-HBN TD
python run_all_cohorts_brain_behavior_optimized.py --cohort cmihbn_td

# CMI-HBN ADHD
python run_all_cohorts_brain_behavior_optimized.py --cohort cmihbn_adhd

# Or run all 5 at once
python run_all_cohorts_brain_behavior_optimized.py --all
```

**Location**: Run from local workspace  
**Status**: ✅ Ready immediately

---

### ✅ Cohort 6: Stanford ASD (Run Directly)

```bash
python run_stanford_asd_brain_behavior_optimized.py
```

**Location**: Run from local workspace  
**Status**: ✅ Ready immediately

---

### 🔧 Cohort 7: NKI (Requires One-Time Setup)

#### One-Time Setup:

```bash
# Step 1: Copy fixed script to Oak (from local machine)
cd /Users/hari/Desktop/SCSNL/2024_age_prediction/scripts
bash COPY_TO_OAK.sh

# Expected output:
# ✅ SUCCESS! Fixed NKI script copied to Oak
```

#### Then Run on Oak:

```bash
# Step 2: SSH to Oak
ssh oak

# Step 3: Run optimization
cd /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/scripts
python run_nki_brain_behavior_optimized.py
```

**Location**: Runs on Oak  
**Status**: ✅ Ready after one-time copy

---

## 📊 What You Get (All Cohorts)

### Console Output:
```
📊 PREDICTION INTEGRITY CHECK:
Actual values: N=99, Mean=65.23, Std=12.45, Range=[42.00, 95.00]
Predicted values: Mean=64.87, Std=11.98, Range=[45.12, 88.34]
Metrics: Spearman ρ=0.293, P-value=0.0033, R²=0.095
✅ No major issues detected
Sample predictions (first 5): [shown]
```

### Files Created:
```
scatter_measure_METHOD_PARAMS_optimized.png    ← Method in filename!
predictions_measure_METHOD.csv                  ← Actual vs predicted
optimization_summary.csv                        ← All measures
```

---

## ⏱️ Runtime Expectations

| Cohort | N Subjects | Measures | Expected Time |
|--------|------------|----------|---------------|
| ABIDE ASD | ~300 | 3 | ~5-10 min |
| ADHD200 TD | ~50-100 | 2 | ~3-5 min |
| ADHD200 ADHD | ~100-150 | 2 | ~3-5 min |
| CMI-HBN TD | ~200 | 10-20 | ~15-30 min |
| CMI-HBN ADHD | ~150 | 10-20 | ~10-20 min |
| Stanford ASD | ~100 | 2 | ~2-5 min |
| NKI | ~300 | 20-40 | ~30-60 min |

**Test mode** (--max-measures 2): ~2-5 min for any cohort

---

## 📋 Quick Reference

### Local Cohorts (Run Immediately)
```bash
cd /Users/hari/Desktop/SCSNL/2024_age_prediction/scripts

# Universal script (5 cohorts)
python run_all_cohorts_brain_behavior_optimized.py --cohort {abide_asd|adhd200_td|adhd200_adhd|cmihbn_td|cmihbn_adhd}

# Stanford ASD
python run_stanford_asd_brain_behavior_optimized.py
```

### Oak Cohort (One-Time Setup, Then Run)
```bash
# ONE-TIME: Copy to Oak
bash COPY_TO_OAK.sh

# THEN: Run on Oak
ssh oak
python /oak/.../scripts/run_nki_brain_behavior_optimized.py
```

---

## 🎯 Summary

| Location | Cohorts | Setup Required | Command |
|----------|---------|----------------|---------|
| **Local** | 6 cohorts | None | Just run the script |
| **Oak** | 1 cohort (NKI) | One-time copy | `bash COPY_TO_OAK.sh` then run |

**Total**: 7/7 cohorts ready! 🎉

---

## ✅ Checklist

- [ ] For local cohorts: Just run the script
- [ ] For NKI: Run `bash COPY_TO_OAK.sh` once
- [ ] All cohorts will show comprehensive integrity checks
- [ ] All results include p-values
- [ ] All plots use "r =" format
- [ ] All filenames include method used

**Everything is ready!** 🚀

---

**Last Updated**: October 2024  
**Status**: Complete  
**All Cohorts**: 7/7 Ready

