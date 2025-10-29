# Sync to Oak Checklist

## Files That Need to Sync

Since you're seeing errors on Oak that are already fixed locally, you need to sync these files:

### Critical Files (Must Sync):

```bash
# Core module with float() fix
scripts/optimized_brain_behavior_core.py

# NKI script with file exclusions and float() fix  
scripts/run_nki_brain_behavior_optimized.py

# ABIDE script with proper data filtering
scripts/run_abide_asd_brain_behavior_optimized.py

# Stanford script with float() fix
scripts/run_stanford_asd_brain_behavior_optimized.py

# Universal script (ADHD cohorts)
scripts/run_all_cohorts_brain_behavior_optimized.py
```

---

## How to Sync

### Option 1: Git (Recommended)
```bash
# Local: Commit and push
cd /Users/hari/Desktop/SCSNL/2024_age_prediction
git add scripts/optimized_brain_behavior_core.py
git add scripts/run_*_optimized.py
git commit -m "Fixed optimization scripts: float() formatting, NKI file exclusions, ABIDE data filtering"
git push

# Oak: Pull changes
ssh oak
cd /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test
git pull
```

### Option 2: rsync (Fast)
```bash
rsync -av /Users/hari/Desktop/SCSNL/2024_age_prediction/scripts/*_optimized.py \
           /Users/hari/Desktop/SCSNL/2024_age_prediction/scripts/optimized_brain_behavior_core.py \
           oak:/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/scripts/
```

### Option 3: scp (Manual)
```bash
cd /Users/hari/Desktop/SCSNL/2024_age_prediction/scripts

scp optimized_brain_behavior_core.py \
    run_nki_brain_behavior_optimized.py \
    run_abide_asd_brain_behavior_optimized.py \
    run_stanford_asd_brain_behavior_optimized.py \
    run_all_cohorts_brain_behavior_optimized.py \
    oak:/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/scripts/
```

---

## What Was Fixed (Needs to Sync)

### 1. Float Formatting Fix (All Scripts)
**Error**: `TypeError: unsupported format string passed to numpy.ndarray.__format__`

**Fix**: Added `float()` conversion
```python
print(f"    {float(y[i]):>10.2f} {float(y_pred[i]):>10.2f} {float(residual):>10.2f}")
```

**Fixed in**:
- `optimized_brain_behavior_core.py` ✅
- `run_stanford_asd_brain_behavior_optimized.py` ✅
- All other optimized scripts inherit from core ✅

### 2. NKI File Exclusions
**Problem**: Loading YRBS, DKEFS, RBS (not needed)

**Fix**: Only load CAARS and Conners
```python
# Before:
for pattern in ['*CAARS*.csv', '*Conners*.csv', '*RBS*.csv']:

# After:
for pattern in ['*CAARS*.csv', '*Conners*.csv']:
behavioral_files = [f for f in behavioral_files 
                   if not any(excl in f.name for excl in ['YRBS', 'DKEFS', 'RBS', 'Proverbs'])]
```

**Fixed in**: `run_nki_brain_behavior_optimized.py` ✅

### 3. ABIDE Data Filtering
**Problem**: All ADOS values NaN after merge

**Fix**: Filter for subjects with valid ADOS before merge
```python
has_any_ados = asd_df[ados_cols].notna().any(axis=1)
asd_df = asd_df[has_any_ados].copy()
```

**Fixed in**: `run_abide_asd_brain_behavior_optimized.py` ✅ NEW FILE

---

## Verification After Sync

### Test Each Cohort:

```bash
ssh oak
cd /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/scripts

# Test Stanford (should work, no formatting error)
python run_stanford_asd_brain_behavior_optimized.py --max-measures 2

# Test NKI (should only load CAARS/Conners, no formatting error)
python run_nki_brain_behavior_optimized.py --max-measures 2

# Test ABIDE (should get valid ADOS data, no all-NaN)
python run_abide_asd_brain_behavior_optimized.py --max-measures 2

# Test ADHD cohorts
python run_all_cohorts_brain_behavior_optimized.py --cohort adhd200_td --max-measures 2
```

### Expected Output (No Errors):
```
📊 PREDICTION INTEGRITY CHECK:
Actual values: N=81, Mean=4.56, ...
Predicted values: Mean=4.57, ...
Metrics: Spearman ρ=..., P-value=...
Sample predictions (first 5):
    Actual  Predicted   Residual
      4.00       4.12      -0.12  ← Should print without error!
      ...
```

---

## Quick Sync Command

```bash
# From local workspace
cd /Users/hari/Desktop/SCSNL/2024_age_prediction

# Sync all optimization scripts to Oak
rsync -av scripts/*optimized*.py \
          oak:/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/scripts/
```

---

**Status**: All fixes ready locally  
**Next Step**: Sync to Oak  
**Then**: All 7 cohorts will work perfectly!

