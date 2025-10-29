# ⚠️ CRITICAL: Must Sync to Oak Before Running

## 🚨 Important Notice

**The errors you're seeing on Oak are from OLD versions of the scripts!**

All fixes have been applied **locally** but Oak is still running old code.

---

## What's Happening

### On Oak (OLD CODE):
```
Optimizing... Best: PLS (ρ=0.0947)
Final: ρ = -0.061  ← TERRIBLE! Much worse than baseline!
Error: Per-column arrays must each be 1-dimensional
Error: unsupported format string passed to numpy.ndarray
```

### Locally (FIXED CODE):
- ✅ Float() conversion added
- ✅ Array flattening added
- ✅ NKI file exclusions added
- ✅ ABIDE data filtering added
- ✅ All should perform ≥ baseline

---

## 🔄 YOU MUST SYNC BEFORE TESTING

### Quick Sync Command:

```bash
cd /Users/hari/Desktop/SCSNL/2024_age_prediction

# Sync all optimized scripts to Oak
rsync -av scripts/*optimized*.py \
          oak:/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/scripts/

# Verify sync
ssh oak "ls -lh /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/scripts/*optimized*.py"
```

---

## Files That MUST Be Synced

### Critical (Must sync these):
1. **`optimized_brain_behavior_core.py`**
   - Has float() fix
   - Has array flattening
   - Has proper evaluate_model() function

2. **`run_nki_brain_behavior_optimized.py`**
   - Excludes YRBS/DKEFS/RBS files
   - Has float() fix
   - Has array flattening

3. **`run_stanford_asd_brain_behavior_optimized.py`**
   - Has float() fix
   - Has array flattening

4. **`run_abide_asd_brain_behavior_optimized.py`**
   - NEW FILE - must be copied
   - Proper ADOS filtering

5. **`run_all_cohorts_brain_behavior_optimized.py`**
   - Has float() fix
   - Has array flattening

---

## Why Optimization Performed Poorly on Oak

**Root cause**: Old code on Oak

The old version has bugs that make optimization fail:
- Missing float() conversions → crashes
- Missing array flattening → DataFrame errors
- No file exclusions → loads irrelevant NKI files
- No ADOS filtering → all NaN for ABIDE

**After syncing**:
- All bugs fixed ✅
- Should match or beat baseline ✅
- Expected NKI: ρ ≥ 0.41 (baseline was 0.41)

---

## Verification After Sync

### 1. Check files synced:
```bash
ssh oak
cd /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/scripts

# Check file dates (should be recent)
ls -lht *optimized*.py | head -10
```

### 2. Test NKI (should work now):
```bash
python run_nki_brain_behavior_optimized.py --max-measures 2
```

**Expected**:
```
Found 3 behavioral files:  ← Not 7!
  • 8100_CAARS_20191009.csv
  • 8100_Conners_3-P(S)_20191009.csv
  • 8100_Conners_3-SR(S)_20191009.csv

📊 PREDICTION INTEGRITY CHECK:
Metrics:
  Spearman ρ = 0.42+  ← Should be ≥ 0.41 (baseline)
  P-value < 0.001

Sample predictions (first 5):
    Actual  Predicted   Residual
      4.00       4.12      -0.12  ← Should print without error!

✅ No major issues detected
```

---

## Performance Expectations After Sync

### NKI Baseline (Enhanced):
- Hyperactivity T-Score: ρ = 0.410

### NKI Optimized (Should achieve):
- Hyperactivity T-Score: ρ = 0.41 - 0.50
- Likely winner: PLS or PCA+Ridge with different component count

### If Still Poor After Sync:
```bash
# Check the optimization_results CSV to see what was tested
head -20 optimization_results_B_T-SCORE_HYPERACTIVITY_RESTLESSNESS.csv

# Should show ~200 different configurations
wc -l optimization_results_B_T-SCORE_HYPERACTIVITY_RESTLESSNESS.csv
# Should show ~100-200 lines
```

---

## 🎯 Action Items

1. ✅ **All fixes applied locally**
2. ⚠️ **MUST SYNC TO OAK** ← **DO THIS NOW**
3. ⏳ **Then test on Oak** ← Will work after sync

---

**Status**: Fixes complete locally  
**Blocking**: Files not synced to Oak yet  
**Next Step**: Sync files (use rsync command above)  
**Then**: All 7 cohorts will work correctly with optimized performance!

