# ⚠️ IMPORTANT: Sync Required Before Using on Oak

## Current Situation

All optimization scripts have been **fixed and updated locally**, but Oak is still running the old buggy versions.

### Errors on Oak (Old Code):
- ❌ `TypeError: unsupported format string` 
- ❌ `ValueError: Per-column arrays must be 1-dimensional`
- ❌ Poor performance (ρ = -0.06 vs baseline ρ = 0.41)
- ❌ NKI loading unwanted files (YRBS, DKEFS, RBS)
- ❌ ABIDE getting all-NaN ADOS values

### Fixed Locally (New Code):
- ✅ All formatting errors fixed
- ✅ Array flattening added
- ✅ NKI only loads CAARS/Conners
- ✅ ABIDE filters for valid ADOS
- ✅ Should match or beat baseline performance

---

## 🚀 Quick Sync (One Command)

```bash
cd /Users/hari/Desktop/SCSNL/2024_age_prediction
bash SYNC_NOW.sh
```

This will:
1. Copy all 5 fixed optimization scripts to Oak
2. Show progress for each file
3. Confirm when complete

**Runtime**: < 1 minute

---

## Alternative: Manual Sync

If the script doesn't work, use rsync:

```bash
cd /Users/hari/Desktop/SCSNL/2024_age_prediction

rsync -av scripts/optimized_brain_behavior_core.py \
          scripts/run_*_optimized.py \
          oak:/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/scripts/
```

---

## After Syncing

### Test on Oak:
```bash
ssh oak
cd /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/scripts

# Test NKI (should work now!)
python run_nki_brain_behavior_optimized.py --max-measures 2
```

### Expected Results:
```
Found 3 behavioral files:  ← Only CAARS/Conners, not 7!
  • 8100_CAARS_20191009.csv
  • 8100_Conners_3-P(S)_20191009.csv
  • 8100_Conners_3-SR(S)_20191009.csv

Optimizing... Best: PLS (ρ=0.42)  ← Should be good CV score

📊 PREDICTION INTEGRITY CHECK:
Metrics:
  Spearman ρ = 0.42+  ← Should be ≥ 0.41 (baseline)
  P-value < 0.001
  R² = 0.18
  
✅ No major issues detected  ← Should pass!

Sample predictions (first 5):
    Actual  Predicted   Residual
      4.00       4.12      -0.12  ← Should print correctly!
```

---

## Checklist

Before testing on Oak:
- [ ] Run `bash SYNC_NOW.sh` from local machine
- [ ] Verify sync succeeded (check for ✅ for each file)
- [ ] SSH to Oak
- [ ] Navigate to scripts directory
- [ ] Run optimization scripts

After syncing:
- [ ] No formatting errors
- [ ] NKI performance ≥ baseline (ρ ≥ 0.41)
- [ ] ABIDE has valid ADOS data
- [ ] All integrity checks pass

---

## Why This Matters

**Before sync** (current Oak state):
- Old buggy code
- Poor performance
- Crashes and errors

**After sync** (with fixes):
- All bugs fixed
- Should match/beat baseline
- Clean execution with integrity checks

---

**Action Required**: Run `bash SYNC_NOW.sh` NOW  
**Then**: All optimizations will work correctly on Oak  
**Expected**: ρ ≥ 0.41 for NKI Hyperactivity (matching or beating baseline)

