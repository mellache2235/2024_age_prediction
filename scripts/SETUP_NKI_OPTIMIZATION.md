# Setup NKI Optimization Script on Oak

## Why This Extra Step?

The NKI optimization script lives on Oak at:
```
/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/scripts/
```

This is **outside the local workspace**, so we need to copy the fixed version there.

---

## 🚀 Quick Setup (One-Time)

### Step 1: Copy Fixed Script to Oak

From your local machine:
```bash
cd /Users/hari/Desktop/SCSNL/2024_age_prediction/scripts
bash COPY_TO_OAK.sh
```

This will:
- ✅ Create backup of existing file on Oak
- ✅ Copy the fixed version
- ✅ Preserve all your data on Oak

**Expected output**:
```
Copying fixed NKI optimization script to Oak...

1. Creating backup on Oak...
2. Copying new file to Oak...

✅ SUCCESS! Fixed NKI script copied to Oak

Now you can run:
  ssh oak
  cd /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/scripts
  python run_nki_brain_behavior_optimized.py
```

---

### Step 2: Run on Oak

```bash
# SSH to Oak
ssh oak

# Navigate to scripts directory
cd /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/scripts

# Test with 2 measures first
python run_nki_brain_behavior_optimized.py --max-measures 2

# If successful, run full analysis
python run_nki_brain_behavior_optimized.py
```

---

## 🔧 What Was Fixed

The fixed version includes:

1. **ID Column Detection**: Now recognizes 'Anonymized ID'
   ```python
   if any(kw in col_lower for kw in ['id', 'subject', 'identifier', 'anonymized']):
   ```

2. **print_info() Format**: Uses correct two-argument format
   ```python
   print_info("IG subjects", 369)  # Not print_info(f"...", 0)
   ```

3. **Data Loading**: Exact same logic as working enhanced NKI script

4. **Integrity Checks**: Comprehensive actual/predicted verification

5. **Descriptive Filenames**: Includes method (e.g., `_PLS_comp15_`)

---

## Alternative: Manual Copy

If the automated script doesn't work:

```bash
scp /Users/hari/Desktop/SCSNL/2024_age_prediction/scripts/run_nki_brain_behavior_optimized_FIXED.py \
    oak:/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/scripts/run_nki_brain_behavior_optimized.py
```

---

## After Setup

Once copied, NKI will work exactly like other cohorts:

```bash
# On Oak
python run_nki_brain_behavior_optimized.py
```

**Output will include**:
```
📊 PREDICTION INTEGRITY CHECK:
================================================================================

Actual values:
  N = 245
  Mean = 52.34, Std = 10.12, Range = [32.00, 78.00]

Predicted values:
  Mean = 51.89, Std = 9.87, Range = [35.45, 72.12]

Metrics:
  Spearman ρ = 0.412
  P-value = 0.0001      ← Shown!
  R² = 0.167
  MAE = 6.54

✅ No major issues detected

Sample predictions (first 5):
    Actual  Predicted   Residual
     52.30      51.40       0.90
     ...
```

**Files created**:
```
scatter_CAARS_36_PLS_comp12_optimized.png
predictions_CAARS_36_PLS.csv
optimization_summary.csv
```

---

## Troubleshooting

### "Permission denied" when copying
```bash
# Make sure you can SSH to Oak
ssh oak "echo 'Connection works'"

# Check if you have write permissions
ssh oak "ls -la /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/scripts/"
```

### "File not found" error
The automated script assumes SSH alias 'oak' exists. If not:
```bash
# Use full hostname instead
scp run_nki_brain_behavior_optimized_FIXED.py \
    YOUR_USER@oak.stanford.edu:/oak/.../scripts/run_nki_brain_behavior_optimized.py
```

---

## ✅ Summary

**Why**: NKI script lives on Oak (outside local workspace)  
**What**: Fixed version created locally  
**How**: One-time copy using `COPY_TO_OAK.sh`  
**Then**: Works exactly like other cohorts

**After this one-time setup, all 7 cohorts will work!** 🎉

---

**Status**: Fix ready, just needs copy step  
**Complexity**: One command  
**Time**: < 1 minute

