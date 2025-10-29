# Testing NKI Optimization Issue

## Problem

**Enhanced script**: ρ = 0.35-0.41 (good results!)  
**Optimized script**: ρ = -0.06 to 0.09 (terrible!)

**This should NEVER happen** - optimization should at minimum match baseline!

---

## Diagnostic Steps

### Step 1: Run Debug Script

```bash
ssh oak
cd /oak/.../scripts
python debug_nki_optimization.py
```

This will:
1. Load data using both enhanced and optimized logic
2. Compare merged subjects (should be same)
3. Test baseline method (PCA 80% + LinearRegression)
4. Show if data is identical or different

**Expected output**:
```
Enhanced merged subjects: 369
Optimized merged subjects: 369
Common IDs: 369 (should be all!)

BASELINE TEST (PCA 80% + LinearRegression):
  Spearman ρ: 0.41  ← Should match enhanced!
```

---

### Step 2: Check If It's a Data Issue

If debug shows ρ ≈ 0.41 for baseline:
- ✅ Data loading is correct
- ❌ Optimization logic has a bug

If debug shows ρ < 0.2 for baseline:
- ❌ Data loading is different
- Need to investigate what's different

---

## Possible Causes

### Cause 1: Data Not Properly Aligned (Most Likely)

Even though code extracts from merged_df, pandas might reorder or something else.

**Solution**: Explicitly verify indices match

### Cause 2: Optimization Testing Wrong Configurations

Maybe the optimization is only testing terrible configurations?

**Check**: Look at `optimization_results_{measure}.csv` - should show ~200 configs with varying scores

### Cause 3: Bug in Cross-Validation Setup

Maybe CV is leaking data or doing something wrong?

**Check**: The `spearman_scorer` function or CV fold creation

### Cause 4: Old Code on Oak

User says files should sync via cloning, but errors suggest old code.

**Check**: Verify `optimized_brain_behavior_core.py` on Oak has latest changes

---

## Quick Fix to Test

### Add This Debug Output to NKI Script

At line ~300 (right before optimization), add:

```python
# DEBUG: Test baseline before optimization
print("\n🔍 BASELINE TEST:")
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

scaler_test = StandardScaler()
X_scaled_test = scaler_test.fit_transform(X_valid)
pca_test = PCA(n_components=min(19, len(y_valid)-1))
pca_scores_test = pca_test.fit_transform(X_scaled_test)
model_test = LinearRegression()
model_test.fit(pca_scores_test, y_valid)
y_pred_test = model_test.predict(pca_scores_test)
rho_test, p_test = spearmanr(y_valid, y_pred_test)
print(f"PCA(19) + LinearRegression: ρ = {rho_test:.3f}, p = {p_test:.4f}")
print(f"This should be ≈ 0.35-0.41 (enhanced script baseline)")
if rho_test < 0.3:
    print("⚠️  BASELINE IS POOR - DATA LOADING ISSUE!")
else:
    print("✅ Baseline is good - optimization should find this or better")
print()
```

This will immediately show if the data is correct by testing the same method as enhanced.

---

## Expected Behavior

### If Data is Correct:
```
BASELINE TEST:
PCA(19) + LinearRegression: ρ = 0.410, p < 0.001
✅ Baseline is good - optimization should find this or better

Optimizing...
Best: PCA+Linear with 20 components (ρ=0.405) ← Should be close!
or
Best: PLS with 15 components (ρ=0.450) ← Even better!
```

### If Data is Wrong:
```
BASELINE TEST:
PCA(19) + LinearRegression: ρ = 0.050, p = 0.65
⚠️  BASELINE IS POOR - DATA LOADING ISSUE!
```

Then we know the problem is in data loading, not optimization!

---

## Next Steps

1. Run `debug_nki_optimization.py` to test baseline
2. If baseline is good (ρ ≈ 0.41):
   - Bug is in optimization logic
   - Need to debug `optimize_comprehensive()`
   
3. If baseline is poor (ρ < 0.3):
   - Bug is in data loading
   - Compare enhanced vs optimized data loading step by step

---

**Status**: Needs diagnosis  
**Tools**: debug_nki_optimization.py created  
**Expected**: Baseline should be ρ ≈ 0.41

