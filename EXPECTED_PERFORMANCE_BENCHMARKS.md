# Expected Performance Benchmarks

## Baseline Performance (Enhanced Scripts)

These are the correlations achieved by the **standard enhanced scripts** (fixed PCA, LinearRegression):

### NKI Baseline Results
From `run_nki_brain_behavior_enhanced.py`:
- **Inattention (Raw)**: ρ = 0.353, p = 0.0012
- **Hyperactivity (Raw)**: ρ = 0.385, p < 0.001
- **Inattention (T-Score)**: ρ = 0.334, p = 0.0023
- **Hyperactivity (T-Score)**: ρ = 0.410, p < 0.001  ← **Best baseline**

**Method**: PCA (80% variance = 19 components) + LinearRegression

---

## Expected Optimized Performance

The **optimized scripts should achieve ≥ baseline** results since they test:

### What Optimization Tests:
- PCA: 5, 10, 15, 20, 25, 30, 35, 40, 45, 50 components
- PLS: 3, 6, 9, 12, 15, 18, 21, 24, 27, 30 components
- Models: Linear, Ridge, Lasso, ElasticNet, PLS
- Feature Selection: Top-K features [50, 100, 150, 200]
- **Total: ~100-200 configurations**

### Expected Improvements:

| Measure | Baseline (Enhanced) | Expected (Optimized) | Improvement |
|---------|---------------------|----------------------|-------------|
| Hyperactivity T-Score | ρ = 0.410 | ρ = 0.41 - 0.50 | +0% to +25% |
| Hyperactivity Raw | ρ = 0.385 | ρ = 0.39 - 0.48 | +1% to +25% |
| Inattention Raw | ρ = 0.353 | ρ = 0.35 - 0.45 | +0% to +28% |
| Inattention T-Score | ρ = 0.334 | ρ = 0.33 - 0.43 | +0% to +29% |

**Minimum expectation**: Match or slightly exceed baseline  
**Typical improvement**: +5-15%  
**Best case**: +20-30% (if PLS or feature selection works much better)

---

## Why Optimization Should Help

### Strategies That May Outperform:

1. **PLS Regression**
   - Optimizes covariance between brain and behavior (not just brain variance)
   - Often achieves 10-20% better correlations than PCA
   - **Expected**: ρ = 0.42 - 0.50 for best measures

2. **Feature Selection**
   - Focuses on most relevant brain regions
   - Reduces noise from irrelevant ROIs
   - **Expected**: ρ = 0.40 - 0.48 if sparse relationship

3. **Optimal PCA Components**
   - Baseline uses 19 PCs (80% variance)
   - Optimal might be 10, 15, 25, or 30
   - Finding the sweet spot can improve correlations

4. **Regularization**
   - Ridge/Lasso can handle overfitting better than LinearRegression
   - **Expected**: ρ = 0.39 - 0.47

---

## Interpretation of Results

### If Optimized ≈ Baseline (±5%)
**Meaning**: LinearRegression with 80% variance PCA is already near-optimal for this data  
**Still valuable**: Cross-validation confirms generalizability

### If Optimized > Baseline (+10-30%)
**Meaning**: Found better strategy (likely PLS or feature selection)  
**Great**: Publication-worthy improvement!

### If Optimized < Baseline (-5% or more)
**Problem**: Something wrong with optimization (check for data leakage, overfitting)  
**Action**: Review optimization results CSV, verify best configuration

---

## Your NKI Results Should Show

After running the optimized version, you should see something like:

```
📊 PREDICTION INTEGRITY CHECK:

Metrics:
  Spearman ρ = 0.425    ← Should be ≥ 0.410 (baseline)
  P-value < 0.001
  R² = 0.180
  MAE = 2.15

Best configuration:
  Strategy: PLS  (or FeatureSelection+Lasso, or PCA+Ridge with different PCs)
  Components: 15
```

**Filename**:
```
scatter_B_T-SCORE_HYPERACTIVITY_RESTLESSNESS_PLS_comp15_optimized.png
```

From the filename alone, you'll know:
- Which optimization strategy won
- Exact parameters used
- Can compare to baseline (which used PCA with 19 components)

---

## Benchmark Summary

| Cohort | Best Baseline | Expected Optimized | Strategies to Try |
|--------|---------------|-------------------|-------------------|
| **NKI** | ρ = 0.410 | ρ = 0.41 - 0.50 | PLS, Ridge, Feature Selection |
| Stanford ASD | ρ = 0.293 | ρ = 0.25 - 0.35 | Feature Selection (already found!) |
| ABIDE ASD | TBD | ρ = 0.30 - 0.45 | PLS, Feature Selection |
| ADHD200 | TBD | ρ = 0.25 - 0.40 | PLS, PCA optimization |
| CMI-HBN | TBD | ρ = 0.30 - 0.45 | PLS, Ridge |

---

## What If Optimization Doesn't Beat Baseline?

**This is OK!** It means:
- ✅ The baseline method was already good
- ✅ Cross-validation confirms it's robust
- ✅ You have confidence in the results

**Still valuable because**:
- Comprehensive search confirms no better method exists
- Cross-validated estimates are more trustworthy
- Rules out alternative explanations

---

## Action Items

1. **Sync files to Oak** (see `SYNC_TO_OAK_CHECKLIST.md`)
2. **Run optimized NKI script**
3. **Compare to baseline**: Should get ρ ≥ 0.41 for Hyperactivity T-Score
4. **Check optimization_summary.csv**: See which method won

If optimized < baseline, investigate:
- Check CV scores vs final scores (overfitting?)
- Review which configuration won
- Ensure data is the same (same subjects, same preprocessing)

---

**Baseline**: ρ = 0.41 (very good!)  
**Expected Optimized**: ρ = 0.41 - 0.50 (equal or better)  
**Status**: Ready to test after sync!

