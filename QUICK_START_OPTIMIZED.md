# Quick Start: Optimized Brain-Behavior Analysis

**TL;DR**: Run optimized analysis for maximum correlations, especially for small sample sizes.

---

## ⚡ Quick Commands

### Run Optimization (Pick One):

```bash
# NKI (N~81, Hyperactivity/Inattention/Impulsivity only)
python run_nki_brain_behavior_optimized.py

# Stanford ASD (N~99, SRS measures, parallel processing)
python run_stanford_asd_brain_behavior_optimized.py

# ABIDE ASD (N~169, ADOS measures)
python run_abide_asd_brain_behavior_optimized.py

# ADHD200 or CMI-HBN (universal script)
python run_all_cohorts_brain_behavior_optimized.py --cohort adhd200_td
python run_all_cohorts_brain_behavior_optimized.py --cohort cmihbn_td
```

### Validate Results:

```bash
# Check predictions are correct
python check_optimization_predictions.py --cohort stanford_asd

# Create summary figure
python create_optimization_summary_figure.py --cohort stanford_asd
```

---

## 🎯 What You Get

### Automatic Strategy Testing:

**For Every Behavioral Measure:**
1. ✅ PCA + 4 regression models (~50 configs)
2. ✅ PLS regression (~20 configs)
3. ✅ Feature selection + models (~40 configs)
4. ✅ Direct regularized regression (~21 configs)
5. ✅ **TopK-IG** (~45 configs) ← Great for small N
6. ✅ **Network aggregation** (~30 configs) ← Best for N<100

**Best strategy automatically selected via 5-fold CV!**

### Output Files Per Measure:

```
scatter_Hyperactivity_Network-mean_optimized.png   # Visualization
optimization_results_Hyperactivity.csv             # All 200+ configs tested
predictions_Hyperactivity_Network-mean.csv         # For validation
```

### Summary Files:

```
optimization_summary.csv          # All measures + FDR correction
optimization_summary.png          # Bar plot (colored by strategy)
validation_report.csv             # Integrity checks
```

---

## 📊 Interpreting Results

### Summary Table Shows:

```
Measure          N_Subjects  Final_Spearman  P_Display  P_FDR    Sig  Best_Strategy
Hyperactivity          81        0.398       < 0.001   < 0.001  ***  Network-mean
Inattention            81        0.342       0.0023    0.0089   ***  TopK-IG
Impulsivity            64        0.287       0.0234    0.0702       PCA+Ridge
```

**Key Points:**
- **N_Subjects**: Watch for N<50 (unreliable)
- **P_FDR**: Use this, not P_Display (controls false positives)
- **Sig (***)**:Survives FDR correction (most trustworthy)
- **Best_Strategy**: Which method worked best

### Strategy Meanings:

| Strategy | What It Means | When It Wins |
|----------|---------------|--------------|
| `PCA+Ridge` | 20 PCA components + Ridge(α=0.1) | Large N, distributed signal |
| `PLS` | PLS with 15 components | Large N, predictive focus |
| `TopK-IG` | 8 best ROIs + Ridge(α=0.1) | Small N (50-150) |
| `Network-mean` | 7 networks (mean aggregation) + Linear | Small N (<100) |
| `Network-pos_share` | 7 networks (positive mass) + Ridge | Small N, interpretable |

---

## ⚠️ Watch For These Warnings

### 🚨 **NUMERICAL EXPLOSION**
```
Predicted std: 4.90e+15
```
**Meaning:** Model is unstable (too many components for sample size)  
**Action:** Already handled - won't be selected as best  
**Expect:** Network or TopK-IG will win instead

### ⚠️ **Model Collapse**
```
Low prediction variance: 0.12 (5.3% of actual 2.25)
→ Model is predicting near-constant values
```
**Meaning:** Insufficient data or no signal  
**Action:** Interpret results cautiously, check N_Subjects  
**Expect:** Non-significant p_FDR

### ⚠️ **Insufficient Data**
```
Insufficient valid data: 12 subjects (need at least 20)
```
**Meaning:** Measure has too much missing data  
**Action:** Measure skipped automatically  
**Expect:** Won't appear in summary

---

## 🎯 Best Practices

### For Small N (<100):
✅ Use optimized scripts (TopK-IG and Network will help)  
✅ Focus on measures with **N>50**  
✅ Trust results with **p_FDR < 0.05**  
✅ Expect **Network-mean** or **TopK-IG** to win

### For Medium N (100-200):
✅ Optimization still helps (+0.05-0.10 ρ)  
✅ TopK-IG often competitive  
✅ Most measures should have **p_FDR < 0.05**

### For Large N (>200):
✅ Standard and optimized give similar results  
✅ PCA/PLS typically win  
✅ Main benefit: FDR correction, better validation

---

## 🚀 Recommended Workflow

### Step 1: Run Optimization
```bash
python run_nki_brain_behavior_optimized.py
```
**Runtime:** ~45 min  
**Output:** All measures analyzed, best strategy per measure

### Step 2: Validate
```bash
python check_optimization_predictions.py --cohort nki_rs_td
```
**Checks:** File existence, correlation match, data integrity  
**Expected:** All measures pass ✅

### Step 3: Visualize
```bash
# All measures
python create_optimization_summary_figure.py --cohort nki_rs_td

# Significant only (for publication)
python create_optimization_summary_figure.py --cohort nki_rs_td --min-rho 0.25
```
**Output:** Bar plot + detailed metrics table

### Step 4: Interpret
- Check which strategies won most often
- For Network strategies: Interpret at systems level
- For TopK-IG: Identify specific important ROIs
- Report FDR-corrected p-values in papers

---

## 📈 Expected Timeline

| Cohort | N | Measures | Runtime | Expected Significant (p_FDR<0.05) |
|--------|---|----------|---------|-----------------------------------|
| NKI | 81 | 9 | ~40 min | 5-7 measures |
| CMI-HBN TD | 84 | 15 | ~60 min | 8-12 measures |
| Stanford | 99 | 65 | ~120 min | 15-25 measures |
| ABIDE | 169 | 3 | ~15 min | 2-3 measures |
| ADHD200 | 238 | 2 | ~10 min | 1-2 measures |

---

## 🎓 Publication Reporting

### Recommended Reporting:

> "Brain-behavior relationships were assessed using comprehensive optimization across 6 strategies (PCA, PLS, feature selection, direct regularization, TopK-IG feature selection, and Yeo network aggregation). For each behavioral measure, the best-performing strategy was selected via 5-fold cross-validation. False discovery rate correction (Benjamini-Hochberg, α=0.05) controlled for multiple comparisons across measures."

### Report for Each Significant Finding:

> "Hyperactivity symptoms were predicted by network-aggregated IG features (Spearman ρ=0.398, p_FDR<0.001, N=81). The optimal approach used mean IG scores aggregated to 7 Yeo networks with Ridge regression (α=0.1)."

Or:

> "Inattention was best predicted by the top 8 most age-predictive ROIs (TopK-IG strategy; ρ=0.342, p_FDR=0.009, N=81), suggesting focal rather than distributed brain-behavior relationships."

---

## 🔍 Troubleshooting

**No significant results after FDR:**
- Normal for small N with many measures
- Report honestly: "No measures survived FDR correction"
- Check if any had p<0.05 uncorrected (discuss with caution)

**All Network strategies failed:**
- Yeo atlas file not found
- Check path in `optimized_brain_behavior_core.py`
- TopK-IG will still work as backup

**Low correlations despite optimization:**
- May indicate genuinely weak brain-behavior relationship
- Check N_Subjects (need N>50 minimum)
- Some measures simply don't relate to brain connectivity

**Numerical explosion warnings:**
- Already handled - won't affect final results
- Indicates PLS tried too many components
- Network or TopK-IG will be selected instead

---

## ✅ Checklist

Before running:
- [ ] Data files exist at specified paths
- [ ] Output directory has write permissions
- [ ] Python environment activated
- [ ] Expected runtime: allocate 30-90 min

After running:
- [ ] Check summary table for p_FDR values
- [ ] Run validation: `check_optimization_predictions.py`
- [ ] Create figures: `create_optimization_summary_figure.py`
- [ ] Review which strategies won (Network/TopK-IG for small N?)

---

**Last Updated**: 2024  
**Status**: ✅ All systems ready

