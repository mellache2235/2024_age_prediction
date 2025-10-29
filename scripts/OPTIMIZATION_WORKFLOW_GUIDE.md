# Complete Optimization Workflow Guide

## 🚀 Step-by-Step Workflow

### Step 1: Run Optimization

```bash
# Run optimization for your cohort
python run_all_cohorts_brain_behavior_optimized.py --cohort stanford_asd

# Or for all cohorts
python run_all_cohorts_brain_behavior_optimized.py --all
```

**Runtime**: ~30-60 minutes per cohort  
**Output**: Individual scatter plots + optimization results

---

### Step 2: Create Summary Figures

```bash
# Create summary of significant results
python create_optimization_summary_figure.py --cohort stanford_asd
```

**Runtime**: < 1 minute  
**Output**: Summary tables and figures

---

### Step 3: Review Results

#### Check Significant Measures

```bash
# View summary table
cat /oak/.../brain_behavior/stanford_asd_optimized/stanford_asd_optimization_summary_significant.csv
```

Look for:
- **Final_Spearman**: Main correlation (higher is better)
- **Final_P_Value**: Significance (< 0.05 typically)
- **Final_R2**: Model fit (should be positive and reasonable, e.g., 0.0-0.5)
- **CV_Spearman**: Cross-validated correlation (should be similar to Final)

#### Red Flags 🚩

Exclude results with:
- ❌ **Extreme R²** (< -10 or > 10): Indicates overfitting/failure
- ❌ **Very large MAE**: Model predictions are way off
- ❌ **Large gap** between CV and Final Spearman: Overfitting

---

### Step 4: Use Results for Publication

#### For Manuscript

**Text**:
```
Brain-behavior correlations were optimized using comprehensive hyperparameter 
search across 4 strategies: PCA+Regression, PLS Regression, Feature Selection, 
and Direct Regularized Regression. Approximately 100-200 configurations were 
tested per behavioral measure using 5-fold cross-validation to maximize 
Spearman correlation. The best model for [measure] achieved ρ = [value], 
p = [value] using [strategy] with [model].
```

**Figures**:
1. Use individual scatter plots for specific measures:
   - `scatter_[measure]_optimized.png` (for slides)
   - `scatter_[measure]_optimized.tiff` (for manuscript)
   - `scatter_[measure]_optimized.ai` (for editing)

2. Use summary figure for overview:
   - `stanford_asd_optimization_summary_figure.png`
   - `stanford_asd_correlations_barplot.png`

**Tables**:
- `stanford_asd_optimization_summary_significant.csv` (import to Excel/Word)
- `stanford_asd_optimization_summary_significant.md` (for GitHub/reports)

---

## 📊 Example Complete Workflow

### Example: Stanford ASD

```bash
# 1. Run optimization (takes ~30-60 min)
python run_all_cohorts_brain_behavior_optimized.py --cohort stanford_asd

# Output shows:
# ✓ social_awareness_tscore: ρ = 0.293, p = 0.0033
# ✗ srs_total_score_standard: ρ = -0.022, p = 0.827 (not significant)

# 2. Create summary (takes < 1 min)
python create_optimization_summary_figure.py --cohort stanford_asd

# Output:
# 📊 Significant Measures:
#   ✓ social_awareness_tscore: ρ = 0.293, p = 0.0033, R² = 0.095
# Significant results: 1/2

# 3. Check output files
ls /oak/.../brain_behavior/stanford_asd_optimized/

# You'll see:
# - scatter_social_awareness_tscore_optimized.png       ← Use for manuscript!
# - optimization_summary.csv                            ← All measures
# - stanford_asd_optimization_summary_significant.csv   ← Only significant
# - stanford_asd_optimization_summary_figure.png        ← Overview figure
# - stanford_asd_correlations_barplot.png               ← Bar plot
```

---

## 🎯 Understanding Your Results

### Good Result Example
```
Measure: social_awareness_tscore
Spearman ρ: 0.293
P-value: 0.0033
R²: 0.095
Strategy: FeatureSelection+Lasso
```

✅ **Interpretation**:
- Moderate positive correlation (ρ = 0.29)
- Highly significant (p < 0.01)
- Reasonable R² (10% variance explained)
- Feature selection worked best

✅ **Use this result** - it's publication-quality!

---

### Failed Result Example
```
Measure: srs_total_score_standard
Spearman ρ: -0.022
P-value: 0.827
R²: -6484.483
MAE: 428.25
```

❌ **Interpretation**:
- No correlation (ρ ≈ 0)
- Not significant (p > 0.05)
- **Extreme negative R²** - severe overfitting
- Very high error

❌ **Do NOT use this result** - the model failed!

---

## 📁 Complete File Organization

After running optimization + summary, you'll have:

```
/oak/.../brain_behavior/cohort_name_optimized/
│
├── optimization_summary.csv                              # All measures (raw)
├── cohort_name_optimization_summary_significant.csv      # Significant only ⭐
├── cohort_name_optimization_summary_significant.md       # Markdown table
│
├── cohort_name_optimization_summary_figure.png           # Multi-panel overview
├── cohort_name_optimization_summary_figure.tiff          # High-res
├── cohort_name_optimization_summary_figure.ai            # Vector
│
├── cohort_name_correlations_barplot.png                  # Bar chart (all measures)
├── cohort_name_correlations_barplot.tiff                 # High-res
│
├── scatter_measure1_optimized.png                        # Individual plots
├── scatter_measure1_optimized.tiff                       # (for each measure)
├── scatter_measure1_optimized.ai
│
├── optimization_results_measure1.csv                     # All configs tested
└── optimization_results_measure2.csv                     # (for each measure)
```

---

## 💡 Pro Tips

### 1. **Start with Test Mode**
```bash
# Test with 2 measures first (~5 min)
python run_all_cohorts_brain_behavior_optimized.py --cohort stanford_asd --max-measures 2

# Then run full if results look good
python run_all_cohorts_brain_behavior_optimized.py --cohort stanford_asd
```

### 2. **Adjust Significance Thresholds**
```bash
# Stricter (for high-quality cohorts)
python create_optimization_summary_figure.py --cohort abide_asd --min-rho 0.3 --max-pvalue 0.01

# More lenient (for exploratory analysis)
python create_optimization_summary_figure.py --cohort adhd200_td --min-rho 0.15 --max-pvalue 0.10
```

### 3. **Compare Across Cohorts**
```bash
# Run for multiple cohorts
for cohort in abide_asd adhd200_td cmihbn_adhd; do
    python create_optimization_summary_figure.py --cohort $cohort
done

# Then compare which cohorts/measures show strongest correlations
```

### 4. **Check CV vs Final Correlation**
- If **CV ≈ Final**: Good! Model generalizes well
- If **CV << Final**: Warning! Might be overfitting
- If **CV >> Final**: Unusual, check for data issues

---

## 🚨 Common Issues

### "No significant results found"
**Solutions**:
- Lower `--min-rho` threshold
- Increase `--max-pvalue` threshold
- Check if optimization actually found any correlations
- Some behavioral measures just don't correlate strongly!

### "R² is extremely negative"
**This means**: Severe overfitting, model failed
**Solution**: Exclude this result, try different behavioral measure

### "Directory not found"
**Solution**: Run optimization first!
```bash
python run_all_cohorts_brain_behavior_optimized.py --cohort stanford_asd
```

---

## ✅ Workflow Checklist

- [ ] Run optimization for cohort
- [ ] Check console output for significant results
- [ ] Run summary figure script
- [ ] Review `optimization_summary_significant.csv`
- [ ] Check for red flags (extreme R², high MAE)
- [ ] Copy publication-ready figures to manuscript folder
- [ ] Document which strategy/model worked best
- [ ] Cite hyperparameter optimization in methods

---

**Last Updated**: October 2024  
**Ready to Use**: ✅ Yes

