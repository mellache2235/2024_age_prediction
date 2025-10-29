# Optimization File Naming Convention

## Overview

All optimized brain-behavior analysis files now include the optimization method in their filenames for easy identification.

---

## File Naming Format

### Scatter Plots

**Format**: `scatter_{measure}_{method}_{params}_optimized.{ext}`

**Examples**:
```
# PLS Regression with 15 components
scatter_social_awareness_tscore_PLS_comp15_optimized.png
scatter_social_awareness_tscore_PLS_comp15_optimized.tiff
scatter_social_awareness_tscore_PLS_comp15_optimized.ai

# PCA + Ridge with 25 components
scatter_ados_total_PCA_Ridge_comp25_optimized.png

# Feature Selection + Lasso with 100 features
scatter_Hyper_Impulsive_FeatureSelection_Lasso_k100_optimized.png

# Direct Ridge (no dimensionality reduction)
scatter_Inattentive_DirectRidge_optimized.png
```

**Components**:
- `{measure}`: Behavioral measure name (underscores replace spaces)
- `{method}`: Optimization strategy (e.g., PLS, PCA_Ridge, FeatureSelection_Lasso)
- `{params}`: Method-specific parameters:
  - `comp{N}`: Number of PCA/PLS components
  - `k{N}`: Number of features selected
  - (omitted for Direct methods)
- Extensions: `.png`, `.tiff`, `.ai`

---

### Prediction Files

**Format**: `predictions_{measure}_{method}.csv`

**Examples**:
```
predictions_social_awareness_tscore_PLS.csv
predictions_ados_total_PCA_Ridge.csv
predictions_Hyper_Impulsive_FeatureSelection_Lasso.csv
```

**Contents**:
```csv
Actual,Predicted,Residual
65.20,63.40,1.80
72.10,74.30,-2.20
...
```

---

### Optimization Results

**Format**: `optimization_results_{measure}.csv`

**Examples**:
```
optimization_results_social_awareness_tscore.csv
optimization_results_ados_total.csv
```

**Contents**: All ~100-200 configurations tested, sorted by CV Spearman

---

### Summary Files

**Format**: `{cohort}_optimization_summary{_type}.{ext}`

**Examples**:
```
stanford_asd_optimization_summary.csv                    # All measures
stanford_asd_optimization_summary_significant.csv        # Filtered
stanford_asd_optimization_summary_figure.png             # Visual summary
stanford_asd_correlations_barplot.png                    # Bar chart
```

---

## Decoding Filenames

### Example 1: PLS Method
```
scatter_social_awareness_tscore_PLS_comp15_optimized.png
                                 │   │
                                 │   └─ 15 PLS components
                                 └───── PLS Regression
```

**Interpretation**: 
- Measure: Social Awareness T-Score
- Method: PLS Regression
- Components: 15
- This was the best performing configuration!

---

### Example 2: Feature Selection Method
```
scatter_ados_total_FeatureSelection_Lasso_k100_optimized.png
                    │                        │
                    │                        └─ 100 features selected
                    └────────────────────────── Feature Selection + Lasso
```

**Interpretation**:
- Measure: ADOS Total
- Method: Feature Selection + Lasso Regression
- Features: Top 100 (selected by F-statistic or MI)
- This beat all other methods for this measure!

---

### Example 3: PCA + Ridge Method
```
scatter_Hyper_Impulsive_PCA_Ridge_comp25_optimized.png
                         │        │
                         │        └─ 25 PCA components
                         └────────── PCA + Ridge Regression
```

**Interpretation**:
- Measure: Hyperactivity/Impulsivity
- Method: PCA then Ridge Regression
- Components: 25
- Ridge won over Lasso/ElasticNet for this measure

---

### Example 4: Direct Method
```
scatter_Inattentive_DirectRidge_optimized.png
                     │
                     └─ No dimensionality reduction
```

**Interpretation**:
- Measure: Inattention
- Method: Direct Ridge Regression
- All features used (no PCA/PLS)
- Regularization alone was sufficient!

---

## Method Name Mappings

| Strategy in Code | Filename Component | Description |
|------------------|-------------------|-------------|
| `PLS` | `PLS` | PLS Regression |
| `PCA+Linear` | `PCA_Linear` | PCA + Linear Regression |
| `PCA+Ridge` | `PCA_Ridge` | PCA + Ridge Regression |
| `PCA+Lasso` | `PCA_Lasso` | PCA + Lasso Regression |
| `PCA+ElasticNet` | `PCA_ElasticNet` | PCA + ElasticNet |
| `FeatureSelection+Ridge` | `FeatureSelection_Ridge` | Feature Selection + Ridge |
| `FeatureSelection+Lasso` | `FeatureSelection_Lasso` | Feature Selection + Lasso |
| `DirectRidge` | `DirectRidge` | Direct Ridge (all features) |
| `DirectLasso` | `DirectLasso` | Direct Lasso (all features) |
| `DirectElasticNet` | `DirectElasticNet` | Direct ElasticNet |

---

## Benefits of New Naming

### 1. **Immediate Identification**
Just from filename, you know:
- Which measure
- Which method won
- Key parameters (components or features)

### 2. **Easy Comparison**
```bash
ls -1 *.png
scatter_ados_total_PLS_comp15_optimized.png
scatter_ados_comm_FeatureSelection_Lasso_k100_optimized.png
scatter_ados_social_PCA_Ridge_comp20_optimized.png
```
Can instantly see which methods worked best for each measure!

### 3. **Reproducibility**
Filename tells you exactly how to reproduce the result:
- `PLS_comp15` = Use PLS with 15 components
- `FeatureSelection_Lasso_k100` = Use F-stat/MI to select top 100, then Lasso
- `PCA_Ridge_comp25` = Use PCA with 25 components, then Ridge

---

## Example Output Directory

After running optimization for Stanford ASD:

```
/oak/.../brain_behavior/stanford_asd_optimized/
├── optimization_summary.csv
│
├── scatter_social_awareness_tscore_FeatureSelection_Lasso_k100_optimized.png
├── scatter_social_awareness_tscore_FeatureSelection_Lasso_k100_optimized.tiff
├── scatter_social_awareness_tscore_FeatureSelection_Lasso_k100_optimized.ai
│
├── scatter_srs_total_score_standard_DirectLasso_optimized.png
├── scatter_srs_total_score_standard_DirectLasso_optimized.tiff
├── scatter_srs_total_score_standard_DirectLasso_optimized.ai
│
├── predictions_social_awareness_tscore_FeatureSelection_Lasso.csv
├── predictions_srs_total_score_standard_DirectLasso.csv
│
├── optimization_results_social_awareness_tscore.csv
└── optimization_results_srs_total_score_standard.csv
```

**From filenames alone**, you can see:
- Social awareness: Feature Selection + Lasso with 100 features worked best
- SRS total: Direct Lasso worked best (but check the results - this one failed!)

---

## Finding Specific Methods

### Find all PLS results:
```bash
ls *_PLS_*.png
```

### Find all Feature Selection results:
```bash
ls *_FeatureSelection_*.png
```

### Find all results with 15 components:
```bash
ls *_comp15_*.png
```

### Find all Lasso results:
```bash
ls *_Lasso_*.png
```

---

## Backward Compatibility

Old filename format:
```
scatter_measure_optimized.png
```

New filename format:
```
scatter_measure_METHOD_PARAMS_optimized.png
```

**Change**: Method and parameters inserted before `_optimized`

---

## Summary Files (No Method in Name)

These aggregate across all measures, so no method in filename:
- `{cohort}_optimization_summary.csv`
- `{cohort}_optimization_summary_significant.csv`
- `{cohort}_optimization_summary_figure.png`
- `{cohort}_correlations_barplot.png`

---

## ✅ Implementation Complete

All scripts updated to include method in filenames:
- ✅ `run_all_cohorts_brain_behavior_optimized.py`
- ✅ `run_stanford_asd_brain_behavior_optimized.py`
- ✅ NKI script (on Oak) - same pattern

**Benefit**: Filenames now tell the complete story! 📊

---

**Date**: October 2024  
**Status**: ✅ Implemented  
**Coverage**: All optimization scripts

