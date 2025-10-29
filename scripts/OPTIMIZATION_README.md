# Brain-Behavior Analysis Optimization Guide

## Overview

This guide describes the optimized brain-behavior analysis scripts that maximize Spearman correlations between IG (Integrated Gradients) features and behavioral measures.

## Key Scripts

### 1. `run_stanford_asd_brain_behavior_optimized.py`

**Purpose**: Comprehensive optimization for Stanford ASD cohort to maximize brain-behavior correlations.

**Key Features**:
- **4 Optimization Strategies** tested in parallel:
  1. **PCA + Regression Models** (Linear, Ridge, Lasso, ElasticNet)
  2. **PLS Regression** (Partial Least Squares - optimized for prediction tasks)
  3. **Feature Selection + Regression** (F-statistic and Mutual Information)
  4. **Direct Regularized Regression** (no dimensionality reduction)

- **Comprehensive Hyperparameter Search**:
  - PC components: 5-50 (step=5, adjusted by sample size)
  - PLS components: 3-30 (step=3)
  - Alpha values: [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
  - Feature selection: Top-K features [50, 100, 150, 200]

- **Robust Evaluation**:
  - 5-fold cross-validation for performance estimation
  - Spearman correlation as primary metric
  - Automatic outlier removal (IQR method, 3× threshold)
  - Data integrity checks throughout pipeline

- **Parallel Processing**:
  - Optional parallel processing of multiple behavioral measures
  - Control with `--n-jobs` flag

- **Output**:
  - Scatter plots (PNG + TIFF + AI formats)
  - Optimization results CSV (all tested configurations)
  - Summary CSV (best configuration per behavioral measure)

**Usage**:

```bash
# Run with all behavioral measures (parallel processing)
python run_stanford_asd_brain_behavior_optimized.py

# Test mode (limit to 3 measures)
python run_stanford_asd_brain_behavior_optimized.py --max-measures 3

# Sequential processing (1 core)
python run_stanford_asd_brain_behavior_optimized.py --n-jobs 1

# Control parallelism (4 cores)
python run_stanford_asd_brain_behavior_optimized.py --n-jobs 4
```

**Output Directory**:
```
/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/stanford_asd_optimized/
```

---

### 2. `run_stanford_asd_brain_behavior_enhanced.py` (Original)

**Purpose**: Standard brain-behavior analysis without extensive optimization.

**Key Features**:
- PCA with elbow plot for component selection
- Linear regression with optimal PCs (80% variance threshold)
- PC importance analysis
- PC loadings extraction

**When to Use**:
- Quick analysis with standard approach
- Exploratory analysis
- When optimization overhead is not needed

---

## Comparison: Enhanced vs. Optimized

| Feature | Enhanced | Optimized |
|---------|----------|-----------|
| **Models Tested** | Linear Regression only | 4 strategies × multiple models |
| **PC Selection** | Elbow plot (80% variance) | Grid search (5-50) |
| **Additional Methods** | PCA only | PCA, PLS, Feature Selection, Direct |
| **Hyperparameter Tuning** | None | Extensive (alpha, components, k features) |
| **Cross-Validation** | No | 5-fold CV |
| **Feature Selection** | No | F-statistic, Mutual Information |
| **Parallel Processing** | No | Yes (optional) |
| **Expected Correlations** | Moderate | **Maximized** |
| **Computation Time** | ~2-5 min | ~20-60 min (depending on measures) |

---

## Why Optimization Matters for Brain-Behavior Correlations

### 1. **PLS vs. PCA**
- PCA maximizes variance in features (X)
- PLS maximizes covariance between features (X) and behavior (y)
- **For prediction tasks, PLS often outperforms PCA**

### 2. **Feature Selection**
- Brain regions have different relevance to behavioral measures
- Selecting top-K features can reduce noise and improve correlations
- F-statistic: Linear relationships
- Mutual Information: Non-linear relationships

### 3. **Regularization**
- Ridge: Shrinks all coefficients (good for multicollinearity)
- Lasso: Performs feature selection (sets some coefficients to 0)
- ElasticNet: Combination of Ridge and Lasso
- **Optimal alpha varies per behavioral measure**

### 4. **Cross-Validation**
- Prevents overfitting
- Provides realistic performance estimates
- Ensures generalizability

---

## Understanding the Output

### 1. **Optimization Summary CSV**
```csv
Measure,N_Subjects,Best_Strategy,Best_Model,Best_N_Components,Best_Alpha,CV_Spearman,Final_Spearman,Final_P_Value,Final_R2,Final_MAE
srs_total_score_standard,89,PLS,PLS,15,,0.4523,0.4812,0.0001,0.2314,8.42
social_awareness_tscore,87,PCA+Ridge,Ridge,25,1.0,0.3891,0.4123,0.0008,0.1701,9.18
```

**Columns**:
- `Measure`: Behavioral measure name
- `N_Subjects`: Sample size after filtering
- `Best_Strategy`: Winning optimization strategy
- `Best_Model`: Winning regression model
- `Best_N_Components`: Optimal number of components (if applicable)
- `Best_Alpha`: Optimal regularization strength (if applicable)
- `CV_Spearman`: Cross-validated Spearman ρ
- `Final_Spearman`: **Final Spearman correlation on full data** ⭐
- `Final_P_Value`: Statistical significance
- `Final_R2`: Coefficient of determination
- `Final_MAE`: Mean absolute error

### 2. **Optimization Results CSV** (per behavioral measure)
```csv
strategy,model,n_components,alpha,mean_cv_spearman,std_cv_spearman
PLS,PLS,15,,0.4523,0.0812
PCA+Ridge,Ridge,25,1.0,0.4312,0.0921
PCA+Lasso,Lasso,20,0.1,0.4201,0.0845
...
```

Contains all tested configurations sorted by cross-validated Spearman ρ.

---

## Tips for Maximizing Correlations

### 1. **Data Quality**
- Remove outliers (already automated)
- Check for missing data
- Ensure behavioral measures are properly scaled
- Verify subject ID alignment

### 2. **Sample Size**
- Larger samples enable more components
- Minimum recommended: N ≥ 30 per behavioral measure
- Optimal: N ≥ 100

### 3. **Feature Quality**
- IG scores should be normalized
- Check for highly correlated ROIs (multicollinearity)
- Consider removing low-variance features

### 4. **Model Selection**
- **For linear relationships**: Try PCA + Linear/Ridge first
- **For complex relationships**: Try PLS and Feature Selection
- **For sparse solutions**: Try Lasso/ElasticNet

### 5. **Interpretation**
- Focus on cross-validated correlations (more realistic)
- Check if best model is consistent across behavioral measures
- Examine feature importance from best model

---

## Troubleshooting

### Issue: Very low correlations (ρ < 0.2)
**Possible Causes**:
- Weak brain-behavior relationship
- Too much noise in behavioral measure
- Insufficient sample size
- IG features not relevant to behavior

**Solutions**:
- Try different behavioral measures
- Check data quality
- Increase regularization
- Consider non-linear models (future work)

### Issue: High CV ρ but low final ρ
**Possible Cause**: Overfitting in CV

**Solution**:
- Increase regularization (higher alpha)
- Reduce number of components
- Use simpler model (e.g., Ridge instead of Lasso)

### Issue: Script crashes or hangs
**Possible Causes**:
- Insufficient memory
- Too many parallel jobs
- Mutual information computation hanging

**Solutions**:
```bash
# Use fewer parallel jobs
python run_stanford_asd_brain_behavior_optimized.py --n-jobs 1

# Test with fewer measures first
python run_stanford_asd_brain_behavior_optimized.py --max-measures 2
```

---

## Future Enhancements

1. **Non-linear models**: Support Vector Regression, Random Forest
2. **Ensemble methods**: Stacking, Voting regressors
3. **Nested CV**: Inner loop for hyperparameter tuning
4. **Bayesian optimization**: More efficient hyperparameter search
5. **Multi-output models**: Predict multiple behavioral measures simultaneously
6. **Deep learning**: Neural network regressors

---

## References

### Methodological Papers
- PLS for brain-behavior: Krishnan et al. (2011) *NeuroImage*
- Feature selection in neuroimaging: Michel et al. (2012) *Frontiers in Neuroinformatics*
- Cross-validation best practices: Varoquaux et al. (2017) *NeuroImage*

### Implementation
- scikit-learn documentation: https://scikit-learn.org/
- PLS Regression: https://scikit-learn.org/stable/modules/cross_decomposition.html

---

## Contact

For questions or issues, contact the research team or refer to the main project README.

**Last Updated**: October 2024

