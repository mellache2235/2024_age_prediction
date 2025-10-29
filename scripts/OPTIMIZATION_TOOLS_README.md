# Optimization Analysis Tools

Two utility scripts for analyzing and validating brain-behavior optimization results.

---

## 1. `create_optimization_summary_figure.py`

Creates comprehensive summary visualizations from optimization results.

### Features:
- **Bar plot** showing Spearman correlations for all behavioral measures
- Color-coded by optimization strategy (PCA, PCA+Selection, Selection, Raw)
- Significance markers (*, **, ***)
- **Detailed metrics table** with key performance indicators
- **Summary statistics** printed to console
- Filter by minimum correlation threshold

### Usage:

```bash
# Single cohort
python create_optimization_summary_figure.py --cohort stanford_asd

# With minimum correlation filter (only show |ρ| >= 0.25)
python create_optimization_summary_figure.py --cohort abide_asd --min-rho 0.25

# All cohorts
python create_optimization_summary_figure.py --all

# All cohorts with filter
python create_optimization_summary_figure.py --all --min-rho 0.30
```

### Output Files:
- `optimization_summary.png` - Bar plot of all correlations
- `optimization_summary.pdf` - Publication-quality PDF version
- `optimization_metrics.csv` - Detailed table with all metrics
- If using `--min-rho X`, files will be named `optimization_summary_minrhoX.XX.*`

### Available Cohorts:
- `abide_asd` - ABIDE ASD
- `stanford_asd` - Stanford ASD
- `adhd200_td` - ADHD200 TD
- `adhd200_adhd` - ADHD200 ADHD
- `cmihbn_td` - CMI-HBN TD
- `cmihbn_adhd` - CMI-HBN ADHD
- `nki_rs_td` - NKI-RS TD

---

## 2. `check_optimization_predictions.py`

Validates prediction integrity and model performance.

### Features:
- Verifies prediction files exist for all measures
- Checks data integrity (no NaN, Inf, constant values)
- Validates that computed correlations match reported values
- Checks residual calculations
- Generates comprehensive validation report

### Usage:

```bash
# Single cohort
python check_optimization_predictions.py --cohort stanford_asd

# All cohorts
python check_optimization_predictions.py --all
```

### Validation Checks:
1. **File existence**: All prediction files present
2. **Data integrity**: No NaN, Inf, or constant values
3. **Correlation match**: Computed ρ matches reported ρ (within 0.001)
4. **Residual accuracy**: Residuals = Actual - Predicted
5. **Sample size**: Correct number of subjects

### Output Files:
- `validation_report.csv` - Detailed validation results for each measure

### What to Look For:

✅ **All checks passed**: Models working correctly
- Files found ✓
- Data integrity OK ✓
- Correlation match ✓
- No issues ✓

⚠️ **Issues found**: Review validation report
- Missing files → Re-run optimization
- Correlation mismatch → Check model implementation
- Data integrity issues → Check data preprocessing

---

## Example Workflow

```bash
# 1. Run optimization for a cohort
python run_all_cohorts_brain_behavior_optimized.py --cohort stanford_asd

# 2. Create summary visualizations
python create_optimization_summary_figure.py --cohort stanford_asd

# 3. Validate predictions
python check_optimization_predictions.py --cohort stanford_asd

# 4. If validations pass, create filtered summary for publication
python create_optimization_summary_figure.py --cohort stanford_asd --min-rho 0.25
```

---

## Interpreting Results

### Summary Figure:
- **Bar length**: Spearman correlation strength
- **Color**: Optimization strategy used
- **Asterisks**: Significance level
  - `***` p < 0.001
  - `**` p < 0.01
  - `*` p < 0.05

### Strategy Colors:
- **Blue**: PCA only
- **Orange**: PCA + Feature Selection
- **Green**: Feature Selection only
- **Red**: Raw features (no dimensionality reduction)

### Validation Report:
- `Correlation_Match = True`: Predictions verified ✓
- `Data_Integrity_OK = True`: No data issues ✓
- `Issues = None`: All checks passed ✓

---

## Troubleshooting

**No prediction files found:**
- Optimization may not have completed
- Check if `optimization_summary.csv` exists
- Re-run optimization script

**Correlation mismatch:**
- Small differences (<0.001) are acceptable (numerical precision)
- Large differences suggest model inconsistency
- Check optimization logs

**Missing measures:**
- Some measures may have insufficient data
- Check optimization logs for warnings
- Normal to have fewer prediction files than total measures

---

## Notes

- Both scripts use the same cohort configuration as the optimization scripts
- Results directories are automatically determined from cohort names
- Scripts are designed to work with output from `run_all_cohorts_brain_behavior_optimized.py`
- Font styling matches main analysis plots (Arial, standardized formatting)

---

**Created**: 2024
**Menon Lab**, Stanford University

