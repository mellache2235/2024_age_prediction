# Brain Age Prediction Analysis Pipeline

Comprehensive pipeline for analyzing brain age prediction using spatiotemporal deep neural networks (stDNN) with Integrated Gradients feature attribution across TD, ADHD, and ASD cohorts.

---
<img width="868" height="609" alt="image" src="https://github.com/user-attachments/assets/f5b32844-55aa-49b6-9cd6-44d7a2313484" />

## 🧠 Study Overview

We develop a normative brain-age prediction model using stDNN trained on HCP-Development cohort, validated on independent TD and clinical cohorts (ADHD, ASD). Integrated Gradients yields "brain fingerprints" identifying neurobiological features underlying psychiatric disorders.

**Pipeline**: Model Development → Validation → Feature Attribution → Brain-Behavior Analysis

---

## 🔧 Environment Setup

### Quick Start
```bash
# Activate environment
conda activate /oak/stanford/groups/menon/software/python_envs/brain_age_2024

# Verify
python scripts/verify_imports.py
```

### Install New Environment
See [`INSTALL.md`](INSTALL.md) for complete installation instructions.

---

## 🚀 Running the Pipeline

### Complete Workflow (Copy-Paste Ready)

```bash
cd /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/scripts

# 1. Network Analysis
python network_analysis_yeo.py --process_all
python network_analysis_yeo.py --process_shared
python create_region_tables.py \
  --config /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/config.yaml \
  --output_dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/region_tables

# 2. Statistical Comparisons  
python run_statistical_comparisons.py

# 3. Brain-Behavior Analysis
# All scripts use plot_styles.py for consistent formatting
# Export: PNG + TIFF + AI (publication-ready, no post-processing)

# TD Cohorts
python run_nki_brain_behavior_enhanced.py           # NKI-RS (CAARS)
python run_adhd200_brain_behavior_enhanced.py       # ADHD-200 TD Subset (NYU)
python run_cmihbn_brain_behavior_enhanced.py        # CMI-HBN TD (C3SR)

# ADHD Cohorts
python run_adhd200_adhd_brain_behavior_enhanced.py  # ADHD-200 ADHD
python run_cmihbn_adhd_brain_behavior_enhanced.py   # CMI-HBN ADHD (C3SR)

# ASD Cohorts
python run_stanford_asd_brain_behavior_enhanced.py  # Stanford ASD (SRS Total Score)
python run_abide_asd_brain_behavior_enhanced.py     # ABIDE ASD (ADOS)

# 🚀 OPTIMIZED Brain-Behavior Analysis (Comprehensive Strategy Testing)
# Tests 6 strategies with ~200-400 configurations per behavioral measure
# Features: TopK-IG, Network Aggregation, FDR Correction, PLS stability limits

# Optimization Strategies:
# 1. PCA + Regression (Linear, Ridge, Lasso, ElasticNet)
# 2. PLS Regression (adaptive component limits for numerical stability)
# 3. Feature Selection + Regression
# 4. Direct Regularized Regression
# 5. TopK-IG (selects top ROIs by IG importance - excellent for small N)
# 6. Network Aggregation (246 ROIs → 7-17 Yeo networks - best for N<100)

# Universal script for all cohorts
python run_all_cohorts_brain_behavior_optimized.py --cohort adhd200_td
python run_all_cohorts_brain_behavior_optimized.py --cohort adhd200_adhd
python run_all_cohorts_brain_behavior_optimized.py --cohort cmihbn_td
python run_all_cohorts_brain_behavior_optimized.py --cohort cmihbn_adhd
python run_all_cohorts_brain_behavior_optimized.py --cohort abide_asd

# Cohort-specific scripts (recommended - better data handling)
python run_stanford_asd_brain_behavior_optimized.py   # Stanford ASD (SRS, parallel processing)
python run_abide_asd_brain_behavior_optimized.py      # ABIDE ASD (ADOS, ID matching)
python run_nki_brain_behavior_optimized.py            # NKI (Hyperactivity/Inattention/Impulsivity only)

# All optimized scripts now include:
# ✓ 6 optimization strategies (including TopK-IG and Network Aggregation)
# ✓ FDR correction (Benjamini-Hochberg) for multiple comparisons
# ✓ Prediction integrity checks (detects model collapse, numerical instability)
# ✓ Sample size reporting (N_Subjects in summary tables)
# ✓ Enhanced data cleaning (missing code filtering, pickle/gzip auto-detection)

# Runtime: ~30-90 min per cohort
# Expected: +10-30% higher correlations vs. standard analysis
# Docs: scripts/TOP_K_IG_STRATEGY.md, scripts/NETWORK_AGGREGATION_STRATEGY.md

# Validation & Visualization Tools
python check_optimization_predictions.py --cohort stanford_asd        # Verify integrity
python create_optimization_summary_figure.py --cohort stanford_asd    # Summary plots
python create_optimization_summary_figure.py --cohort abide_asd --min-rho 0.25  # Filter

# 🧠 Network-Level Analysis (Separate from optimization)
# Dedicated analysis using Yeo network-aggregated features (7-17 networks)
# For detailed network-level insights and interpretation
python run_network_brain_behavior_analysis.py --cohort nki_rs_td
python run_network_brain_behavior_analysis.py --cohort cmihbn_td --method pos_share
python run_network_brain_behavior_analysis.py --all  # All cohorts

# 4. Network IG Correlations (Age & Behavior)
python compute_network_age_correlations.py \
  --preset brain_age_td \
  --target-key Predicted_Brain_Age:brain_age_pred \
  --apply-fdr \
  --scatter-plots \
  --scatter-output-dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_correlations/plots \
  --output-dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_correlations
```

Run TD correlations (ages and predicted brain age) using the built-in preset:
```bash
python compute_network_age_correlations.py \
  --preset brain_age_td \
  --target-key Predicted_Brain_Age:brain_age_pred \
  --apply-fdr \
  --scatter-plots
```

For brain-behavior cohorts:
```bash
python compute_network_age_correlations.py \
  --preset brain_behavior_adhd200 \
  --apply-fdr \
  --scatter-plots
```
Presets are defined in `config/network_correlation_presets.yaml` and include IG directories, age sources, and behavior target sources, so the CLI stays minimal.

python compute_network_age_correlations.py \
  --datasets adhd200_adhd_optimized \
  --root-dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior \
  --parcellation yeo17 \
  --aggregation-method pos_share \
  --skip-chronological \
  --target-key Hyperactivity_Observed:y_true_hyperactivity \
  --target-key Hyperactivity_Predicted:y_pred_hyperactivity \
  --target-source Hyperactivity_Observed=adhd200_adhd_optimized::/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/adhd200_adhd_optimized/predictions.csv::subject_id::Hyperactivity_Observed \
  --target-source Hyperactivity_Predicted=adhd200_adhd_optimized::/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/adhd200_adhd_optimized/predictions.csv::subject_id::Hyperactivity_Predicted \
  --apply-fdr \
  --save-subject-level \
  --output-dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_correlations_behavior

# 5. Combined Plots
python plot_brain_behavior_td_cohorts.py
python plot_pc_loadings_heatmap.py --dataset nki_rs_td
python plot_pc_loadings_heatmap.py --dataset adhd200_td
python plot_pc_loadings_heatmap.py --dataset cmihbn_td

# 6. Brain Age Plots
# Note: TD cohorts (NKI, CMI-HBN TD, ADHD200 TD) use _oct25 NPZ files from:
#   /oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/
#   (exception: these NPZ bundles remain in the original repo, not `_test`)

python plot_brain_age_td_cohorts.py \
  --output_dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_age_plots
python plot_brain_age_adhd_cohorts.py \
  --output_dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_age_plots
python plot_brain_age_asd_cohorts.py \
  --output_dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_age_plots
```

---

## 📁 Output Locations

```
results/
├── network_analysis_yeo/        # Network statistics & radar plots
├── region_tables/               # ROI tables (diverse subsets for manuscripts)
├── statistical_comparisons/     # Cosine similarity, Spearman ρ, Aitchison, JS divergence
├── brain_behavior/              # Individual scatter plots (PNG/TIFF/AI), 6×6 inch
│   ├── nki_rs_td/
│   ├── adhd200_td/
│   ├── adhd200_adhd/
│   ├── cmihbn_td/
│   ├── cmihbn_adhd/
│   ├── stanford_asd/           # SRS Total Score, Social Awareness
│   ├── abide_asd/              # ADOS (total, social, comm)
│   ├── stanford_asd_optimized/ # 🚀 Optimized (filenames include method!)
│   ├── abide_asd_optimized/    # 🚀 Optimized
│   ├── adhd200_td_optimized/   # 🚀 Optimized
│   ├── adhd200_adhd_optimized/ # 🚀 Optimized
│   ├── cmihbn_td_optimized/    # 🚀 Optimized
│   ├── cmihbn_adhd_optimized/  # 🚀 Optimized
│   ├── nki_rs_td_optimized/    # 🚀 Optimized
│   └── combined_plots/
```

**Note**: Optimized output filenames include the method used (e.g., `scatter_measure_PLS_comp15_optimized.png`) for easy identification of which optimization strategy worked best.

```
├── brain_age_plots/             # Combined scatter plots (PNG/TIFF/AI)
├── integrated_gradients/        # IG scores CSV files with subject IDs
└── count_data/                  # Consensus features CSVs
```

**All paths**: `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/`

**Note**: TD cohort NPZ files (_oct25) are located in `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/` (exception: these live in the original repo, not `_test`).

---

## 📊 Key Scripts

| Purpose | Script | Output |
|---------|--------|--------|
| **Network Analysis (IG Attribution)** | `network_analysis_yeo.py` | JSON, CSV, radar plots |
| **Region Tables** | `create_region_tables.py` | CSV tables (full + diverse subsets) |
| **Statistical Tests** | `run_statistical_comparisons.py` | 6 metrics, 12 comparisons |
| **Brain-Behavior (Standard)** | `run_*_brain_behavior_enhanced.py` | Fast analysis, good correlations |
| **Brain-Behavior (Optimized - Universal)** | `run_all_cohorts_*_optimized.py` | 6 strategies, FDR correction |
| **Brain-Behavior (Optimized - Dedicated)** | `run_stanford/abide/nki_*_optimized.py` | Better data handling, cohort-specific |
| **Brain-Behavior (Network-Level)** ⭐ | `run_network_brain_behavior_analysis.py` | Network predictors (7-17 networks) |
| **Network IG ↔ Targets** | `scripts/compute_network_age_correlations.py` | CSV summaries of network-level IG vs. chronological/predicted age or behavior |
| **Optimization Validation** | `check_optimization_predictions.py` | Integrity verification |
| **Optimization Summary** | `create_optimization_summary_figure.py` | Bar plots, tables (FDR corrected) |
| **Brain Age** | `plot_brain_age_*.py` | Combined scatter plots |
| **Network Radar (Counts + IG Effect)** | `scripts/plot_combined_network_radar.py` | TD/ADHD/ASD radar panels (counts + mean IG) |

**Note**: CMI-HBN TD currently uses enhanced script (optimized version to be created). CMI-HBN ADHD now has optimized version! ✅

---

### Network IG ↔ Target Correlations

Run the TD cohorts (chronological + predicted brain age) with the preset—no manual paths required:
```bash
python compute_network_age_correlations.py \
  --preset brain_age_td \
  --apply-fdr
```
The preset (see `config/network_correlation_presets.yaml`) points to the IG directories, fold `.bin` ages, and the `Predicted_Brain_Age:brain_age_pred` target key for each cohort. Outputs land in `results/network_correlations/`.

For brain-behavior cohorts, use the behavior preset:
```bash
python compute_network_age_correlations.py \
  --preset brain_behavior_adhd200 \
  --apply-fdr
```
That preset pulls observed/predicted scores directly from each cohort’s `predictions.csv` and aligns them to the IG rows. Update the YAML if you add new behaviors or datasets.

Once the CSV summaries are generated, create radar plots that show Spearman ρ (effect size per network):
```bash
python scripts/plot_combined_network_radar.py \
  --td /oak/.../shared_TD/shared_network_analysis.csv \
  --adhd /oak/.../shared_ADHD/shared_network_analysis.csv \
  --asd /oak/.../shared_ASD/shared_network_analysis.csv \
  --output /oak/.../shared_network_radar \
  --td-ig "HCP-Development=/oak/.../hcp_dev_Chronological_Age_network_correlations.csv" \
  --td-ig "NKI-RS TD=/oak/.../nki_rs_td_Chronological_Age_network_correlations.csv" \
  --td-ig "CMI-HBN TD=/oak/.../cmihbn_td_Chronological_Age_network_correlations.csv" \
  --td-ig "ADHD-200 TD=/oak/.../adhd200_td_Chronological_Age_network_correlations.csv" \
  --ig-column Spearman_rho
```
(Optional: add `--no-ig-abs` to keep the sign of ρ, or swap `Spearman_rho` for `Pearson_r`). The first row still shows count-based overlap; the TD 2×2 and ADHD/ASD 1×2 grids now encode correlation strength instead of mean IG.

---

### Network IG ↔ Behavior Correlations

To correlate network-aggregated IG scores with behavioral targets (observed or predicted), point the script at the brain-behavior IG NPZ bundles and supply the desired array keys:

```bash
cd /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/scripts

python compute_network_age_correlations.py \
  --datasets adhd200_adhd_optimized \
  --root-dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior \
  --parcellation yeo17 \
  --aggregation-method pos_share \
  --skip-chronological \
  --target-key Hyperactivity_Observed:y_true_hyperactivity \
  --target-key Hyperactivity_Predicted:y_pred_hyperactivity \
  --target-source Hyperactivity_Observed=adhd200_adhd_optimized::/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/adhd200_adhd_optimized/predictions.csv::subject_id::Hyperactivity_Observed \
  --target-source Hyperactivity_Predicted=adhd200_adhd_optimized::/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/adhd200_adhd_optimized/predictions.csv::subject_id::Hyperactivity_Predicted \
  --apply-fdr \
  --save-subject-level \
  --output-dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_correlations_behavior
```

- `--target-key LABEL:NPZ_KEY` can be repeated for each behavioral endpoint (e.g., observed vs predicted). Adjust paths per cohort; ADHD example above pulls from the optimized IG directory.
- Use `--skip-chronological` if the NPZs do not include age arrays.
- `--save-subject-level` exports subject-aligned network matrices with all requested targets.
- Outputs mirror the age workflow (`dataset_target_network_correlations.csv` plus an aggregated summary).

---

## 📖 Key Features

### Brain-Behavior Analysis
Two modes available:

#### 1. **Standard Mode** (Enhanced Scripts)
- Fixed 80% variance threshold, LinearRegression
- Fast (~2-5 min per script)
- Good for exploratory analysis
- Usage: `python run_*_brain_behavior_enhanced.py`

#### 2. **🚀 OPTIMIZED Mode** (Comprehensive Strategy Testing)
Maximizes Spearman ρ through exhaustive testing of 6 distinct strategies:

**Optimization Strategies:**
1. **PCA + Regression**: Dimensionality reduction (5-50 components) + 4 models
2. **PLS Regression**: Adaptive component limits (safe N/5 for small N, prevents numerical instability)
3. **Feature Selection + Regression**: SelectKBest + models
4. **Direct Regularized**: Ridge/Lasso/ElasticNet on all 246 ROIs
5. **TopK-IG** ⭐: Select 5-10 most important ROIs by IG magnitude (best for N<100)
   - Adaptive K: N/15, N/10, N/8 for small samples
   - Interpretable: "These 8 ROIs predict both age AND behavior"
6. **Network Aggregation** ⭐⭐: 246 ROIs → 7-17 Yeo networks (excellent for N<100)
   - Multiple methods: mean, abs_mean, pos_share, neg_share, signed_share
   - Ratio improvement: 0.3:1 → 12:1 for N=84
   - Highly interpretable: Network-level insights

**Key Features:**
- **Adaptive to sample size**: Strategies auto-adjust to N (strict limits for small N)
- **Numerical stability**: Catches model collapse, numerical explosions (±10^15)
- **FDR correction**: Benjamini-Hochberg across all measures (controls multiple comparisons)
- **Reproducibility**: Fixed random seed (42), deterministic results
- **Integrity checks**: Automatic validation, flags unreliable results
- **Sample size reporting**: N shown in all tables

**Performance:**
- **~200-400 configurations** tested per measure
- **+10-30% higher correlations** (when signal exists)
- **Small N handling**: TopK-IG and Network strategies prevent overfitting
- **Runtime**: 30-90 min per cohort

**Documentation:**
- `scripts/TOP_K_IG_STRATEGY.md` - Feature selection for small N
- `scripts/NETWORK_AGGREGATION_STRATEGY.md` - Network-level analysis
- `scripts/OPTIMIZATION_TOOLS_README.md` - Validation tools

#### Common Features (Both Modes)
- **Data integrity checks**: ID alignment verification, NaN detection, duplicate checks
- **Centralized styling**: plot_styles.py ensures 100% consistency
- **Triple export**: PNG + TIFF + AI (publication-ready)

#### Optimized Mode Output
- **Scatter plots**: Filename includes method (e.g., `scatter_measure_PLS_comp15_optimized.png`)
- **Predictions CSV**: Actual vs predicted values for all subjects
- **Summary CSV**: Best configuration per measure (CV_Spearman, Final_Spearman, p-value, R²)
- **Integrity checks**: Automatic verification printed to console (✅ = good, ❌ = don't use)
- See `OPTIMIZATION_GUIDE.md` for complete details ⭐

#### Syncing Between Local and Oak
This repo clones from local to Oak. After making changes, sync:
```bash
bash SYNC_NOW.sh  # Or use git/rsync
```

**Note**: Many `SpearmanRConstantInputWarning`