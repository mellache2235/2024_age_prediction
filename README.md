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

**IG-Feature Based** (pre-computed IG CSVs, PCA + LinearRegression):
```bash
# TD cohorts
python scripts/nki_brain_behavior_ig_analysis.py                          # NKI-RS TD (CAARS, FDR corrected)
python run_all_cohorts_brain_behavior_optimized.py --cohort adhd200_td    # ADHD-200 TD
python run_all_cohorts_brain_behavior_optimized.py --cohort cmihbn_td     # CMI-HBN TD

# ADHD cohorts (use --measure HY or --measure IN)
python scripts/adhd200_brain_behavior_ig_analysis.py --measure HY         # ADHD-200 ADHD (NYU, Hyperactivity)
python scripts/adhd200_brain_behavior_ig_analysis.py --measure IN         # ADHD-200 ADHD (NYU, Inattention)
python scripts/cmihbn_brain_behavior_ig_analysis.py                       # CMI-HBN ADHD (C3SR)

# ASD cohorts
python scripts/stanford_brain_behavior_ig_analysis.py                     # Stanford ASD (SRS)
python scripts/abide_brain_behavior_ig_analysis.py                        # ABIDE ASD (ADOS)
```
These scripts perform **two analyses** per cohort:
1. **IG → Behavior**: PCA (50 components) + LinearRegression predicts behavioral scores from IG features
2. **BAG → Behavior**: Correlates Brain Age Gap (predicted - observed age) with behavioral scores (FDR correction applied for NKI)

Ages loaded from oct25 NPZ files (aligned by fold_0.bin row order), behavioral scores merged by subject_id. Generates PNG/AI scatter plots for significant correlations.

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

# 4. Network Importance (Dominance Analysis)
- Run `compute_network_importance_regression.py --preset <name>` to quantify each network's contribution via dominance analysis:
  ```bash
  python compute_network_importance_regression.py --preset brain_age_td
  python compute_network_importance_regression.py --preset brain_age_adhd
  python compute_network_importance_regression.py --preset brain_age_asd
  ```
  Add `--save-subject-level` to export per-subject network IG matrices with ages for external analysis. Each run produces:
  - `<dataset>_network_importance.csv` (full metrics)
  - `dominance_multivariate_network_age_<dataset>.csv` (radar-ready: Network, Dominance %)
  - `dominance_permutation_pvalues.csv` (model significance summary)
  - `dominance_multivariate_network_age_pooled.csv` (dominance averaged across all cohorts in the preset, for 1×3 overlap radar)
  
  Use the per-dataset radar CSVs with `plot_combined_network_radar.py --*-ig` to build TD 2×2, ADHD 1×2, and ASD 1×2 grids. Use the pooled CSV for a single-panel radar showing average network importance across cohorts.
  Presets in `config/network_importance_presets.yaml` encode IG directories, age sources, parcellation (Yeo-17), and aggregation method (abs_mean). By default, importance is computed via **dominance analysis** (manual implementation using itertools.combinations), which decomposes R² into each network's total contribution across all possible subset models. Includes permutation testing (5000 shuffles) to assess overall model significance. Faster alternatives (coefficients, permutation, LOO) are available by editing `importance_method` in the preset or using `--importance-method`.
- After generating importance tables, use `plot_combined_network_radar.py` to visualize:
  - **Count-based overlap (1×3)**: Provide shared TD/ADHD/ASD count CSVs to generate a single row showing cross-cohort consensus.
    ```bash
    python scripts/plot_combined_network_radar.py \
      --td /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_analysis_yeo/shared_TD/shared_network_analysis.csv \
      --adhd /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_analysis_yeo/shared_ADHD/shared_network_analysis.csv \
      --asd /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_analysis_yeo/shared_ASD/shared_network_analysis.csv \
      --output /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_analysis_yeo/radar_panels/shared_network_radar
    ```
  - **Effect-size grids**: Add `--*-ig` arguments with regression importance summaries to render cohort-specific panels. The radar script automatically uses `Effect_Size_Pct` (percentage drops) for clearer visualization; override with `--ig-column Effect_Size` for raw Δρ values.
- `plot_combined_network_radar.py` labels bars with the Yeo network names (e.g., DefaultA, DorsAttnA) after stripping the `Network_` prefix.
- Minimal example for an ASD-only panel:
  ```bash
  python scripts/plot_combined_network_radar.py \
    --asd /oak/.../shared_ASD/shared_network_analysis.csv \
    --output /oak/.../results/network_analysis_yeo/radar_panels/abide_autism_radar \
    --asd-ig "ABIDE Autism=/oak/.../network_correlations/abide_asd_Chronological_Age_network_correlations.csv" \
    --ig-column Spearman_rho
  ```
- Full TD 2×2 grid (HCP-Development, NKI-RS TD, CMI-HBN TD, ADHD-200 TD):
  ```bash
  python scripts/plot_combined_network_radar.py \
    --td /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_analysis_yeo/shared_TD/shared_network_analysis.csv \
    --output /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_analysis_yeo/radar_panels/td_cohorts_radar \
    --td-ig "HCP-Development=/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_correlations/hcp_dev_Chronological_Age_network_correlations.csv" \
    --td-ig "NKI-RS TD=/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_correlations/nki_rs_td_Chronological_Age_network_correlations.csv" \
    --td-ig "CMI-HBN TD=/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_correlations/cmihbn_td_Chronological_Age_network_correlations.csv" \
    --td-ig "ADHD-200 TD=/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_correlations/adhd200_td_Chronological_Age_network_correlations.csv" \
    --ig-column Spearman_rho
  ```

# 5. Combined Plots
```bash
python plot_brain_behavior_td_cohorts.py
python plot_pc_loadings_heatmap.py --dataset nki_rs_td
python plot_pc_loadings_heatmap.py --dataset adhd200_td
python plot_pc_loadings_heatmap.py --dataset cmihbn_td
```

# 6. Brain Age Plots
```bash
# Note: TD cohorts (NKI, CMI-HBN TD, ADHD200 TD) use _oct25 NPZ files from the original repo:
#   /oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/

python plot_brain_age_td_cohorts.py \
  --output_dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_age_plots
python plot_brain_age_adhd_cohorts.py \
  --output_dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_age_plots
python plot_brain_age_asd_cohorts.py \
  --output_dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_age_plots
```

---

## 🎓 Model Training

To train the brain age prediction ConvNet model:
```bash
python train_Convnet_hyperopt_wandb_v4.py
```
This script trains the 1D-CNN architecture on HCP-Development data using:
- 5-fold cross-validation
- Wandb hyperparameter optimization
- Early stopping with validation monitoring
- Bias correction via linear regression

Pre-trained models are available in `scripts/train_regression_models/dev/` for immediate use in IG computation and brain-behavior analysis.

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
| **Network Importance (Dominance)** | `compute_network_importance_regression.py` | Dominance analysis CSVs + radar-ready outputs |
| **Optimization Validation** | `check_optimization_predictions.py` | Integrity verification |
| **Optimization Summary** | `create_optimization_summary_figure.py` | Bar plots, tables (FDR corrected) |
| **Brain Age** | `plot_brain_age_*.py` | Combined scatter plots |
| **Network Radar (Counts + IG Effect)** | `scripts/plot_combined_network_radar.py` | TD/ADHD/ASD radar panels (counts + mean IG) |

**Note**: CMI-HBN TD currently uses enhanced script (optimized version to be created). CMI-HBN ADHD now has optimized version! ✅

---

### Network Importance via Dominance Analysis

Quantifies each Yeo-17 network's contribution to age prediction using dominance analysis (netneurotools).

**Workflow:**
1. Aggregate 500-fold IG attributions → 20 network features per subject (excluding Yeo17_0)
2. Z-score networks and ages
3. Compute total dominance for each network (decomposition of model R²)
4. Run 5000 permutations to test overall model significance

**Commands:**

*Run all cohorts in a group:*
```bash
python compute_network_importance_regression.py --preset brain_age_td     # 4 TD cohorts
python compute_network_importance_regression.py --preset brain_age_adhd   # 2 ADHD cohorts
python compute_network_importance_regression.py --preset brain_age_asd    # 2 ASD cohorts
```

*Run individual cohorts* (faster, for parallel processing):
```bash
# Set output directory
OUTDIR=/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_importance

# TD cohorts
python compute_network_importance_regression.py --preset hcp_dev_only --importance-method dominance --output-dir $OUTDIR
python compute_network_importance_regression.py --preset nki_rs_td_only --importance-method dominance --output-dir $OUTDIR
python compute_network_importance_regression.py --preset cmihbn_td_only --importance-method dominance --output-dir $OUTDIR
python compute_network_importance_regression.py --preset adhd200_td_only --importance-method dominance --output-dir $OUTDIR

# ADHD cohorts  
python compute_network_importance_regression.py --preset adhd200_adhd_only --importance-method dominance --output-dir $OUTDIR
python compute_network_importance_regression.py --preset cmihbn_adhd_only --importance-method dominance --output-dir $OUTDIR

# ASD cohorts
python compute_network_importance_regression.py --preset abide_asd_only --importance-method dominance --output-dir $OUTDIR
python compute_network_importance_regression.py --preset stanford_asd_only --importance-method dominance --output-dir $OUTDIR
```

*Generate radar plots after analysis completes:*

**Individual cohort radars** (single panel per dataset):
```bash
OUTDIR=/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_importance

# Create output directory
mkdir -p $OUTDIR/radar_plots

# TD cohorts (individual)
python scripts/plot_combined_network_radar.py \
  --td $OUTDIR/dominance_multivariate_network_age_hcp_dev.csv \
  --output $OUTDIR/radar_plots/hcp_dev_dominance --transform sqrt

python scripts/plot_combined_network_radar.py \
  --td $OUTDIR/dominance_multivariate_network_age_nki_rs_td.csv \
  --output $OUTDIR/radar_plots/nki_rs_td_dominance --transform sqrt --show-values

python scripts/plot_combined_network_radar.py \
  --td $OUTDIR/dominance_multivariate_network_age_cmihbn_td.csv \
  --output $OUTDIR/radar_plots/cmihbn_td_dominance --transform sqrt

python scripts/plot_combined_network_radar.py \
  --td $OUTDIR/dominance_multivariate_network_age_adhd200_td.csv \
  --output $OUTDIR/radar_plots/adhd200_td_dominance --transform sqrt

# ADHD cohorts (individual)
python scripts/plot_combined_network_radar.py \
  --adhd $OUTDIR/dominance_multivariate_network_age_adhd200_adhd.csv \
  --output $OUTDIR/radar_plots/adhd200_adhd_dominance --transform sqrt

python scripts/plot_combined_network_radar.py \
  --adhd $OUTDIR/dominance_multivariate_network_age_cmihbn_adhd.csv \
  --output $OUTDIR/radar_plots/cmihbn_adhd_dominance --transform sqrt

# ASD cohorts (individual)
python scripts/plot_combined_network_radar.py \
  --asd $OUTDIR/dominance_multivariate_network_age_abide_asd.csv \
  --output $OUTDIR/radar_plots/abide_asd_dominance --transform sqrt

python scripts/plot_combined_network_radar.py \
  --asd $OUTDIR/dominance_multivariate_network_age_stanford_asd.csv \
  --output $OUTDIR/radar_plots/stanford_asd_dominance --transform sqrt
```

**Multi-panel grids** (TD 2×2, ADHD 1×2, ASD 1×2):
```bash
# TD 2×2 grid
python scripts/plot_combined_network_radar.py \
  --td-ig "HCP-Development=$OUTDIR/dominance_multivariate_network_age_hcp_dev.csv" \
  --td-ig "NKI-RS TD=$OUTDIR/dominance_multivariate_network_age_nki_rs_td.csv" \
  --td-ig "CMI-HBN TD=$OUTDIR/dominance_multivariate_network_age_cmihbn_td.csv" \
  --td-ig "ADHD-200 TD=$OUTDIR/dominance_multivariate_network_age_adhd200_td.csv" \
  --output $OUTDIR/radar_plots/td_dominance_2x2 --transform sqrt

# ADHD 1×2 grid
python scripts/plot_combined_network_radar.py \
  --adhd-ig "ADHD-200 ADHD=$OUTDIR/dominance_multivariate_network_age_adhd200_adhd.csv" \
  --adhd-ig "CMI-HBN ADHD=$OUTDIR/dominance_multivariate_network_age_cmihbn_adhd.csv" \
  --output $OUTDIR/radar_plots/adhd_dominance_1x2 --transform sqrt

# ASD 1×2 grid
python scripts/plot_combined_network_radar.py \
  --asd-ig "ABIDE ASD=$OUTDIR/dominance_multivariate_network_age_abide_asd.csv" \
  --asd-ig "Stanford ASD=$OUTDIR/dominance_multivariate_network_age_stanford_asd.csv" \
  --output $OUTDIR/radar_plots/asd_dominance_1x2 --transform sqrt
```

**Note:** `--transform sqrt` applies square-root transformation to compress the dynamic range, making networks with small dominance values (1-5%) more visible as bars.

**Outputs** (in `results/network_importance/`):
- `<dataset>_network_importance.csv` - Full metrics (Total_Dominance, Dominance_Pct, Model_R2_Adj, P_Value, N_Subjects)
- `dominance_multivariate_network_age_<dataset>.csv` - Radar-ready (Network, Dominance %)
- `dominance_permutation_pvalues.csv` - Model significance summary (one row per dataset: adjusted R², p-value, N)
- `dominance_multivariate_network_age_pooled.csv` - Average dominance across all cohorts in the preset

**Visualization:**
Use the radar-ready CSVs with `plot_combined_network_radar.py`:
- **Cohort-specific grids**: TD 2×2, ADHD 1×2, ASD 1×2 using per-dataset CSVs
- **Pooled overlap**: Single panel using `dominance_multivariate_network_age_pooled.csv`

**Requirements:**
- Presets in `config/network_importance_presets.yaml`
- No external dependencies beyond numpy/pandas/sklearn/scipy

**Notes:**
- Network "0" (Yeo17_0, unassigned ROIs) is excluded from analysis
- Dominance percentages sum to 100% within each dataset
- Permutation p-values test H₀: "network predictors have no relationship with age"
- For pooled omnibus analysis (all TD/ADHD/ASD subjects combined): export subject-level data with `--save-subject-level`, concatenate the CSVs externally, and run your own dominance code on the combined matrix for increased statistical power

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

**Note**: Many `