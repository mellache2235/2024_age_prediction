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
  --datasets nki_rs_td cmihbn_td adhd200_td \
  --root-dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/figures \
  --parcellation yeo7 \
  --target-key Predicted_Brain_Age:brain_age_pred \
  --apply-fdr \
  --output-dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_correlations

python compute_network_age_correlations.py \
  --datasets adhd200_adhd_optimized \
  --root-dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior \
  --parcellation yeo17 \
  --aggregation-method pos_share \
  --skip-chronological \
  --target-key Hyperactivity_Observed:y_true_hyperactivity \
  --target-key Hyperactivity_Predicted:y_pred_hyperactivity \
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

Aggregate Integrated Gradients across all 500 folds, collapse ROIs to Yeo networks, and correlate network importance with chronological age (default), predicted brain age, or any behavioral targets stored alongside the IG files:

```bash
python scripts/compute_network_age_correlations.py \
  --datasets nki_rs_td cmihbn_td adhd200_td \
  --root-dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/figures \
  --parcellation yeo7 \
  --target-key Predicted_Brain_Age:brain_age_pred \
  --apply-fdr \
  --output-dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_correlations
```

- Uses `ig_files_td/` when present; falls back to `ig_files/` otherwise.
- Correlates with chronological age automatically; add `--skip-chronological` for behavior-only runs.
- Repeat `--target-key LABEL:NPZ_KEY` to include predicted outputs or behaviors (e.g., `--target-key Hyperactivity:y_true --target-key Predicted_Hyperactivity:y_pred`).
- Accepts `--parcellation yeo7|yeo17` and aggregation modes (`--aggregation-method mean|abs_mean|pos_share|neg_share|signed_share`).
- Attempts to infer subject IDs (and ages when required) from each `_ig.npz`; override with `--subject-key` / `--age-key` if the heuristic guesses incorrectly.
- Pass `--apply-fdr` to append Benjamini-Hochberg corrected p-values per dataset/target.
- Pass `--save-subject-level` to dump per-subject network IG matrices with all requested targets merged in.

Outputs land in the directory specified by `--output-dir`:

```
network_correlations/
├── nki_rs_td_Chronological_Age_network_correlations.csv
├── nki_rs_td_Predicted_Brain_Age_network_correlations.csv
├── cmihbn_td_Chronological_Age_network_correlations.csv
├── ...
└── network_correlations_yeo7_mean.csv   # Combined summary across datasets/targets
```

Each CSV reports `Pearson_r`, `Spearman_rho`, descriptive stats (`Mean_IG`, `Std_IG`), and (optionally) FDR-adjusted p-values per network.

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

**Note**: Many `SpearmanRConstantInputWarning` messages during optimization are **normal** - they're from bad configurations being tested and rejected. See `OPTIMIZATION_GUIDE.md` for details.

### Statistical Comparisons
6 complementary metrics per comparison:
1. Cosine similarity (permutation p-value, 10k iterations)
2. Spearman ρ on ROI ranks
3. Aitchison (CLR) distance
4. Jensen-Shannon divergence
5. ROI-wise two-proportion tests (FDR corrected)
6. Network-level aggregation

---

## 🐛 Troubleshooting

**Environment issues**: See `INSTALL.md`

**Missing packages**: Run `python scripts/verify_imports.py`

**Font not found**: Arial loaded automatically from HPC path

---

## 🧠 Network Aggregation Methods

For small sample sizes (N<100), network-level features provide better statistical power and interpretability.

### Aggregation Methods (All Tested in Optimization):

**Simple Methods:**
- `mean`: Average IG scores within network
- `abs_mean`: Average absolute IG scores (preserves magnitude)

**Signed Mass Methods** (Recommended - from research):
For each network *g* with ROIs, compute:
- \( P_g = \sum_{i \in g} \max(IG_i, 0) \) (positive mass)
- \( N_g = \sum_{i \in g} \max(-IG_i, 0) \) (negative mass)  
- \( A = \sum_j |IG_j| \) (total absolute mass across ALL 246 ROIs)

Then create features:
- `pos_share`: \( P_g / A \) (positive mass fraction)
- `neg_share`: \( N_g / A \) (negative mass fraction)
- `signed_share`: \( (P_g - N_g) / A \) (net mass fraction)

**Benefits:**
- Normalizes by total IG magnitude (comparable across subjects)
- Separates positive/negative contributions
- Values in [0,1] for shares (interpretable as proportions)

### Performance by Sample Size:

| N | Best Approach | Expected ρ | Ratio |
|---|---------------|------------|-------|
| 81 (NKI) | Network (7) + Ridge | 0.30-0.40 | 12:1 ✅ |
| 84 (CMI-HBN) | Network (7) + Linear | 0.30-0.40 | 12:1 ✅ |
| 169 (ABIDE) | Network or TopK-IG | 0.35-0.45 | 24:1 ✓ |
| 238 (ADHD200) | PCA/PLS | 0.40-0.50 | 3:1 ✓ |

---

## 📧 Contact

Menon Lab, Stanford University

**Last Updated**: 2024

### Network Radar Panels (Counts + Mean IG)

Create three matching radar panels for TD, ADHD, and ASD cohorts. The default inputs are the shared-network count summaries, and the optional effect-size radar uses the mean network IG magnitudes (averaged across folds) emitted by `compute_network_age_correlations.py`.

```bash
python scripts/plot_combined_network_radar.py \
  --td /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_analysis_yeo/shared_TD/shared_network_analysis.csv \
  --adhd /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_analysis_yeo/shared_ADHD/shared_network_analysis.csv \
  --asd /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_analysis_yeo/shared_ASD/shared_network_analysis.csv \
  --output /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_analysis_yeo/radar_panels/shared_network_radar \
  --td-ig "HCP-Development=/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/network_correlations/hcp_development_Chronological_Age_network_correlations.csv" \
  --td-ig "NKI-RS TD=/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_correlations/nki_rs_td_Chronological_Age_network_correlations.csv" \
  --td-ig "CMI-HBN TD=/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_correlations/cmihbn_td_Chronological_Age_network_correlations.csv" \
  --td-ig "ADHD-200 TD=/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_correlations/adhd200_td_Chronological_Age_network_correlations.csv" \
  --adhd-ig "ADHD-200 ADHD=/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_correlations_behavior/adhd200_adhd_Hyperactivity_Observed_network_correlations.csv" \
  --adhd-ig "CMI-HBN ADHD=/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_correlations_behavior/cmihbn_adhd_Hyperactivity_Observed_network_correlations.csv" \
  --asd-ig "Stanford ASD=/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_correlations_behavior/stanford_asd_SRS_total_network_correlations.csv" \
  --asd-ig "ABIDE ASD=/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_correlations_behavior/abide_asd_ADOS_total_network_correlations.csv" \
  --ig-target Chronological_Age \
  --ig-column Mean_IG
```

- Saves the count-based overlap radar (`shared_network_radar.{png,tiff,ai}`) across TD/ADHD/ASD.
- Generates three mean-IG figures: a TD 2×2 grid (order required: HCP-Development, NKI-RS TD, CMI-HBN TD, ADHD-200 TD) plus 1×2 grids for ADHD and ASD cohorts.
- HCP-Development IG summaries currently live in the non-`_test` repo; all other paths point to `_test` results.
- Use `--ig-column` to switch to other metrics (e.g., `Pearson_r`), and `--ig-aggregation` (mean/sum/median) if multiple rows per network remain after filtering.
- Pass `--no-ig-abs` if you need signed IG values; negative values are shifted so the minimum sits at zero before normalization.
- Adjust `--ig-radius-label` to customize the legend beneath the effect-size panels.

