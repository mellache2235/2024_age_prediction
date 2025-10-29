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

# 4. Combined Plots
python plot_brain_behavior_td_cohorts.py
python plot_pc_loadings_heatmap.py --dataset nki_rs_td
python plot_pc_loadings_heatmap.py --dataset adhd200_td
python plot_pc_loadings_heatmap.py --dataset cmihbn_td

# 5. Brain Age Plots
# Note: TD cohorts (NKI, CMI-HBN TD, ADHD200 TD) use _oct25 NPZ files from:
#   /oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/

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

**Note**: TD cohort NPZ files (_oct25) are located in `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/`

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
| **Optimization Validation** | `check_optimization_predictions.py` | Integrity verification |
| **Optimization Summary** | `create_optimization_summary_figure.py` | Bar plots, tables (FDR corrected) |
| **Brain Age** | `plot_brain_age_*.py` | Combined scatter plots |

**Note**: CMI-HBN TD currently uses enhanced script (optimized version to be created). CMI-HBN ADHD now has optimized version! ✅

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

