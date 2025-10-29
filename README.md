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

# 🚀 NEW: Optimized Brain-Behavior Analysis (Maximize Spearman ρ)
# Comprehensive optimization: PCA, PLS, Feature Selection, Regularization
# Tests ~100-200 configurations per behavioral measure

# Universal script for most cohorts (recommended)
python run_all_cohorts_brain_behavior_optimized.py --cohort abide_asd
python run_all_cohorts_brain_behavior_optimized.py --cohort adhd200_td
python run_all_cohorts_brain_behavior_optimized.py --cohort adhd200_adhd
python run_all_cohorts_brain_behavior_optimized.py --cohort cmihbn_td
python run_all_cohorts_brain_behavior_optimized.py --cohort cmihbn_adhd
python run_all_cohorts_brain_behavior_optimized.py --all  # Run all cohorts

# Cohort-specific scripts (special data requirements)
python run_stanford_asd_brain_behavior_optimized.py  # Stanford ASD (SRS)
python run_nki_brain_behavior_optimized.py           # NKI (CAARS/Conners)

# Runtime: ~30-60 min per cohort (vs ~2-5 min standard), +10-30% higher correlations
# See: scripts/UNIVERSAL_OPTIMIZATION_GUIDE.md for details

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
│   ├── stanford_asd_optimized/ # 🚀 Optimized (max Spearman ρ)
│   ├── abide_asd_optimized/    # 🚀 Optimized
│   ├── adhd200_td_optimized/   # 🚀 Optimized
│   ├── adhd200_adhd_optimized/ # 🚀 Optimized
│   ├── cmihbn_td_optimized/    # 🚀 Optimized
│   ├── cmihbn_adhd_optimized/  # 🚀 Optimized
│   ├── nki_rs_td_optimized/    # 🚀 Optimized
│   └── combined_plots/
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
| **Network Analysis** | `network_analysis_yeo.py` | JSON, CSV, radar plots |
| **Region Tables** | `create_region_tables.py` | CSV tables (full + diverse subsets) |
| **Statistical Tests** | `run_statistical_comparisons.py` | 6 metrics, 12 comparisons |
| **Brain-Behavior (Standard)** | `run_*_brain_behavior_enhanced.py` | Fast analysis, good correlations |
| **Brain-Behavior (Optimized)** | `run_*_brain_behavior_optimized.py` | 🚀 Max Spearman ρ (+10-30%) |
| **Brain Age** | `plot_brain_age_*.py` | Combined scatter plots |

---

## 📖 Key Features

### Brain-Behavior Analysis
Two modes available:

#### 1. **Standard Mode** (Enhanced Scripts)
- Fixed 80% variance threshold, LinearRegression
- Fast (~2-5 min per script)
- Good for exploratory analysis
- Usage: `python run_*_brain_behavior_enhanced.py`

#### 2. **🚀 NEW: Optimized Mode** (Maximizes Spearman ρ)
- **4 Optimization Strategies**:
  1. PCA + Regression (Linear, Ridge, Lasso, ElasticNet)
  2. PLS Regression (optimized for covariance)
  3. Feature Selection + Regression (F-stat, Mutual Info)
  4. Direct Regularized Regression
- **Comprehensive hyperparameter search**:
  - PC components: 5-50 (auto-adjusted)
  - PLS components: 3-30
  - Alpha values: [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
  - Feature selection: Top-K = [50, 100, 150, 200]
  - **~100-200 configurations tested per behavioral measure**
- **5-fold cross-validation** maximizing Spearman ρ
- **Expected improvement**: +10-30% higher correlations
- **Runtime**: ~30-60 min per cohort
- **Universal script**: `python run_all_cohorts_brain_behavior_optimized.py --cohort {cohort}`
- **Quick start**: See `scripts/UNIVERSAL_OPTIMIZATION_GUIDE.md`

#### Common Features (Both Modes)
- **Data integrity checks**: ID alignment verification, NaN detection, duplicate checks
- **Centralized styling**: plot_styles.py ensures 100% consistency
- **Triple export**: PNG + TIFF + AI (publication-ready)

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

## 📧 Contact

Menon Lab, Stanford University

**Last Updated**: 2024

