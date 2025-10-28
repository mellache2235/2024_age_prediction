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

# Optional: Enable optimization for higher correlations
# Edit script, set: OPTIMIZE = True (line ~46)
# Tests Ridge/Lasso/ElasticNet, grid search over PCs
# Runtime: ~10-30 min per script (vs ~2-5 min standard)

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
│   ├── stanford_asd/           # NEW: SRS Total Score, Social Awareness
│   ├── abide_asd/              # NEW: ADOS (total, social, comm)
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
| **Brain-Behavior** | `run_optimized_brain_behavior.py --all` | Scatter plots, optimized models |
| **Brain Age** | `plot_brain_age_*.py` | Combined scatter plots |

---

## 📖 Key Features

### Brain-Behavior Analysis
- **Standard mode** (default): Fixed 80% variance threshold, LinearRegression, fast (~2-5 min)
- **Optimization mode** (optional): Set `OPTIMIZE = True` in script to enable:
  - Grid search over # of PCs (5, 10, 15, ..., n_samples-10)
  - 4 models tested: Linear, Ridge, Lasso, ElasticNet
  - Regularization strengths: 0.001, 0.01, 0.1, 1.0, 10.0, 100.0
  - 5-fold CV maximizing Spearman ρ
  - Higher correlations but slower (~10-30 min per script)
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

