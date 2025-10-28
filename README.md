# Brain Age Prediction Analysis Pipeline

Comprehensive pipeline for analyzing brain age prediction using spatiotemporal deep neural networks (stDNN) with Integrated Gradients feature attribution across TD, ADHD, and ASD cohorts.

---
<img width="668" height="309" alt="image" src="https://github.com/user-attachments/assets/f5b32844-55aa-49b6-9cd6-44d7a2313484" />

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
python run_network_analysis.py
python create_region_tables.py
python create_polar_network_plots.py

# 2. Statistical Comparisons  
python run_statistical_comparisons.py

# 3. Brain-Behavior Analysis (choose A or B)

# Option A: Standard
python run_nki_brain_behavior_enhanced.py
python run_adhd200_brain_behavior_enhanced.py
python run_cmihbn_brain_behavior_enhanced.py

# Option B: Optimized (RECOMMENDED - higher correlations)
python run_optimized_brain_behavior.py --all

# 4. Combined Plots
python plot_brain_behavior_td_cohorts.py
python plot_pc_loadings_heatmap.py --dataset nki_rs_td
python plot_pc_loadings_heatmap.py --dataset adhd200_td
python plot_pc_loadings_heatmap.py --dataset cmihbn_td

# 5. Brain Age Plots
python plot_brain_age_td_cohorts.py
python plot_brain_age_adhd_cohorts.py
python plot_brain_age_asd_cohorts.py
```

---

## 📁 Output Locations

```
results/
├── network_analysis_yeo/        # Network statistics & radar plots
├── region_tables/               # ROI tables (diverse subsets for manuscripts)
├── statistical_comparisons/     # Cosine similarity, Spearman ρ, Aitchison, JS divergence
├── brain_behavior/              # Individual scatter plots (PNG/TIFF/AI)
├── brain_behavior_optimized/    # Optimized models (RECOMMENDED)
├── brain_age_plots/             # Combined scatter plots (PNG/TIFF/AI)
```

**All paths**: `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/`

---

## 📊 Key Scripts

| Purpose | Script | Output |
|---------|--------|--------|
| **Network Analysis** | `run_network_analysis.py` | JSON, CSV, radar plots |
| **Region Tables** | `create_region_tables.py` | CSV tables (full + diverse subsets) |
| **Statistical Tests** | `run_statistical_comparisons.py` | 6 metrics, 12 comparisons |
| **Brain-Behavior** | `run_optimized_brain_behavior.py --all` | Scatter plots, optimized models |
| **Brain Age** | `plot_brain_age_*.py` | Combined scatter plots |

---

## 🎨 Plot Styling

**All plots**: Arial font, #5A6FA8 dots (size=100), #D32F2F line (width=3.0pt), NO top/right spines, 300 DPI

**Export formats**: PNG + TIFF + AI (publication-ready, no post-processing needed)

**Dataset titles**:
- `adhd200_td` → "ADHD-200 TD Subset (NYU)"
- `cmihbn_td` → "CMI-HBN TD Subset"

**Module**: `scripts/plot_styles.py` (SINGLE SOURCE OF TRUTH)

---

## 📖 Key Features

### Optimized Brain-Behavior Analysis (RECOMMENDED)
- **Hyperparameter tuning**: Grid search over # of PCs, model type (Linear/Ridge/Lasso/ElasticNet), regularization
- **Cross-validated**: 5-fold CV maximizing Spearman ρ
- **Data integrity checks**: ID alignment verification, NaN detection, duplicate checks
- **76% less code**: Single config-driven script for all datasets
- **Higher correlations**: Regularization prevents overfitting

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

