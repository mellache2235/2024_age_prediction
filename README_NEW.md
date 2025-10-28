# Brain Age Prediction Analysis Pipeline

A comprehensive, production-quality pipeline for analyzing brain age prediction models across multiple datasets (TD, ADHD, ASD cohorts). Includes feature attribution, network analysis, brain-behavior correlations, statistical comparisons, and publication-ready visualizations.

---

## 📋 Table of Contents

1. [Environment Setup](#environment-setup)
2. [Pipeline Overview](#pipeline-overview)
3. [Analysis Workflows](#analysis-workflows)
   - [Network Analysis & Feature Attribution](#1-network-analysis--feature-attribution)
   - [Statistical Comparisons](#2-statistical-comparisons)
   - [Brain-Behavior Correlations](#3-brain-behavior-correlations)
   - [Brain Age Prediction Plots](#4-brain-age-prediction-plots)
4. [Output Files](#output-files)
5. [Plot Styling](#plot-styling)

---

## 🔧 Environment Setup

### Required Packages

```bash
# Core scientific computing
numpy>=1.21.0, pandas>=1.3.0, scipy>=1.7.0, scikit-learn>=1.0.0

# Visualization
matplotlib>=3.4.0, seaborn>=0.11.0

# Deep learning (for model inference)
torch>=1.10.0, pytorch-lightning>=1.5.0, captum>=0.5.0

# Utilities
openpyxl>=3.0.0, pyyaml>=5.4.0, statsmodels>=0.13.0
```

### Installation on HPC

```bash
# Option 1: Use shared environment
source /oak/stanford/groups/menon/software/python_envs/atif_env/bin/activate

# Option 2: Create your own
conda create -n brain_age python=3.9
conda activate brain_age
pip install -r requirements.txt
```

**Note**: Install PyTorch first with appropriate CUDA version:
```bash
# For GPU (CUDA 11.7)
pip install torch==1.13.1+cu117 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117

# For CPU only
pip install torch torchvision torchaudio
```

---

## 📊 Pipeline Overview

```
INPUT: IG feature CSVs → ANALYSIS → OUTPUT: Tables, Plots, Statistics
         ↓                   ↓              ↓
    [ROI scores]      [Network/ROI]    [PNG + AI files]
                      [Statistical]    [Summary CSVs]
                      [Behavioral]     [Network tables]
```

**Datasets Analyzed**:
- **TD Cohorts**: NKI-RS, ADHD200 TD, CMI-HBN TD
- **ADHD Cohorts**: ADHD200 ADHD, CMI-HBN ADHD
- **ASD Cohorts**: ABIDE ASD, Stanford ASD

---

## 🔬 Analysis Workflows

### 1. Network Analysis & Feature Attribution

**Purpose**: Identify which brain networks and ROIs are most important for age prediction across datasets.

#### Step 1a: Create Count Data CSVs
```bash
cd scripts
python create_count_data.py
```
**Input**: Excel files with consensus features  
**Output**: `results/count_data/*.csv` (ROI counts per dataset)  
**What it does**: Converts Excel consensus features to CSV format for analysis

#### Step 1b: Run Network Analysis
```bash
python run_network_analysis.py
```
**Input**: Count data CSVs  
**Output**: 
- `results/network_analysis_yeo/*/` directories
  - `*_network_analysis_results.json` (summary statistics)
  - `*_network_proportions.csv` (network-level aggregation)
  - `*_roi_counts.csv` (ROI-level counts)

**What it does**: Maps ROIs to Yeo 17 networks, computes network-level statistics

#### Step 1c: Create Region Tables
```bash
python create_region_tables.py
```
**Input**: Count data CSVs  
**Output**: `results/region_tables/`
- Individual dataset tables: `*_top_regions.csv`
- Shared region tables: `shared_*_regions.csv` (TD/ADHD/ASD)
- Diverse subset tables: `shared_*_regions_diverse_subset.csv` (for manuscripts, max 30 regions, max 3 per network)

**What it does**: Creates human-readable tables of important brain regions with network diversity

#### Step 1d: Create Radar Plots
```bash
python create_polar_network_plots.py
```
**Input**: Network analysis results (JSON)  
**Output**: `results/radar_plots/`
- `*_polar_network.png` (individual datasets)
- `*_polar_network.ai` (Adobe Illustrator format)

**Plots Generated** (7 total):
- Individual: NKI, ADHD200 TD, CMI-HBN TD, ADHD200 ADHD, CMI-HBN ADHD, ABIDE ASD, Stanford ASD
- Shows network-level importance as radar chart

**Styling**: Arial font, modern teal/cyan fill, dark teal outline, light gray background

---

### 2. Statistical Comparisons

**Purpose**: Rigorously compare feature importance patterns across datasets with multiple statistical metrics.

#### Step 2: Run Statistical Comparisons
```bash
python run_statistical_comparisons.py
```

**Input**: IG feature CSVs from `integrated_gradients/`  
**Output**: `results/statistical_comparisons/`
- `master_summary.csv` (all comparisons)
- `{dataset1}_vs_{dataset2}.csv` (pairwise global metrics)
- `{dataset1}_vs_{dataset2}_networks.csv` (network-level aggregation)

**Comparisons Run** (12 total):
- **Within-group**: TD cohorts (3), ADHD cohorts (1), ASD cohorts (1)
- **Between-group**: TD vs ADHD (2), TD vs ASD (2), ADHD vs ASD (2)

**Metrics Computed**:
1. **Cosine Similarity** (with permutation p-value, 10k permutations)
2. **Spearman ρ** (on ROI ranks)
3. **Aitchison Distance** (CLR/compositional)
4. **Jensen-Shannon Divergence** (distribution-aware)
5. **ROI-wise Two-Proportion Tests** (with FDR correction)
6. **Network-Level Aggregation** (by Yeo 17 networks)

**Plots Generated**: None (statistical tables only)

---

### 3. Brain-Behavior Correlations

**Purpose**: Optimize prediction of behavioral scores from brain features using PCA + regression.

#### Option A: Standard Analysis (Fixed 80% Variance Threshold)

Individual scripts for each dataset:

```bash
# TD cohorts
python run_nki_brain_behavior_enhanced.py
python run_adhd200_brain_behavior_enhanced.py
python run_cmihbn_brain_behavior_enhanced.py

# ADHD cohorts
python run_adhd200_adhd_brain_behavior_enhanced.py
python run_cmihbn_adhd_brain_behavior_enhanced.py
```

**Input**: 
- IG feature CSVs
- Behavioral data (CAARS, Conners, C3SR, RBS)

**Output**: `results/brain_behavior/{dataset}/`
- `elbow_plot.png` (scree plot)
- `scatter_{measure}.png` + `.ai` (predicted vs observed, one per measure)
- `linear_regression_results.csv` (summary: N, PCs, ρ, p, R²)
- `pc_importance_{measure}.csv` (PC rankings)
- `PC{N}_loadings.csv` (top 10 brain regions per PC)

**Plots Generated** (~20 total):
- **NKI-RS TD**: CAARS, Conners (parent + self), RBS measures
- **ADHD200 TD**: Hyperactivity, Inattention (NYU site only)
- **CMI-HBN TD**: C3SR measures
- **ADHD200 ADHD**: Hyperactivity, Inattention
- **CMI-HBN ADHD**: C3SR measures

#### Option B: Optimized Analysis (Hyperparameter Tuning)

**NEW** - Single unified script with automatic optimization:

```bash
# Single dataset
python run_optimized_brain_behavior.py --dataset nki

# All datasets
python run_optimized_brain_behavior.py --all

# Test mode (first 5 measures)
python run_optimized_brain_behavior.py --dataset nki --max-measures 5
```

**Input**: Same as Option A  
**Output**: `results/brain_behavior_optimized/{dataset}/`
- `summary.csv` (best model for each measure)
- `{measure}.png` + `.ai` (scatter plots with best model)
- `{measure}_optimization.csv` (full grid search results)

**What it optimizes**:
- **Number of PCs**: 5, 10, 15, ..., min(50, n_samples-10)
- **Model type**: Linear, Ridge, Lasso, ElasticNet
- **Regularization (α)**: 0.001, 0.01, 0.1, 1.0, 10.0, 100.0
- **Criterion**: Maximizes Spearman ρ via 5-fold CV

**Features**:
- ✅ Data integrity checks (ID alignment, NaN detection, duplicates)
- ✅ Dynamic PC range (prevents n_components > n_samples errors)
- ✅ Parallel cross-validation (uses all CPU cores)
- ✅ 76% less code than separate scripts
- ✅ Config-driven (easy to add new datasets)

#### Step 3b: Create Combined Plots

```bash
# TD cohorts (Hyperactivity + Inattention)
python plot_brain_behavior_td_cohorts.py

# Custom 1x3 subplot (NKI HY, NKI IN, ADHD200 HY)
python plot_brain_behavior_custom_1x3.py
```

**Input**: Individual scatter plots from Step 3  
**Output**: `results/brain_behavior/combined_plots/`
- `hyperactivity_td_cohorts.png` + `.ai` (3-panel: NKI, ADHD200, CMI-HBN)
- `inattention_td_cohorts.png` + `.ai` (3-panel: NKI, ADHD200, CMI-HBN)
- `custom_1x3_nki_adhd200_hyperactivity_inattention.png` + `.ai`

**Plots Generated** (3 combined figures)

#### Step 3c: PC Loadings Heatmaps

```bash
python plot_pc_loadings_heatmap.py --dataset nki_rs_td
python plot_pc_loadings_heatmap.py --dataset adhd200_td
python plot_pc_loadings_heatmap.py --dataset cmihbn_td
```

**Input**: PC loadings from brain-behavior analysis  
**Output**: `results/brain_behavior/{dataset}/`
- `pc_loadings_heatmap.png` (top 15 ROIs × first 3 PCs)
- `pc_loadings_top_regions.csv`

**Plots Generated**: 3 heatmaps (one per TD cohort)  
**Styling**: Red-blue colormap, shows which brain regions load on each PC

---

### 4. Brain Age Prediction Plots

**Purpose**: Visualize brain age prediction accuracy across cohorts.

#### Step 4: Create Combined Brain Age Plots

```bash
# TD cohorts
python plot_brain_age_td_cohorts.py

# ADHD cohorts
python plot_brain_age_adhd_cohorts.py

# ASD cohorts
python plot_brain_age_asd_cohorts.py
```

**Input**: NPZ files with predictions (from model inference, not included in this repo)  
**Output**: `results/brain_age_plots/`
- `td_cohorts_combined_scatter.png` + `.ai` (3-panel: NKI, ADHD200, CMI-HBN)
- `adhd_cohorts_combined_scatter.png` + `.ai` (2-panel: ADHD200, CMI-HBN)
- `asd_cohorts_combined_scatter.png` + `.ai` (2-panel: ABIDE, Stanford)

**Plots Generated** (3 combined figures)  
**Styling**: #5A6FA8 dots, #D32F2F regression line, Arial font, ρ/p/MAE in corner

---

## 📁 Output Files

```
results/
├── count_data/                          # ROI count CSVs
├── network_analysis_yeo/                # Network-level statistics
│   ├── {dataset}/
│   │   ├── *_network_analysis_results.json
│   │   ├── *_network_proportions.csv
│   │   └── *_roi_counts.csv
│   └── shared_{TD|ADHD|ASD}/
├── region_tables/                       # Human-readable ROI tables
│   ├── {dataset}_top_regions.csv
│   ├── shared_{TD|ADHD|ASD}_regions.csv
│   └── shared_{TD|ADHD|ASD}_regions_diverse_subset.csv  # For manuscripts
├── radar_plots/                         # Network radar charts (PNG + AI)
├── statistical_comparisons/             # Statistical comparison results
│   ├── master_summary.csv
│   ├── {dataset1}_vs_{dataset2}.csv
│   └── {dataset1}_vs_{dataset2}_networks.csv
├── brain_behavior/                      # Brain-behavior analysis
│   ├── {dataset}/
│   │   ├── elbow_plot.png
│   │   ├── scatter_{measure}.png + .ai
│   │   ├── linear_regression_results.csv
│   │   └── PC*_loadings.csv
│   └── combined_plots/
│       ├── hyperactivity_td_cohorts.png + .ai
│       └── inattention_td_cohorts.png + .ai
├── brain_behavior_optimized/            # Optimized brain-behavior (NEW)
│   └── {dataset}/
│       ├── summary.csv
│       ├── {measure}.png + .ai
│       └── {measure}_optimization.csv
└── brain_age_plots/                     # Brain age prediction plots
    ├── td_cohorts_combined_scatter.png + .ai
    ├── adhd_cohorts_combined_scatter.png + .ai
    └── asd_cohorts_combined_scatter.png + .ai
```

---

## 🎨 Plot Styling

**All plots use consistent, publication-ready styling**:

### Scatter Plots (Brain-Behavior & Brain Age)
- **Font**: Arial
- **Dots**: #5A6FA8 (bluer color), α=0.7, size=80
- **Regression Line**: #D32F2F (red), width=2.5
- **Statistics**: Bottom-right corner, NO bounding box
- **Spines**: NO top/right spines, left/bottom width=1.5
- **Ticks**: Major ticks only (length=6, width=1.5)
- **Export**: PNG (300 dpi) + AI (Adobe Illustrator)

### Radar Plots (Network Analysis)
- **Font**: Arial
- **Fill**: #4ECDC4 (modern teal/cyan), α=0.6
- **Line**: #1A535C (dark teal), width=2.5
- **Background**: #F8F9FA (light gray)
- **Grid**: Subtle radial lines
- **Export**: PNG (300 dpi) + AI

### Heatmaps (PC Loadings)
- **Colormap**: Red-blue diverging
- **Font**: Arial
- **Export**: PNG (300 dpi)

---

## 🚀 Recommended Workflow Order

```bash
# 1. Setup environment
conda activate brain_age

# 2. Network analysis
cd scripts
python create_count_data.py
python run_network_analysis.py
python create_region_tables.py
python create_polar_network_plots.py

# 3. Statistical comparisons
python run_statistical_comparisons.py

# 4. Brain-behavior (choose A or B)

# Option A: Standard analysis
python run_nki_brain_behavior_enhanced.py
python run_adhd200_brain_behavior_enhanced.py
python run_cmihbn_brain_behavior_enhanced.py
python run_adhd200_adhd_brain_behavior_enhanced.py
python run_cmihbn_adhd_brain_behavior_enhanced.py

# Option B: Optimized analysis (RECOMMENDED)
python run_optimized_brain_behavior.py --all

# 5. Combined brain-behavior plots
python plot_brain_behavior_td_cohorts.py
python plot_brain_behavior_custom_1x3.py
python plot_pc_loadings_heatmap.py --dataset nki_rs_td
python plot_pc_loadings_heatmap.py --dataset adhd200_td
python plot_pc_loadings_heatmap.py --dataset cmihbn_td

# 6. Brain age plots
python plot_brain_age_td_cohorts.py
python plot_brain_age_adhd_cohorts.py
python plot_brain_age_asd_cohorts.py
```

---

## 📝 Key Scripts Summary

| Script | Generates Plots? | Datasets | Output Type |
|--------|-----------------|----------|-------------|
| `create_count_data.py` | ❌ | All | CSV tables |
| `run_network_analysis.py` | ❌ | All | JSON + CSV |
| `create_region_tables.py` | ❌ | All | CSV tables |
| `create_polar_network_plots.py` | ✅ | All | PNG + AI (7 radar plots) |
| `run_statistical_comparisons.py` | ❌ | All | CSV tables |
| `run_*_brain_behavior_enhanced.py` | ✅ | Individual | PNG + AI (scatter + elbow) |
| `run_optimized_brain_behavior.py` | ✅ | All | PNG + AI (optimized) |
| `plot_brain_behavior_td_cohorts.py` | ✅ | TD | PNG + AI (2 combined) |
| `plot_brain_behavior_custom_1x3.py` | ✅ | NKI + ADHD200 | PNG + AI (1 combined) |
| `plot_pc_loadings_heatmap.py` | ✅ | TD | PNG (3 heatmaps) |
| `plot_brain_age_*.py` | ✅ | TD/ADHD/ASD | PNG + AI (3 combined) |

**Total Plots Generated**: ~50+ individual plots + combined figures

---

## 📖 Additional Documentation

### Diverse Subset Tables for Manuscripts

The `shared_*_regions_diverse_subset.csv` files ensure network diversity:
- Max 30 regions total
- Max 3 regions per brain network
- Prioritizes high-count regions while maintaining diversity
- Prevents repetitive regions (e.g., MTG ×4, PFC ×6)
- Ideal for manuscript tables

### Statistical Comparison Interpretation

**Cosine Similarity**:
- >0.9: Very high agreement
- 0.7-0.9: High agreement
- 0.5-0.7: Moderate agreement
- <0.5: Low agreement

**Spearman ρ**:
- >0.8: Very strong correlation
- 0.6-0.8: Strong correlation
- 0.4-0.6: Moderate correlation
- <0.4: Weak correlation

**Aitchison Distance**: Lower = more similar (compositional structure)  
**Jensen-Shannon Divergence**: Lower = more similar (distributions)

### Brain-Behavior Optimization Benefits

The optimized pipeline (`run_optimized_brain_behavior.py`) provides:
- **Higher correlations**: Regularization + optimal PCs
- **Faster execution**: Parallel CV, vectorized operations
- **Less code**: 76% reduction vs separate scripts
- **Data integrity**: Automated checks for ID alignment, NaN, duplicates
- **Robust models**: Cross-validated, prevents overfitting

---

## 🐛 Troubleshooting

**PyTorch installation issues**:
```bash
pip install --upgrade pip
pip install torch==1.13.1 --index-url https://download.pytorch.org/whl/cpu
```

**Pandas building from source**:
```bash
pip install pandas==1.5.3 --only-binary :all:
```

**Font not found**:
Ensure Arial font is available at:
`/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/clustering_analysis/arial.ttf`

---

## 📧 Contact

For questions or issues, contact the Menon Lab at Stanford University.

**Last Updated**: 2024

