# Brain Age Prediction Analysis Pipeline

A comprehensive, production-quality pipeline for analyzing brain age prediction models across multiple datasets (TD, ADHD, ASD cohorts). Includes feature attribution, network analysis, brain-behavior correlations, statistical comparisons, and publication-ready visualizations.

---

## 🧠 Study Overview

![Study Overview](docs/study_overview.png)

### Research Pipeline

We develop a normative brain-age prediction model using spatiotemporal deep neural networks (stDNN) trained on rs-fMRI time series data from the HCP-Development cohort (healthy controls, Aim 1). The model is validated on independent typically developing (TD) and clinical cohorts (ADHD, ASD) to quantify deviations from normative maturation (brain-age gap, Aim 2).

**Integrated Gradients** - an explainable AI feature attribution method - yields region- and subject-level attributions ("brain fingerprints") that identify neurobiological features and brain networks underlying psychiatric disorders (Aims 3 & 4). This approach enables:

1. **Normative Model Development**: Train stDNN on HCP-Development cohort (246 brain regions × regional fMRI timeseries)
2. **Cross-Cohort Validation**: Apply model to TD, ADHD, and ASD cohorts to identify brain-age deviations
3. **Feature Attribution**: Use Integrated Gradients to identify critical brain regions and networks for age prediction
4. **Brain-Behavior Analysis**: Relate individual brain fingerprints to ADHD/ASD symptom severity

**Key Findings**:
- Consistent brain networks identified across TD cohorts (HCP-Dev, NKI-RS, ADHD-200 TD, CMI-HBN TD)
- Default mode network, salience network, and executive control networks are critical for normative brain development
- Brain fingerprints significantly correlate with behavioral measures in both TD and clinical populations
- High similarity in feature importance maps across cohorts (cosine similarity: 0.XXX-0.XXX)

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

### ⚡ Quick Start (Recommended)

```bash
# Activate shared environment
conda activate /oak/stanford/groups/menon/software/python_envs/brain_age_2024

# Verify installation
python scripts/verify_imports.py
```

### 📦 Full Installation Instructions

**See [`INSTALL.md`](INSTALL.md) for complete installation guide**, including:
- Copy-paste ready conda commands
- All package versions (tested for compatibility)
- GPU (CUDA 11.7) and CPU installation options
- Import verification script
- Troubleshooting guide
- SLURM job script example

### Required Packages Summary

```
# Core scientific computing
numpy==1.23.5, pandas==1.5.3, scipy==1.10.1, scikit-learn==1.2.2, statsmodels==0.14.0

# Deep learning
torch==1.13.1+cu117, pytorch-lightning==1.9.5, captum==0.6.0, timm==0.9.2

# Visualization
matplotlib==3.7.1, seaborn==0.12.2

# Utilities
openpyxl==3.1.2, pyyaml==6.0, tabulate==0.9.0, tensorboard==2.13.0, einops==0.6.1
```

**All versions tested - no conflicts!** ✅

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
├── integrated_gradients/                # IG feature attribution CSVs (input data)
├── network_analysis_yeo/                # Network-level statistics
│   ├── {dataset}/                       # Individual datasets
│   │   ├── *_network_analysis_results.json
│   │   ├── *_network_proportions.csv
│   │   └── *_roi_counts.csv
│   └── shared_{TD|ADHD|ASD}/           # Shared analyses
├── region_tables/                       # Human-readable ROI tables
│   ├── {dataset}_top_regions.csv
│   ├── shared_{TD|ADHD|ASD}_regions.csv
│   └── shared_{TD|ADHD|ASD}_regions_diverse_subset.csv  # For manuscripts
├── radar_plots/                         # Network radar charts (PNG + AI, NOT CREATED YET)
├── statistical_comparisons/             # Statistical comparison results (NOT CREATED YET)
│   ├── master_summary.csv
│   ├── {dataset1}_vs_{dataset2}.csv
│   └── {dataset1}_vs_{dataset2}_networks.csv
├── brain_behavior/                      # Brain-behavior analysis
│   ├── {dataset}/                       # NKI, ADHD200 TD/ADHD, CMI-HBN TD/ADHD
│   │   ├── elbow_plot.png
│   │   ├── scatter_{measure}.png + .ai
│   │   ├── linear_regression_results.csv
│   │   └── PC*_loadings.csv
│   └── combined_plots/
│       ├── hyperactivity_td_cohorts.png + .ai
│       └── inattention_td_cohorts.png + .ai
├── brain_behavior_optimized/            # Optimized brain-behavior (NOT CREATED YET)
│   └── {dataset}/
│       ├── summary.csv
│       ├── {measure}.png + .ai
│       └── {measure}_optimization.csv
├── brain_age_plots/                     # Brain age prediction plots
│   ├── Individual: {dataset}_age_scatter.png
│   ├── td_cohorts_combined_scatter.png + .ai + .svg
│   ├── adhd_cohorts_combined_scatter.png + .ai + .svg
│   └── asd_cohorts_combined_scatter.png + .ai + .svg
├── brain_age_predictions/               # NPZ files with predictions (input data)
│   └── npz_files/*.npz
└── cosine_similarity/                   # Legacy cosine similarity results
    ├── comprehensive_cosine_similarity_results.json
    ├── cosine_similarity_heatmap.png
    └── cosine_similarity_results.csv
```

---

## 🎨 Plot Styling - SINGLE SOURCE OF TRUTH

**All plots use `plot_styles.py` for 100% consistency and ZERO post-processing**:

### Scatter Plots (Brain-Behavior & Brain Age)
- **Font**: Arial, 18pt title (bold), 16pt labels, 14pt stats/ticks
- **Dots**: #5A6FA8 (fill AND edge - same color), α=0.7, size=100, edge width=1.5pt
- **Regression Line**: #D32F2F (red), width=3.0pt, α=1.0 (fully opaque)
- **Statistics**: Bottom-right corner, NO bounding box, black text
- **Spines**: NO top/right spines, left/bottom width=1.5pt, black
- **Ticks**: Major ticks ONLY (length=6pt, width=1.2pt), NO minor ticks, black labels
- **Figure Size**: (8, 6) inches for single plots, (6, 4.5) per subplot
- **Export**: PNG (300 dpi) + TIFF (300 dpi, LZW compression) + AI (vector format)

**Publication-Ready**: Optimized for direct use - no need to adjust fonts, line thickness, or colors in Affinity Designer!

**Centralized Module**: `scripts/plot_styles.py`
- All styling constants defined once
- Two main functions: `create_single_scatter_plot()` and `create_multi_panel_scatter()`
- Helper function: `get_dataset_title()` for clear, descriptive titles
- Change styling once → updates all 50+ plots automatically
- Guarantees plots can be placed side-by-side without visible differences

**Clear Dataset Titles**:
- `adhd200_td` → "ADHD-200 TD Subset (NYU)"
- `cmihbn_td` → "CMI-HBN TD Subset"
- `nki_rs_td` → "NKI-RS"
- `adhd200_adhd` → "ADHD-200 ADHD"
- `cmihbn_adhd` → "CMI-HBN ADHD"

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
| **Data Preparation** | | | |
| `create_count_data.py` | ❌ | All | CSV tables |
| **Network Analysis** | | | |
| `run_network_analysis.py` | ❌ | All | JSON + CSV |
| `create_region_tables.py` | ❌ | All | CSV tables |
| `create_polar_network_plots.py` | ✅ | All | PNG + AI (7 radar plots) |
| **Statistical Comparisons** | | | |
| `run_statistical_comparisons.py` | ❌ | All pairwise | CSV tables (12 comparisons) |
| `statistical_comparison_utils.py` | - | - | Reusable functions |
| **Brain-Behavior** | | | |
| `run_*_brain_behavior_enhanced.py` | ✅ | Individual | PNG + AI (scatter + elbow) |
| `run_optimized_brain_behavior.py` | ✅ | All | PNG + AI (optimized) |
| `brain_behavior_utils.py` | - | - | Reusable functions |
| `plot_brain_behavior_td_cohorts.py` | ✅ | TD | PNG + AI (2 combined) |
| `plot_brain_behavior_custom_1x3.py` | ✅ | NKI + ADHD200 | PNG + AI (1 combined) |
| `plot_pc_loadings_heatmap.py` | ✅ | TD | PNG (3 heatmaps) |
| **Brain Age** | | | |
| `plot_brain_age_*.py` | ✅ | TD/ADHD/ASD | PNG + AI + SVG (3 combined) |
| **Plotting Utilities** | | | |
| `plot_styles.py` | - | - | **SINGLE SOURCE OF TRUTH for styling** |

**Total Plots Generated**: ~50+ individual plots + combined figures  
**Plot Styling**: 100% consistent via `plot_styles.py`

---

## 📖 Additional Documentation

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

