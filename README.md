# Age Prediction Analysis Pipeline

A comprehensive pipeline for brain age prediction analysis using pre-trained models and existing data files. This repository provides tools for feature attribution, network-level analysis, brain-behavior correlations, and statistical analysis.

## 📦 **Environment Setup**

### **Required Packages**

The pipeline requires Python 3.8+ with the following packages:

```bash
# Core scientific computing
numpy
pandas
scipy
scikit-learn

# Visualization
matplotlib
seaborn

# Data handling
openpyxl  # For Excel file support
tabulate  # For pretty console tables

# Deep learning (if using models)
torch
```

### **Installation on HPC**

**Option 1: Use existing environment**
```bash
# Activate the shared environment
source /oak/stanford/groups/menon/software/python_envs/atif_env/bin/activate
```

**Option 2: Create your own environment**
```bash
# Create conda environment
conda create -n brain_age python=3.10
conda activate brain_age

# Install packages
pip install numpy pandas scipy scikit-learn matplotlib seaborn openpyxl tabulate torch
```

**Option 3: Install from requirements file**
```bash
# Navigate to the repository on HPC
cd /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test

# Upgrade pip first (important!)
pip install --upgrade pip setuptools wheel

# Install all dependencies
pip install -r requirements.txt
```

**Troubleshooting Installation Issues:**

If pandas fails to install (build errors on HPC):
```bash
# Use older pandas version with pre-built wheels
pip install pandas==1.5.3

# Then install other packages
pip install numpy scipy scikit-learn matplotlib seaborn openpyxl tabulate
```

If still having issues on HPC:
```bash
# Force pre-built wheels only (no source builds)
cd /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test
pip install --only-binary :all: -r requirements.txt
```

## 🎯 **Quick Start - Complete Workflow**

### **Option 1: Run Everything at Once (Recommended)**

Use the pipeline runner for beautiful, clear output:

```bash
cd /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/scripts/
python run_pipeline.py
```

This will run all steps with:
- ✅ Clear section headers
- ✅ Progress indicators
- ✅ Color-coded success/warning/error messages
- ✅ Timing information
- ✅ Summary of output files

### **Option 2: Run Steps Individually**

**Run the complete pipeline step-by-step with these commands on HPC:**

```bash
# Navigate to scripts directory
cd /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/scripts/

# Optional: Clean up old PDF files (if you have any from previous runs)
find /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results -name "*.pdf" -type f -delete

# Step 1: Convert Excel count data to CSV format
# Converts Excel files from original repository to CSV files in test directory
python convert_count_data.py

# Step 2: Run network analysis on individual datasets
# Creates radar plots for each dataset using count data
python network_analysis_yeo.py --process_all

# Step 3: Run shared network analysis across cohorts
# Creates radar plots using minimum overlap count for TD, ADHD, and ASD
python network_analysis_yeo.py --process_shared

# Step 4: Create region tables (individual + shared + diverse subsets)
# Generates CSV tables with brain regions and counts
# Includes diverse subsets for manuscript tables (max 30 regions, max 3 per network)
python create_region_tables.py \
  --config /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/config.yaml \
  --output_dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/region_tables

# Step 5: Generate brain age prediction plots
# Note: .npz files should be uploaded to /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_age_predictions/npz_files/
# The scripts will automatically look in this directory
# TD Cohorts (2x2 layout: HCP-Dev, NKI, CMI-HBN TD, ADHD200 TD)
python plot_brain_age_td_cohorts.py \
  --output_dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_age_plots

# ADHD Cohorts (1x2 layout: CMI-HBN ADHD, ADHD200 ADHD)
python plot_brain_age_adhd_cohorts.py \
  --output_dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_age_plots

# ASD Cohorts (1x2 layout: ABIDE ASD, Stanford ASD)
python plot_brain_age_asd_cohorts.py \
  --output_dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_age_plots

# Step 6 (Optional): Brain-behavior correlation analysis for TD cohorts

# Option A: Quick analysis (CSV results only)
bash run_nki_brain_behavior_only.sh
bash run_adhd200_td_brain_behavior_only.sh
bash run_cmihbn_td_brain_behavior_only.sh

# Option B: Full analysis with plots (recommended) - NO ARGUMENTS NEEDED
# All paths are pre-configured in the scripts. Just run:
python run_nki_brain_behavior_enhanced.py
python run_adhd200_brain_behavior_enhanced.py
python run_cmihbn_brain_behavior_enhanced.py

# Each script does everything in one go:
#   - Loads data and performs PCA with elbow method
#   - Uses ALL PCs in linear regression to predict behavioral scores
#   - Calculates Spearman correlation (predicted vs actual behavior)
#   - Creates scatter plots, elbow plots
#   - Ranks PC importance (which PCs contribute most)
#   - Shows top brain regions per PC

# Step 6b (Optional): Create PC loadings heatmaps
# Shows which brain regions load most strongly on first 3 PCs
python plot_pc_loadings_heatmap.py --dataset nki_rs_td
python plot_pc_loadings_heatmap.py --dataset adhd200_td
python plot_pc_loadings_heatmap.py --dataset cmihbn_td

# Step 6c (Optional): Create combined 3-panel plots for hyperactivity and inattention
python plot_brain_behavior_td_cohorts.py

# Step 7 (Optional): Cosine similarity analysis
# Compares feature importance maps across TD, ADHD, and ASD cohorts
python cosine_similarity_analysis.py

# Optional: Analyze shared TD regions with high counts
# (Install tabulate for pretty tables: pip install tabulate)
python analyze_td_shared_regions.py --threshold 450
```

## 📥 **Downloading Results**

**All PNG plots are saved to:**
```
/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/
```

**Key folders for files to download:**

1. **Network Analysis Plots** (Steps 2 & 3) - PNG files:
   - Individual datasets: `results/network_analysis_yeo/{dataset_name}/{dataset_name}_network_radar_plot.png`
     - 8 datasets: dev, nki, adhd200_td, cmihbn_td, adhd200_adhd, cmihbn_adhd, abide_asd, stanford_asd
   - Shared cohorts: `results/network_analysis_yeo/shared_{TD,ADHD,ASD}/shared_network_radar.png`
     - 3 files: shared_TD, shared_ADHD, shared_ASD

2. **Region Tables** (Step 4) - CSV files:
   - Individual: `results/region_tables/{dataset_name}_region_table.csv` (8 files)
   - Shared (full): `results/region_tables/shared_regions_{TD,ADHD,ASD}.csv` (3 files - all consistently identified regions)
   - Shared (diverse subsets): `results/region_tables/shared_regions_{TD,ADHD,ASD}_diverse_subset.csv` (3 files - for manuscript tables)

3. **Brain Age Plots** (Step 5) - PNG files:
   - Location: `results/brain_age_plots/`
   - Files:
     - `td_cohorts_combined_scatter.png` (2x2 subplot: HCP-Dev, NKI, CMI-HBN TD, ADHD200 TD)
     - `adhd_cohorts_combined_scatter.png` (1x2 subplot: CMI-HBN ADHD, ADHD200 ADHD)
     - `asd_cohorts_combined_scatter.png` (1x2 subplot: ABIDE ASD, Stanford ASD)

4. **Brain-Behavior Plots** (Step 6, optional) - PNG files:
   - Location: `results/brain_behavior/{dataset_name}/`
   - Individual scatter plots: `{behavioral_measure}_scatter.png`
   - Elbow plots: `elbow_plot.png`
   - PC loadings heatmaps: `pc_loadings_heatmap.png`
   - Combined 3-panel plots: `results/brain_behavior/combined_plots/`
     - `hyperactivity_combined.png` (NKI, ADHD200, CMI-HBN)
     - `inattention_combined.png` (NKI, ADHD200, CMI-HBN)

5. **Cosine Similarity Analysis** (Step 7, optional) - CSV and PNG files:
   - Location: `results/cosine_similarity/`
   - `cosine_similarity_results.csv` - Similarity values and ranges
   - `cosine_similarity_heatmap.png` - 3x3 heatmap (TD, ADHD, ASD)

**To download files from HPC:**
```bash
# From your local machine, run:

# Download all PNG plots
scp -r username@login.sherlock.stanford.edu:/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_analysis_yeo/ ./local_folder/network_plots/
scp -r username@login.sherlock.stanford.edu:/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_age_plots/ ./local_folder/brain_age_plots/

# Download CSV tables
scp -r username@login.sherlock.stanford.edu:/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/region_tables/ ./local_folder/region_tables/

# Or download entire results folder (includes everything):
scp -r username@login.sherlock.stanford.edu:/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/ ./local_folder/
```

## 📁 **Output Files**

All results are saved to: `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/`

### **Step 1: Converted CSV Files**
- **Location**: `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/count_data/`
- **Files**: `{dataset_name}_count_data.csv` for each dataset
- **Purpose**: CSV versions of Excel count data files for faster processing

### **Step 2: Individual Network Analysis**
- **Location**: `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_analysis_yeo/{dataset_name}/`
- **Files per dataset**:
  - `{dataset_name}_network_analysis.csv` - Network-level aggregated data
  - `{dataset_name}_network_radar_plot.png` - Radar chart visualization
  - `{dataset_name}_network_analysis_results.json` - Analysis results
- **Datasets**: dev, nki, adhd200_td, cmihbn_td, adhd200_adhd, cmihbn_adhd, abide_asd, stanford_asd

### **Step 3: Shared Network Analysis**
- **Location**: `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_analysis_yeo/`
- **Folders**:
  - `shared_TD/` - TD cohorts (HCP-Dev, NKI, CMI-HBN TD, ADHD200 TD)
  - `shared_ADHD/` - ADHD cohorts (CMI-HBN ADHD, ADHD200 ADHD)
  - `shared_ASD/` - ASD cohorts (ABIDE ASD, Stanford ASD)
- **Files per folder**:
  - `shared_network_analysis.csv` - Network-level aggregated data
  - `shared_network_radar.png` - Radar chart with minimum overlap counts
  - `shared_network_analysis_results.json` - Analysis results

### **Step 4: Region Tables**
- **Location**: `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/region_tables/`
- **Individual Tables**: `{dataset_name}_region_table.csv` - One per dataset (all regions with count ≥ 289)
  - Examples: `dev_region_table.csv`, `nki_region_table.csv`, `adhd200_td_region_table.csv`, etc.
- **Shared Region Tables** (regions consistently identified across cohorts):
  - **Full tables** (all consistently identified regions):
    - `shared_regions_TD.csv` - TD cohorts (HCP-Dev, NKI, CMI-HBN TD, ADHD200 TD)
    - `shared_regions_ADHD.csv` - ADHD cohorts (CMI-HBN ADHD, ADHD200 ADHD)
    - `shared_regions_ASD.csv` - ASD cohorts (ABIDE ASD, Stanford ASD)
  - **Diverse subset tables** (for manuscript - network-diverse representation):
    - `shared_regions_TD_diverse_subset.csv` - Max 30 regions, max 3 per brain area
    - `shared_regions_ADHD_diverse_subset.csv` - Max 30 regions, max 3 per brain area
    - `shared_regions_ASD_diverse_subset.csv` - Max 30 regions, max 3 per brain area
- **Format**: CSV with columns: Brain Regions, Subdivision, (ID) Region Label, Count
- **Count Method**: Minimum count across datasets for shared regions
- **Filtering**: 
  - **Individual tables**: Count ≥ 289 (significance threshold: 289/500 = 58%)
  - **Shared tables (full)**: Count ≥ 289, appearing in ≥2 datasets within cohort group
  - **Diverse subsets**: Selected from full shared tables to ensure network diversity (avoids repetition of same regions like MTG×4, PFC×6)

#### **Using Diverse Subset Tables for Manuscript**

The diverse subset tables are designed for manuscript reporting and ensure representation across different brain networks. These tables:

- **Limit repetition**: Maximum 3 regions per brain area (e.g., prevents listing MTG 4 times, PFC 6 times)
- **Prioritize high counts**: Regions are selected in descending order by count
- **Ensure diversity**: Represents multiple functional networks (default mode, salience, frontoparietal, etc.)
- **Manageable length**: Maximum 30 regions per table (suitable for Word document tables)

**Manuscript text template:**
> "Results revealed that key nodes within the [default mode network/salience network/frontoparietal network] play a critical role (**Table X**). These regions demonstrated remarkable consistency across both the [HCP-Dev/NKI/etc.] cohorts, with cosine similarity between feature importance maps ranging from X.XX to X.XX. Notably, consistently identified regions included [list top regions from diverse subset table]. At the network level, the [network name] emerged as shared functional substrates that reliably captured [normative brain development/aging patterns in ADHD/ASD population] (**Figure Y**)."

**For manuscript tables**, use the `*_diverse_subset.csv` files which provide a curated, network-diverse selection of the most important regions.

### **Step 5: Brain Age Plots**
- **Location**: `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_age_plots/`
- **Files**:
  - `td_cohorts_combined_scatter.png` - 2x2 subplot (4 core TD datasets)
  - `adhd_cohorts_combined_scatter.png` - 1x2 subplot (2 ADHD datasets)
  - `asd_cohorts_combined_scatter.png` - 1x2 subplot (2 ASD datasets)
- **Input**: .npz files in `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_age_predictions/npz_files/`

### **Step 6: Brain-Behavior Analysis (Optional)**
- **Location**: `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/`
- **Subdirectories**:
  - `nki_rs_td/` - NKI-RS TD analysis results
  - `adhd200_td/` - ADHD200 TD analysis results (NYU site only)
  - `cmihbn_td/` - CMI-HBN TD analysis results
  - `combined_plots/` - 3-panel combined plots for hyperactivity and inattention
- **Files per dataset**:
  - `elbow_plot.png` - PCA variance explained
  - `{behavioral_measure}_scatter.png` - Predicted vs observed behavioral scores
  - `pc_loadings_heatmap.png` - Brain region loadings on first 3 PCs
  - `linear_regression_results.csv` - Statistics (N, ρ, p-value, R²)
  - `pc_loadings_top_regions.csv` - Top brain regions per PC

### **Step 7: Cosine Similarity Analysis (Optional)**
- **Location**: `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/cosine_similarity/`
- **Purpose**: Quantify similarity of brain feature importance maps across TD, ADHD, and ASD cohorts
- **Method**: 
  - Computes cosine similarity between mean feature maps (IG scores)
  - Provides both pooled group-level similarities and pairwise dataset ranges
  - Reports min, max, mean, and std across all pairwise comparisons
- **Cohorts analyzed**:
  - TD (pooled): HCP-Dev, NKI, CMI-HBN TD, ADHD200 TD
  - ADHD (pooled): ADHD200 ADHD, CMI-HBN ADHD
  - ASD (pooled): ABIDE ASD, Stanford ASD
- **Files**:
  - `cosine_similarity_results.csv` - Numerical results with ranges
  - `cosine_similarity_heatmap.png` - 3x3 similarity matrix visualization
- **Usage**: `python cosine_similarity_analysis.py` (no arguments needed)
- **Output**: Console report with manuscript-ready text for similarity ranges

## 🔧 **Configuration**

The pipeline uses `config.yaml` for all file paths and parameters:

**Config file location**: `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/config.yaml`

Key paths configured:
```yaml
network_analysis:
  count_data:
    dev: "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/figures/dev/ig_files/top_50_consensus_features_hcp_dev_aging.xlsx"
    nki: "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/figures/nki/ig_files/top_50_consensus_features_nki_cog_dev_aging.xlsx"
    adhd200_td: "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/figures/adhd200/ig_files_td/top_50_consensus_features_adhd200_td_aging.xlsx"
    cmihbn_td: "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/figures/cmihbn/ig_files_td/top_50_consensus_features_cmihbn_td_aging.xlsx"
    adhd200_adhd: "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/figures/adhd200/ig_files/top_50_consensus_features_adhd200_adhd_aging.xlsx"
    cmihbn_adhd: "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/figures/cmihbn/ig_files/top_50_consensus_features_cmihbn_adhd_aging.xlsx"
    abide_asd: "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/figures/abide/ig_files/top_50_consensus_features_abide_asd_aging.xlsx"
    stanford_asd: "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/figures/stanford/ig_files/top_50_consensus_features_stanford_asd_aging.xlsx"
  yeo_atlas_path: "/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/feature_attribution/csv_files/subregion_func_network_Yeo_updated_yz.csv"
  roi_labels_path: "/oak/stanford/groups/menon/projects/cdla/2021_hcp_earlypsychosis/scripts/restfmri/classify/CNN1dPyTorch/brainnetome_roi_labels.txt"
```

**Note**: Original data files (Excel files, atlas files) remain in the `2024_age_prediction` directory. Only results and converted files are stored in `2024_age_prediction_test`.

## 📊 **Available Datasets**

### **TD Cohorts (Core 4 for Analysis)**
- `dev` - HCP-Dev
- `nki` - NKI
- `adhd200_td` - ADHD200 TD
- `cmihbn_td` - CMI-HBN TD

### **ADHD Cohorts**
- `adhd200_adhd` - ADHD200 ADHD
- `cmihbn_adhd` - CMI-HBN ADHD

### **ASD Cohorts**
- `abide_asd` - ABIDE ASD
- `stanford_asd` - Stanford ASD

## 🎨 **Plot Features**

### **Brain Age Scatter Plots**
- **Subplot Layout**: Each dataset in separate panel
- **Styling**: Blue dots with blue edges, no grid, clean spines
- **Statistics**: R², MAE, P-value, N displayed per subplot
- **Format**: PNG only
- **TD Layout**: 2x2 (4 core datasets: HCP-Dev, NKI, CMI-HBN TD, ADHD200 TD)
- **ADHD Layout**: 1x2 (CMI-HBN ADHD, ADHD200 ADHD)
- **ASD Layout**: 1x2 (ABIDE ASD, Stanford ASD)

### **Network Analysis Radar Plots**
- **Type**: Polar area plots (radar charts)
- **Styling**: Light blue fill, dark blue lines, no grid
- **Network Mapping**: Yeo 17-network atlas
- **Scaling**: Intelligent min-max scaling for shared cohorts when values are similar

### **Brain Age Prediction Plots**
- **Layout**: Subplots for each dataset (2x2 for TD, 1x2 for ADHD/ASD)
- **Styling**: 
  - No identity line (only regression line)
  - No grid
  - No top/right spines
  - Blue dots with blue edges (size: 50)
- **Statistics**: R², MAE, P-value (< 0.001 format), N
- **Titles**: Clear dataset names for each subplot

### **Brain-Behavior Correlation Analysis**

**Individual TD Cohort Scripts (Recommended for Testing):**

Each cohort has a dedicated bash script with pre-configured paths:

**1. NKI-RS TD (`run_nki_brain_behavior_only.sh`):**
- **Script**: `brain_behavior_nki.py`
- **Data Sources**:
  - IG CSV: `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/integrated_gradients/nki_cog_dev_wIDS_features_IG_convnet_regressor_single_model_fold_0.csv`
  - CAARS Behavioral: `/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/FLUX/assessment_data/8100_CAARS-S-S_20191009.csv`
- **Features**:
  - Auto-detects CAARS columns (Inattention, Hyperactivity T-scores and totals)
  - Handles non-numeric data gracefully
  - Drops 'Unnamed: 0' index column from IG CSV
  - Sanitizes output filenames
- **Usage**:
```bash
cd /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/scripts
bash run_nki_brain_behavior_only.sh
```

**2. ADHD200 TD (`run_adhd200_td_brain_behavior_only.sh`):**
- **Script**: `brain_behavior_td_simple.py`
- **Data Sources**:
  - PKLZ File: `/oak/stanford/groups/menon/deriveddata/public/adhd200/restfmri/timeseries/group_level/brainnetome/normz/adhd200_run-rest_brainnetome_mean_regMov-6param_wmcsf_dt1_bpf008-09_normz_246ROIs.pklz` (⚠️ **WITHOUT** `_nn` suffix)
  - IG CSV: `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/count_data/adhd200_td_count_data.csv`
  - Behavioral: Embedded in .pklz file (Hyper/Impulsive, Inattentive columns)
- **Features**:
  - Loads single .pklz file
  - Filters for TD subjects (DX/label == 0)
  - Filters for quality (mean_fd < 0.5)
  - Handles NaNs in behavioral columns
- **Usage**:
```bash
cd /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/scripts
bash run_adhd200_td_brain_behavior_only.sh
```

**3. CMI-HBN TD (`run_cmihbn_td_brain_behavior_only.sh`):**
- **Script**: `brain_behavior_td_simple.py`
- **Data Sources**:
  - PKLZ Directory: `/oak/stanford/groups/menon/deriveddata/public/cmihbn/restfmri/timeseries/group_level/brainnetome/normz/` (⚠️ **Directory**, not single file)
  - IG CSV: `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/count_data/cmihbn_td_count_data.csv`
  - C3SR Behavioral: `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/prepare_data/cmihbn/behavior/` (auto-detects Conners CSV)
- **Features**:
  - Loads all run1 .pklz files from directory
  - Concatenates multiple files
  - Filters for valid subjects (label != 99)
  - Filters for TD subjects (label == 0)
  - Filters for quality (mean_fd < 0.5)
  - Merges with C3SR behavioral data (C3SR_HY_T, C3SR_IN_T)
- **Usage**:
```bash
cd /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/scripts
bash run_cmihbn_td_brain_behavior_only.sh
```

**Linear Regression Approach (Enhanced Script):**
1. Load IG scores from CSV (with subject IDs)
2. Load behavioral data (CAARS for NKI, embedded for ADHD200, C3SR for CMI-HBN)
3. Match subjects between IG and behavioral data
4. Perform PCA on IG scores → determine optimal # of PCs via elbow method
5. **Use ALL PCs together** in linear regression to predict behavioral scores
6. Calculate Spearman correlation between predicted and actual behavioral scores
7. Rank PC importance (absolute regression coefficients)
8. Create scatter plots (predicted vs actual behavioral scores)
9. Generate PC loading reports (top 10 brain regions contributing to each PC)
10. Save results: scatter plots (PNG), importance rankings (CSV), PC loadings (CSV)

**Key Difference from Individual PC Correlation:**
- Uses **all PCs simultaneously** in a regression model (not one at a time)
- Predicts behavioral scores from brain features
- Reports **one correlation per behavioral measure** (model performance)
- Shows which PCs are most important for prediction

**Expected Output Files (Enhanced Script):**
- **Per Dataset**: 
  - Elbow plot: `{dataset}_pca_elbow_plot.png` (optimal PC selection)
  - PC loadings: `{dataset}_pc_loadings.csv` (top 10 brain regions per PC)
  
- **Per Behavioral Measure**:
  - Scatter plot: `{dataset}_predicted_vs_actual_{behavior}.png` (predicted vs actual behavioral scores)
  - Results CSV: `{dataset}_{behavior}_regression_results.csv` (Spearman ρ, p-value, R² CV)
  - PC importance: `{dataset}_{behavior}_pc_importance.csv` (ranked PCs by contribution)

**Example for NKI-RS TD** (4 CAARS measures):
- `nki_rs_td_pca_elbow_plot.png`
- `nki_rs_td_pc_loadings.csv`
- `nki_rs_td_predicted_vs_actual_CAARS_A_TOTAL.png` (×4 behavioral measures)
- `nki_rs_td_CAARS_A_TOTAL_regression_results.csv` (×4)
- `nki_rs_td_CAARS_A_TOTAL_pc_importance.csv` (×4)

**Results CSV Contains**: Spearman_r, P_value, R2_CV_mean, R2_CV_std, N_subjects, N_PCs_used

**PC Importance CSV Contains**: PC (ranked by importance), Importance (abs coefficient), Rank

**PC Loadings CSV Contains**: PC, Region_1, Loading_1, Abs_Loading_1, ..., Region_10, Loading_10, Abs_Loading_10

**Plot Features**: Predicted vs actual scatter, regression line, Spearman ρ and p-value (< 0.001 format) in bottom-right, no grid, no top/right spines

**Enhanced Analysis Scripts (Option B - Recommended):**

Three standalone scripts with all paths pre-configured (no arguments needed):

**1. NKI-RS TD** (`run_nki_brain_behavior_enhanced.py`):
- **Pre-configured paths**:
  - IG CSV: `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/integrated_gradients/nki_cog_dev_wIDS_features_IG_convnet_regressor_single_model_fold_0.csv`
  - Behavioral DIR: `/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/FLUX/assessment_data` (loads CAARS, Conners 3, RBS files)
  - Output: `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/nki_rs_td`
- **Usage**: `python run_nki_brain_behavior_enhanced.py`

**2. ADHD200 TD** (`run_adhd200_brain_behavior_enhanced.py`):
- **Pre-configured paths**:
  - PKLZ: `/oak/stanford/groups/menon/deriveddata/public/adhd200/restfmri/timeseries/group_level/brainnetome/normz/adhd200_run-rest_brainnetome_mean_regMov-6param_wmcsf_dt1_bpf008-09_normz_246ROIs.pklz`
  - IG CSV: `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/integrated_gradients/adhd200_td_features_all_sites_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS_single_model_predictions.csv`
  - Output: `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/adhd200_td`
- **Site filtering**: **NYU site only** (to avoid scale differences between NYU and Peking)
- **Outlier removal**: Removes values >3 SD from mean
- **Usage**: `python run_adhd200_brain_behavior_enhanced.py`

**3. CMI-HBN TD** (`run_cmihbn_brain_behavior_enhanced.py`):
- **Pre-configured paths**:
  - PKLZ Directory: `/oak/stanford/groups/menon/deriveddata/public/cmihbn/restfmri/timeseries/group_level/brainnetome/normz`
  - IG CSV: `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/integrated_gradients/cmihbn_td_features_all_sites_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS_single_model_predictions.csv`
  - C3SR File: `/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/prepare_data/adhd/C3SR.csv`
  - Output: `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/cmihbn_td`
- **Usage**: `python run_cmihbn_brain_behavior_enhanced.py`

**What these scripts do:**
1. Load IG scores and behavioral data
2. Merge datasets by subject ID
3. Perform PCA with up to 50 components on **all subjects**
4. Create **elbow plot** to determine optimal number of PCs (80% variance threshold)
5. Fit **linear regression** on **all data** (no train/test split) using optimal PCs
6. Calculate **Spearman ρ** and **R²** between predicted and actual behavioral scores
7. Create **scatter plots** for each behavioral measure (with ρ and p-value)
8. Rank **PC importance** (absolute regression coefficients)
9. Extract **PC loadings** (top 10 brain regions per PC)
10. Save all results to CSV files and PNG plots

**Analysis approach:**
- **Descriptive analysis**: Fits on entire dataset (no cross-validation)
- **Goal**: Describe how well brain PCs explain behavioral variance in the sample
- **Metrics**: Spearman ρ (correlation), R² (variance explained), both on all data
- **NaN handling**: Each behavioral measure analyzed separately with available subjects

**Console output includes:**
- Data loading progress with subject counts
- Behavioral data availability (non-null counts per measure)
- PCA variance explained by components
- Optimal number of PCs selected (80% threshold)
- **N subjects** and **N features (PCs)** used for each measure
- **Spearman ρ** and **p-value** for each behavioral measure
- **R²** (variance explained) for each behavioral measure
- File save confirmations with filenames

**Output files:**
- `elbow_plot.png` - Scree plot and cumulative variance
- `scatter_{behavioral_measure}.png` - Predicted vs actual (one per measure, title = dataset name)
- `linear_regression_results.csv` - Summary table with behavioral measure, N subjects, N features, ρ, p-value, R²
- `pc_importance_{behavioral_measure}.csv` - PC rankings (one per measure)
- `PC{N}_loadings.csv` - Top 10 brain regions per PC

**PC loadings heatmap** (`plot_pc_loadings_heatmap.py`):
- Creates heatmap showing which brain regions load most strongly on first 3 PCs
- Similar to cognitive measure loadings, but for brain ROIs
- Red-blue colormap (red = positive loading, blue = negative loading)
- Shows top 15 regions (default) with highest absolute loadings
- Includes variance explained by each PC
- Output: `pc_loadings_heatmap.png` and `pc_loadings_top_regions.csv`
- **Usage**: 
  - `python plot_pc_loadings_heatmap.py --dataset nki_rs_td`
  - `python plot_pc_loadings_heatmap.py --dataset adhd200_td`
  - `python plot_pc_loadings_heatmap.py --dataset cmihbn_td`
  - Optional: `--n_top 20` to show more regions

**Combined visualization** (`plot_brain_behavior_td_cohorts.py`):
- Creates 3-panel subplot figures comparing all TD cohorts
- `hyperactivity_combined.png` - NKI-RS TD, ADHD200 TD, CMI-HBN TD side-by-side
- `inattention_combined.png` - NKI-RS TD, ADHD200 TD, CMI-HBN TD side-by-side
- Each panel shows scatter plot with N, ρ, p-value, R² annotations
- Output: `.../brain_behavior/combined_plots/`
- **Usage**: `python plot_brain_behavior_td_cohorts.py` (run after individual analyses)

**Comprehensive Script (`comprehensive_brain_behavior_analysis.py`):**
- **Datasets**: NKI-RS TD, ADHD-200 ADHD/TD, CMI-HBN ADHD/TD, ABIDE ASD, Stanford ASD, HCP-Dev
- **Styling**:
  - No identity line
  - No grid
  - No top/right spines
- **Statistics**: Pearson's r, P-value (with FDR correction)
- **Format**: PNG only

### **General Plotting Conventions**
All plots follow these conventions for publication-ready figures:
- ✅ **Font**: Arial (loaded from HPC path)
- ✅ **Clean aesthetics**: No grid, NO top/right spines (only left/bottom visible)
- ✅ **Tick marks**: Visible on left/bottom axes (direction='out', length=6, width=1.5, fontsize=12)
- ✅ **Consistent colors**: 
  - Data points: #5A6FA8 (darker blue/purple)
  - Best fit line: #D32F2F (red)
  - Alpha: 0.7 for dots, 0.9 for lines
  - Dot size: 80
- ✅ **Clear labels**: 
  - Axis labels: fontsize=14, normal weight
  - Titles: fontsize=16, bold, pad=15
- ✅ **Statistics placement**: Bottom-right corner, NO bounding box, fontsize=14
- ✅ **P-value format**: "P < 0.001" for very small p-values (using "P" not "p")
- ✅ **Correlation format**: "R = 0.XXX" (using "R" for Spearman ρ)
- ✅ **MAE format**: "MAE = X.XX years" (for brain age plots)
- ✅ **File format**: PNG only (no PDF or SVG)
- ✅ **Professional appearance**: Clean, minimal style for publication

### **Shared Region Analysis Methodology**
The pipeline uses a **top 50% percentile + significance threshold** approach, with **top 20% selection for shared regions**:

1. **IG Processing**: During integrated gradients analysis, only features (ROIs) in the top 50th percentile are counted
2. **Count Data**: Each dataset's count data already reflects only the top 50% of features
3. **Significance Threshold**: Only regions with counts ≥ 289 (out of 500 subjects, ~58%) are included
4. **Individual Tables**: All significant regions (count ≥ 289) are included
5. **Find Overlap**: Identify regions that appear in multiple datasets (from their significant regions)
6. **Top 20% Selection**: For overlap tables, select only the top 20% with highest minimum counts
7. **Minimum Count**: For shared regions, use the minimum count across datasets
8. **Network Mapping**: Map shared regions to Yeo 17-network atlas
9. **Radar Plots**: Visualize network-level aggregation of shared regions

**Example for TD cohorts:**
- HCP-Dev: ~123 regions (top 50% of 246 ROIs) → filter to count ≥ 289 → **all included in individual table**
- NKI: ~123 regions (top 50% of 246 ROIs) → filter to count ≥ 289 → **all included in individual table**
- CMI-HBN TD: ~123 regions (top 50% of 246 ROIs) → filter to count ≥ 289 → **all included in individual table**
- ADHD200 TD: ~123 regions (top 50% of 246 ROIs) → filter to count ≥ 289 → **all included in individual table**
- **Overlap**: Regions that appear in at least 2 datasets → **select top 20% of overlapping regions**
- **Radar plot**: Shows network-level aggregation of shared regions

This approach ensures individual tables show all **statistically significant** regions (≥289 counts), while overlap tables focus on the **strongest shared effects** (top 20% of overlapping regions) consistently identified across cohorts.

## 📂 **Repository Structure**

```
2024_age_prediction_test/
├── config.yaml                    # Configuration file
├── scripts/                       # Analysis scripts
│   ├── convert_count_data.py     # Step 1: Convert Excel to CSV
│   ├── network_analysis_yeo.py   # Steps 2-3: Network analysis
│   ├── create_region_tables.py   # Step 4: Create region tables
│   ├── plot_brain_age_td_cohorts.py      # Step 5: TD plots (2x2)
│   ├── plot_brain_age_adhd_cohorts.py    # Step 5: ADHD plots (1x2)
│   ├── plot_brain_age_asd_cohorts.py     # Step 5: ASD plots (1x2)
│   ├── create_polar_network_plots.py     # Radar chart functions
│   ├── plot_network_analysis.py  # Network comparison plots
│   ├── comprehensive_brain_behavior_analysis.py # Step 6: Brain-behavior
│   ├── cosine_similarity_analysis.py     # Similarity analysis
│   └── feature_comparison.py     # Feature comparison tools
├── utils/                         # Utility modules
│   ├── plotting_utils.py         # Plotting functions
│   ├── count_data_utils.py       # Data processing
│   └── statistical_utils.py      # Statistical analysis
└── results/                       # Output directory
    ├── count_data/               # Step 1: Converted CSV files
    ├── network_analysis_yeo/     # Steps 2-3: Network analysis
    │   ├── {dataset_name}/      # Individual dataset results
    │   ├── shared_TD/           # Shared TD analysis
    │   ├── shared_ADHD/         # Shared ADHD analysis
    │   └── shared_ASD/          # Shared ASD analysis
    ├── region_tables/            # Step 4: Region tables
    ├── brain_age_predictions/    # Brain age prediction data
    │   └── npz_files/           # .npz files with predicted/actual ages
    ├── brain_age_plots/          # Step 5: Brain age plots
    └── brain_behavior_analysis/  # Step 6: Brain-behavior results
```

## 🚀 **Key Features**

- **Modular Design**: Clean, organized scripts with reusable utilities
- **Publication-Ready Plots**: Standardized visualization with consistent aesthetics
- **PNG Output**: Single format for all plots (easy to use and share)
- **Statistical Rigor**: FDR correction for multiple comparisons
- **Network Analysis**: Yeo atlas-based network-level analysis with radar charts
- **Shared Analysis**: Minimum overlap count approach for cross-cohort comparisons
- **Organized Structure**: Clear directory organization for all results
- **Complete Workflow**: 6 easy steps from data conversion to final plots

## 📝 **Important Notes**

### **Directory Structure**
- **Original Data**: Remains in `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/`
- **Test Repository**: All new results in `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/`
- **No Overwriting**: Using `_test` suffix ensures original repository is preserved

### **File Formats**
- **Plots**: PNG only (no PDF or SVG)
- **Tables**: CSV format
- **Data**: .npz files for brain age predictions
- **Config**: YAML format

### **Network Analysis**
- **Individual Datasets**: Uses count data from each dataset
- **Shared Analysis**: Uses minimum overlap count across cohorts
- **Radar Plots**: All network visualizations use polar area plots (radar charts)
- **Core TD Datasets**: Only 4 datasets (HCP-Dev, NKI, CMI-HBN TD, ADHD200 TD)

### **Workflow Order**
1. **Convert data first** - Creates CSV files needed for subsequent steps
2. **Network analysis** - Processes individual and shared analyses
3. **Region tables** - Creates comprehensive region tables
4. **Brain age plots** - Generates visualization of predictions
5. **Brain-behavior** - Optional correlation analysis

## 🔍 **Troubleshooting**

### **Missing CSV Files**
If you see warnings about missing CSV files:
```bash
python convert_count_data.py
```

### **Missing Excel Files**
Excel files should be in the original repository:
- Path: `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/figures/{dataset}/ig_files/`
- Check `config.yaml` for correct paths

### **Network Analysis Errors**
If network analysis fails:
- Ensure CSV files exist in `results/count_data/`
- Check that `convert_count_data.py` completed successfully
- Verify Yeo atlas path in `config.yaml`

### **Column Name Errors**
If you see errors about 'attribution' vs 'Count':
- The code now handles both column names automatically
- CSV files use 'Count', Excel files may use 'attribution'

### **Import Errors**
If you get module import errors:
- Ensure you're running from the `scripts/` directory
- Check that `utils/` directory is in the parent directory
- Verify Python path includes the repository root

---

**Ready to run!** All scripts are configured with full HPC paths. Simply copy and paste the commands from the Quick Start section in order.