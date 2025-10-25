# Age Prediction Analysis Pipeline

A comprehensive, modular pipeline for brain age prediction analysis using **pre-trained models** and existing data files. This repository provides tools for feature attribution, network-level analysis, brain-behavior correlations, and statistical analysis with proper multiple comparison correction.

## 🎯 **Ready-to-Use Pipeline**

This pipeline is designed to work with **existing trained models** and **pre-processed data files**:

- ✅ **Pre-trained HCP-Dev models** (PyTorch Lightning checkpoints + legacy PyTorch models)
- ✅ **Existing count data** (Excel files with consensus features)
- ✅ **Behavioral data** (CSV files with clinical measures)
- ✅ **Pre-processed imaging data** (.pklz and .bin files)
- ✅ **ROI labels and atlas files** (Brainnetome 246 ROI)

**No additional training required** - the pipeline uses existing models to generate results and plots.

## 📂 **Existing Data Files**

The pipeline is configured to use the following existing data files:

### **Pre-trained Models**
- **HCP-Dev Models**: `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/train_regression_models/dev/`
- **Legacy Model**: `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/train_regression_models/dev/best_outer_fold_0_hcp_dev_model_2_27_24.pt`

### **Count Data (Excel Files)**
- **HCP-Dev**: `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/figures/dev/ig_files/top_50_consensus_features_hcp_dev_aging.xlsx`
- **NKI**: `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/figures/nki/ig_files/top_50_consensus_features_nki_cog_dev_aging.xlsx`
- **ADHD200**: `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/figures/adhd200/ig_files/top_50_consensus_features_adhd200_*_aging.xlsx`
- **CMI-HBN**: `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/figures/cmihbn/ig_files/top_50_consensus_features_cmihbn_*_aging.xlsx`
- **ABIDE**: `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/figures/abide/ig_files/top_50_consensus_features_abide_asd_aging.xlsx`
- **Stanford**: `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/figures/stanford/ig_files/top_50_consensus_features_stanford_asd_aging.xlsx`

### **Behavioral Data**
- **NKI**: `8100_CAARS-S-S_20191009.csv` (CAARS_36, CAARS_37 for IN, HY)
- **CMI-HBN**: `/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/prepare_data/adhd/C3SR.csv` (C3SR_HY_T, C3SR_IN_T)
- **ADHD200**: `adhd200_run-rest_brainnetome_mean_regMov-6param_wmcsf_dt1_bpf008-09_normz_246ROIs_nn.pklz`
- **ABIDE**: ADOS Total, Social, Comm scores
- **Stanford**: `/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/prepare_data/stanford_autism/SRS_data_20230925.csv` (srs_total_score_standard)

### **Imaging Data (.bin files)**
- **HCP-Dev**: `/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/hcp_dev_age_five_fold/`
- **NKI**: `/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/nki_age_cog_dev_wIDs/fold_0.bin`
- **ADHD200**: `/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/adhd200_regression_age_*_wIDs/fold_0.bin`
- **CMI-HBN**: `/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/cmihbn_age*/fold_0.bin`
- **ABIDE**: `/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/abide_asd_*/fold_0.bin`
- **Stanford**: `/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/stanford_autism_*/fold_0.bin`

### **Atlas and Labels**
- **ROI Labels**: `/oak/stanford/groups/menon/projects/cdla/2021_hcp_earlypsychosis/scripts/restfmri/classify/CNN1dPyTorch/brainnetome_roi_labels.txt`
- **Yeo Atlas**: `/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/feature_attribution/csv_files/subregion_func_network_Yeo_updated_yz.csv`
- **Brain Atlas**: `/oak/stanford/groups/menon/projects/sryali/2019_DNN/scripts/features/BN_Atlas_246_2mm.nii`
- **Arial Font**: `/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/clustering_analysis/arial.ttf`

## ⚡ **Quick Start with Existing Models**

**Ready to run the complete pipeline with existing data:**

> **Note**: All commands below assume you are running from the `scripts/` directory. The `config.yaml` file is located in the parent directory, and results will be saved to `../results/`.

```bash
# Option 1: Retrain models with consistent architecture (recommended)
python brain_age_prediction.py \
  --config ../config.yaml \
  --retrain_models \
  --model_dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/train_regression_models/dev \
  --output_dir ../results/brain_age_prediction

# Option 2: Use existing pre-trained models (may have architecture issues)
python brain_age_prediction.py \
  --config ../config.yaml \
  --use_existing_models \
  --model_dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/train_regression_models/dev \
  --output_dir ../results/brain_age_prediction

# 2. Convert existing count data and create region tables
python convert_count_data.py
python create_region_tables.py

# 3. Generate all plots
python plot_brain_age_correlations.py \
  --results_file ../results/brain_age_prediction/brain_age_prediction_results_*.json \
  --output_dir ../results/figures/brain_age_plots

python plot_network_analysis.py \
  --config ../config.yaml \
  --output_dir ../results/figures/network_plots

# 4. Run brain-behavior analysis
python comprehensive_brain_behavior_analysis.py --dataset nki_rs_td
python comprehensive_brain_behavior_analysis.py --dataset adhd200_adhd
python comprehensive_brain_behavior_analysis.py --dataset abide_asd
```

**Expected Results:**
- ✅ Brain age predictions with bias correction
- ✅ Comprehensive brain age gap analysis
- ✅ Comprehensive region tables (individual + shared + overlap)
- ✅ Network analysis plots (polar plots only)
- ✅ Brain-behavior correlations with FDR correction

## 🚀 Features

- **Modular Architecture**: Clean, organized codebase with reusable utilities
- **Network Analysis**: Yeo atlas-based network-level feature analysis
- **Brain-Behavior Correlations**: Statistical analysis with FDR correction
- **Feature Attribution**: Integrated gradients and consensus analysis
- **Publication-Ready Plots**: Standardized visualization with consistent aesthetics
- **Comprehensive Documentation**: Detailed docstrings and usage examples
- **PEP8 Compliant**: Clean, readable code following Python standards

## 📁 Repository Structure

```
2024_age_prediction/
├── main.py                          # Main entry point for the pipeline
├── config.yaml                      # Configuration file
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── utils/                           # Utility modules
│   ├── __init__.py
│   ├── data_utils.py               # Data loading and preprocessing
│   ├── model_utils.py              # Neural network models and training
│   ├── plotting_utils.py           # Visualization functions
│   ├── statistical_utils.py        # Statistical analysis and corrections
│   ├── feature_utils.py            # Feature attribution and analysis
│   └── count_data_utils.py         # Count data processing utilities
│
├── scripts/                         # Analysis scripts
│   ├── brain_age_prediction.py     # Brain age prediction pipeline
│   ├── compute_integrated_gradients.py # IG computation for all datasets
│   ├── generate_count_data.py      # Generate count data from IG scores
│   ├── convert_count_data.py       # Convert Excel count data to CSV
│   ├── create_region_tables.py     # Create region importance tables
│   ├── network_analysis_yeo.py     # Yeo atlas network analysis
│   ├── cosine_similarity_analysis.py # Cosine similarity between cohorts
│   ├── comprehensive_brain_behavior_analysis.py # Brain-behavior correlations
│   ├── feature_comparison.py       # Feature comparison between cohorts
│   │
│   ├── plotting/                   # Separate plotting scripts
│   │   ├── plot_brain_age_correlations.py    # Brain age prediction plots
│   │   ├── plot_brain_visualization.py       # 3D brain surface plots
│   │   ├── plot_network_analysis.py          # Network analysis plots
│   │   └── plot_brain_behavior_analysis.py   # Brain-behavior plots
│   │
│   └── [legacy scripts]            # Original scripts (to be refactored)
│
├── results/                         # Output directory
│   ├── figures/                    # Generated plots
│   │   ├── brain_age_correlations/ # Brain age prediction plots
│   │   ├── brain_visualization/    # 3D brain surface plots
│   │   ├── network_analysis/       # Network analysis plots (polar plots only)
│   │   └── brain_behavior_analysis/ # Brain-behavior plots
│   ├── tables/                     # Statistical tables
│   ├── region_tables/              # Comprehensive region tables (individual + shared + overlap)
│   ├── count_data/                 # Count data CSV files
│   ├── network_analysis/           # Network analysis results
│   └── brain_behavior/             # Brain-behavior results
│
└── [data directories]              # Data files (not in repo)
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for model training)
- Conda or Miniconda

### Setup

#### Option 1: Using Conda (Recommended)

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd 2024_age_prediction
   ```

2. **Create a conda environment**:
   ```bash
   # Create environment with Python 3.9
   conda create -n age_prediction python=3.9 -y
   
   # Activate the environment
   conda activate age_prediction
   ```

3. **Install PyTorch with CUDA support** (if you have a CUDA-capable GPU):
   ```bash
   # For CUDA 11.8
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
   
   # For CUDA 12.1
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
   
   # For CPU only
   conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
   ```

4. **Install other dependencies**:
   ```bash
   # Install scientific computing packages
   conda install numpy pandas scipy scikit-learn matplotlib seaborn -c conda-forge -y
   
   # Install neuroimaging packages
   conda install nilearn -c conda-forge -y
   
   # Install PyTorch Lightning
   conda install pytorch-lightning -c conda-forge -y
   
   # Install remaining packages via pip
   pip install captum statsmodels pyyaml
   ```

5. **Verify installation**:
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   python -c "import utils; print('Installation successful!')"
   ```

#### Option 2: Using pip (Alternative)

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd 2024_age_prediction
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python -c "import utils; print('Installation successful!')"
   ```

### Environment Management

**To deactivate the environment**:
```bash
conda deactivate  # For conda
# or
deactivate        # For venv
```

**To remove the environment** (if needed):
```bash
conda env remove -n age_prediction  # For conda
# or
rm -rf venv                        # For venv
```

**To reactivate the environment**:
```bash
conda activate age_prediction  # For conda
# or
source venv/bin/activate       # For venv
```

## 🚀 Quick Start

### 1. Configuration

The `config.yaml` file is already configured with the correct paths for the HPC system. Key sections include:

```yaml
# Network Analysis Configuration
network_analysis:
  count_data:
    dev: "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/figures/dev/ig_files/top_50_consensus_features_hcp_dev_aging.xlsx"
    nki: "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/figures/nki/ig_files/top_50_consensus_features_nki_cog_dev_aging.xlsx"
    # ... other datasets

# Comprehensive Brain-Behavior Analysis Datasets
nki_rs_td:
  ig_csv: "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/count_data/nki_count_data.csv"
  behavioral_data: "/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/nki_age_cog_dev_wIDs/fold_0.bin"
# ... other datasets
```

**Important**: The CSV file paths point to the `2024_age_prediction_test` directory where `convert_count_data.py` saves the converted files.

### 2. Complete Analysis Workflow

**🎯 Using Pre-trained Models (Recommended)**

Follow this 9-step workflow using existing trained models and data:

> **Note**: Run all commands from the `scripts/` directory. Use `../config.yaml` for config file and `../results/` for output paths.

```bash
# Step 1: Retrain models with consistent architecture and test on external datasets
python brain_age_prediction.py \
  --config ../config.yaml \
  --retrain_models \
  --model_dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/train_regression_models/dev

# Step 2: Generate brain age prediction plots
python plot_brain_age_correlations.py \
  --results_file ../results/brain_age_prediction/brain_age_prediction_results_*.json \
  --output_dir ../results/figures/brain_age_plots

# Step 3: Compute Integrated Gradients for each dataset (using pre-trained model)
python compute_integrated_gradients.py --dataset nki_rs_td --fold 0
python compute_integrated_gradients.py --dataset cmihbn_td --fold 0
python compute_integrated_gradients.py --dataset adhd200_td --fold 0
python compute_integrated_gradients.py --dataset adhd200_adhd --fold 0
python compute_integrated_gradients.py --dataset cmihbn_adhd --fold 0
python compute_integrated_gradients.py --dataset abide_asd --fold 0
python compute_integrated_gradients.py --dataset stanford_asd --fold 0

# Step 4: Convert existing count data Excel files to CSV format
# (Converts Excel files from results/figures/*/ig_files/ to CSV files in test directory)
python convert_count_data.py
# This creates CSV files in: /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/count_data/
# Files: nki_count_data.csv, adhd200_adhd_count_data.csv, cmihbn_adhd_count_data.csv, etc.

# Step 5: Create comprehensive region tables (individual + shared + overlap)
# (Uses the converted CSV files from step 4)
python create_region_tables.py --config ../config.yaml --output_dir ../results/region_tables

# Step 6: Compute cosine similarity between discovery and validation cohorts
# (Uses the converted CSV files from step 4)
python cosine_similarity_analysis.py --analysis_type all

# Step 7: Generate feature maps and network analysis for each dataset
python network_analysis_yeo.py --process_all

# Step 8: Brain behavior analysis for TD, ADHD, ASD
# (Uses the converted CSV files from step 4 - config.yaml points to correct paths)
python comprehensive_brain_behavior_analysis.py --dataset nki_rs_td
python comprehensive_brain_behavior_analysis.py --dataset cmihbn_td
python comprehensive_brain_behavior_analysis.py --dataset adhd200_td
python comprehensive_brain_behavior_analysis.py --dataset adhd200_adhd
python comprehensive_brain_behavior_analysis.py --dataset cmihbn_adhd
python comprehensive_brain_behavior_analysis.py --dataset abide_asd
python comprehensive_brain_behavior_analysis.py --dataset stanford_asd

# Step 9: Create all plots (separate plotting scripts)
# Brain age prediction plots
python plot_brain_age_correlations.py --results_file ../results/brain_age_prediction_results.json

# Brain visualization plots (3D brain surface plots)
python plot_brain_visualization.py --config ../config.yaml

# Network analysis plots
python plot_network_analysis.py --config ../config.yaml

# Brain-behavior analysis plots
python plot_brain_behavior_analysis.py --results_file ../results/brain_behavior_results.json
```

**🔄 Alternative: Training New Models (Optional)**

If you need to train new models instead of using existing ones:

```bash
# Train new models on HCP-Dev (5-fold CV)
python brain_age_prediction.py \
  --hcp_dev_dir /oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/hcp_dev_age_five_fold \
  --output_dir ../results/training

# Then run analysis with newly trained models
python brain_age_prediction.py \
  --config ../config.yaml \
  --use_existing_models \
  --model_dir results/training
```

### 3. Complete Workflow Example

```bash
# Step 1: Train models
python train_brain_age_model.py

# Step 2: Test on external data
python test_external_dataset.py --dataset nki_rs_td

# Step 3: Compute IG scores
python compute_integrated_gradients.py --dataset nki_rs_td --fold 0

# Step 4: Brain-behavior analysis
python brain_behavior_analysis.py --dataset nki_rs_td
```

### 4. Available Datasets

For external testing, IG computation, and brain-behavior analysis:
- `nki_rs_td` - NKI-RS TD cohort
- `adhd200_adhd` - ADHD-200 ADHD cohort  
- `cmihbn_adhd` - CMI-HBN ADHD cohort
- `adhd200_td` - ADHD-200 TD cohort
- `cmihbn_td` - CMI-HBN TD cohort
- `abide_asd` - ABIDE ASD cohort
- `stanford_asd` - Stanford ASD cohort

## 📊 Analysis Modules

### Region Tables Creation (`scripts/create_region_tables.py`)

Creates comprehensive region tables including both individual dataset tables and overlap tables:

**Individual Dataset Tables:**
1. **Individual Dataset Tables**: Top regions for each dataset (e.g., `nki_region_table.csv`, `adhd200_adhd_region_table.csv`)
2. **Shared TD Regions**: Regions shared across TD cohorts (`shared_regions_TD.csv`)
3. **Shared ADHD Regions**: Regions shared across ADHD cohorts (`shared_regions_ADHD.csv`)
4. **Shared ASD Regions**: Regions shared across ASD cohorts (`shared_regions_ASD.csv`)
5. **Overall Shared Regions**: Regions shared across all datasets (`shared_regions_all.csv`)

**Overlap Tables:**
1. **TD Overlap**: Regions shared across TD cohorts (`overlap_regions_TD.csv`)
2. **ADHD Overlap**: Regions shared across ADHD cohorts (`overlap_regions_ADHD.csv`)
3. **ASD Overlap**: Regions shared across ASD cohorts (`overlap_regions_ASD.csv`)

**Table Format:**
- `Brain Regions`: Anatomical names from Gyrus column
- `Subdivision`: Brainnetome subdivision codes (a9l, a8m, a8dl, etc.) from Region Alias column
- `(ID) Region Label`: ROI identifier from (ID) Region Label column
- `Count`: Minimum count across overlapping datasets

**Features:**
- **Individual Dataset Analysis**: Top regions for each dataset with accurate brain region mapping
- **Shared Region Analysis**: Regions appearing across multiple datasets within conditions
- **Overlap Analysis**: Regions appearing in at least 2 datasets within each condition
- **Accurate Brain Region Mapping**: Uses Gyrus column for anatomical names, Region Alias for subdivisions
- **Minimum Count Logic**: Uses minimum count across datasets for shared regions
- **CSV Integration**: Uses converted CSV files from count data processing
- **Sorted by Importance**: Tables sorted by count (highest first)

**Usage:**
```bash
# Create all region tables (individual + shared + overlap)
python create_region_tables.py --config ../config.yaml --output_dir ../results/region_tables

# Output files include:
# Individual: nki_region_table.csv, adhd200_adhd_region_table.csv, etc.
# Shared: shared_regions_TD.csv, shared_regions_ADHD.csv, shared_regions_ASD.csv, shared_regions_all.csv
# Overlap: overlap_regions_TD.csv, overlap_regions_ADHD.csv, overlap_regions_ASD.csv
```

### Comprehensive Cosine Similarity Analysis (`scripts/cosine_similarity_analysis.py`)

Computes cosine similarity for multiple comparison types using count data:

**Analysis Types:**
1. **Discovery vs Validation**: HCP-Dev vs NKI-RS TD, CMI-HBN TD, ADHD-200 TD
2. **Within-Condition**: ADHD200 ADHD vs CMI-HBN ADHD, ABIDE ASD vs Stanford ASD
3. **Pooled Condition**: Pooled ADHD vs Pooled ASD
4. **Cross-Condition**: TD vs ADHD, TD vs ASD

**Features:**
- **Count Data**: Uses attribution count data from IG scores
- **Region Alignment**: Automatically aligns common regions between cohorts
- **Pooled Data**: Creates pooled datasets by averaging across cohorts
- **Statistical Summary**: Computes mean, std, min, max, and range of similarities
- **Structured Analysis**: Organized by comparison type

**Usage:**
```bash
# Option 1: Run all analyses (requires data_dir with all count data files)
python cosine_similarity_analysis.py --analysis_type all --data_dir ../results/count_data/

# Option 2: Run specific analysis types
python cosine_similarity_analysis.py --analysis_type discovery_validation --discovery_csv hcp_dev_count_data.csv --nki_csv nki_rs_td_count_data.csv --cmihbn_csv cmihbn_td_count_data.csv --adhd200_csv adhd200_td_count_data.csv

python cosine_similarity_analysis.py --analysis_type within_condition --data_dir ../results/count_data/
python cosine_similarity_analysis.py --analysis_type pooled_condition --data_dir ../results/count_data/
python cosine_similarity_analysis.py --analysis_type cross_condition --data_dir ../results/count_data/
```

**Expected Count Data Files:**
- `hcp_dev_count_data.csv` (discovery)
- `nki_rs_td_count_data.csv`, `cmihbn_td_count_data.csv`, `adhd200_td_count_data.csv` (TD cohorts)
- `adhd200_adhd_count_data.csv`, `cmihbn_adhd_count_data.csv` (ADHD cohorts)
- `abide_asd_count_data.csv`, `stanford_asd_count_data.csv` (ASD cohorts)

**Output:**
- Individual cosine similarities for each comparison
- Summary statistics (mean, std, min, max, range) for each analysis type
- Pooled data files for cross-condition comparisons
- Comprehensive JSON file with all results organized by analysis type

### Separate Plotting Scripts

The pipeline now includes dedicated plotting scripts for different analysis types:

#### Brain Age Correlation Plots (`scripts/plot_brain_age_correlations.py`)
- **Scatter Plots**: True vs predicted age with correlation metrics
- **Brain Age Gap Distributions**: Histograms of BAG values
- **Group Comparisons**: Box plots comparing BAG between groups
- **Statistical Testing**: T-tests and effect sizes for group differences

#### Brain Visualization Plots (`scripts/plot_brain_visualization.py`)
- **3D Brain Surface Plots**: NIfTI-based brain feature maps
- **Top Features Visualization**: Brain maps for top N important regions
- **Atlas Integration**: Uses BN_Atlas_246_2mm.nii for visualization

#### Network Analysis Plots (`scripts/plot_network_analysis.py`)
- **Polar Bar Plots**: Network-level feature importance (similar to R ggplot2)
- **Network Comparisons**: Polar plots comparing networks across datasets (no more bar plots)
- **Heatmaps**: Network analysis results across all datasets
- **Yeo Atlas Integration**: Groups regions by Yeo 17-network atlas

#### Polar Area Plots (`scripts/create_polar_network_plots.py`)
- **Filled Polar Area Plots**: Radar charts with filled areas and connected data points (like attached image)
- **Individual Polar Plots**: Single dataset polar area plots
- **Comparison Polar Plots**: Side-by-side polar plots for multiple datasets
- **Combined Polar Plots**: Overlaid polar plots for direct comparison
- **Customizable Styling**: Colors, transparency, and grid options

#### Cohort-Specific Brain Age Plots
- **TD Cohorts** (`scripts/plot_brain_age_td_cohorts.py`): Combined scatter plot for all TD cohorts (ABIDE TD, CMI-HBN TD, ADHD200 TD, NKI)
- **ADHD Cohorts** (`scripts/plot_brain_age_adhd_cohorts.py`): Combined scatter plot for all ADHD cohorts (CMI-HBN ADHD, ADHD200 ADHD)
- **ASD Cohorts** (`scripts/plot_brain_age_asd_cohorts.py`): Combined scatter plot for all ASD cohorts (ABIDE ASD, Stanford ASD)
- **Multi-Dataset Panels**: Single panels with different colors/markers for each dataset
- **Overall Statistics**: Combined R², MAE, and correlation across all datasets in each cohort
- **Bias-Corrected Data**: Uses .npz files with bias-corrected predicted and actual ages

#### Brain-Behavior Analysis Plots (`scripts/plot_brain_behavior_analysis.py`)
- **Correlation Matrices**: Heatmaps of brain-behavior correlations
- **PCA Variance Plots**: Explained variance and cumulative variance
- **FDR Correction Plots**: Before/after FDR correction comparisons
- **Behavioral Distributions**: Histograms of behavioral measures

**Usage:**
```bash
# Cohort-specific brain age plots (combined panels)
python plot_brain_age_td_cohorts.py --npz_dir . --output_dir ../results/brain_age_plots
python plot_brain_age_adhd_cohorts.py --npz_dir . --output_dir ../results/brain_age_plots
python plot_brain_age_asd_cohorts.py --npz_dir . --output_dir ../results/brain_age_plots

# Brain age prediction plots (from JSON results)
python plot_brain_age_correlations.py --results_file ../results/brain_age_prediction_results.json

# Brain visualization plots
python plot_brain_visualization.py --config ../config.yaml

# Network analysis plots
python plot_network_analysis.py --config ../config.yaml

# Polar area plots (like attached image)
python create_polar_network_plots.py --network_csv ../results/network_analysis/nki_network_analysis.csv --output_dir ../results/polar_plots
python create_polar_network_plots.py --config ../config.yaml --output_dir ../results/polar_plots --comparison

# Brain-behavior analysis plots
python plot_brain_behavior_analysis.py --results_file ../results/brain_behavior_results.json
```

### Generate Count Data (`scripts/generate_count_data.py`)

Generates count data from Integrated Gradient scores for network analysis:

- **IG Score Processing**: Processes individual IG scores from trained models
- **Percentile Thresholding**: Selects top features based on percentile (default: 50%)
- **Count Data Generation**: Creates count data for network-level analysis
- **Yeo Atlas Integration**: Optional region name mapping using Yeo atlas
- **Multiple Output Formats**: Supports various output configurations

**Required Arguments:**
- `--ig_csv`: Path to IG scores CSV file (generated by compute_integrated_gradients.py)
- `--output`: Output CSV file path for count data

**Optional Arguments:**
- `--percentile`: Percentile threshold for top features (default: 50, range: 0-100)
- `--atlas_file`: Path to Yeo atlas CSV file for region names (optional)
- `--no_absolute`: Don't use absolute values of IG scores (default: use absolute values)

### Network Analysis with Yeo Atlas (`scripts/network_analysis_yeo.py`)

Network-level analysis using Yeo 17-network atlas to group ROIs by brain networks:

- **Yeo Atlas Integration**: Groups ROIs by Yeo 17-network atlas
- **Network Aggregation**: Aggregates attribution scores by brain networks
- **Network-Level Plots**: Creates polar and bar plots for network-level data
- **Statistical Analysis**: Computes mean, std, and count statistics per network
- **Comprehensive Output**: Saves network data and analysis results

**Usage:**
```bash
# Step 1: Generate count data from IG scores
python generate_count_data.py \
  --ig_csv ../results/integrated_gradients/nki_rs_td/nki_rs_td_features_IG_convnet_regressor_trained_on_hcp_dev_fold_0.csv \
  --output ../results/count_data/nki_rs_td_count_data.csv \
  --percentile 50

# Step 2: Run Yeo network analysis
python network_analysis_yeo.py \
  --count_csv ../results/count_data/nki_rs_td_count_data.csv \
  --yeo_atlas /path/to/yeo_atlas.csv
```


### Legacy Network Analysis (`scripts/network_analysis.py`)

Performs network-level analysis using consensus count data and Yeo atlas:

- **Consensus Feature Analysis**: Identifies features consistently important across models
- **Network Mapping**: Maps features to Yeo 17-network atlas
- **Network-Level Scoring**: Aggregates feature importance at network level
- **Visualization**: Creates network and feature importance plots

**Key Functions**:
- `analyze_network_consensus()`: Main analysis function
- `map_features_to_networks()`: Maps features to brain networks
- `compute_network_level_scores()`: Aggregates scores by network

### Brain Age Prediction (`scripts/brain_age_prediction.py`)

Complete brain age prediction pipeline with bias correction:

- **Nested Cross-Validation**: 5-fold outer CV with inner CV for hyperparameter optimization
- **Weights & Biases Integration**: Hyperparameter optimization with W&B logging
- **Hyperparameter Search**: Learning rate, dropout, weight decay, batch size, epochs
- **Bias Correction**: Linear regression-based correction using TD cohorts
- **External Testing**: Testing on multiple external datasets
- **Performance Metrics**: R², MAE, correlation analysis with statistical significance
- **Visualization**: Brain age prediction scatter plots

**Supported Datasets**:
- HCP-Dev (discovery cohort)
- NKI-RS TD, ADHD-200 TD, CMI-HBN TD (for bias correction)
- All external datasets for testing

### Comprehensive Brain-Behavior Analysis (`scripts/comprehensive_brain_behavior_analysis.py`)

Analyzes correlations between **individual IG scores** and behavioral measures across all datasets:

- **IG-Based Analysis**: Uses Integrated Gradient scores for each individual subject
- **Multiple Comparison Correction**: FDR (Benjamini-Hochberg) correction
- **Site-Specific Analysis**: Separate analysis for ADHD200 sites
- **Statistical Testing**: Comprehensive correlation analysis
- **Visualization**: Correlation matrices and summary plots

**Supported Datasets**:
- **NKI-RS TD**: CAARS_36 (IN), CAARS_37 (HY)
- **ADHD-200 ADHD**: HY, IN (by site)
- **CMI-HBN ADHD**: C3SR_HY_T, C3SR_IN_T
- **ADHD-200 TD**: HY, IN (by site)
- **CMI-HBN TD**: C3SR_HY_T, C3SR_IN_T
- **ABIDE ASD**: ADOS Total, Social, Comm
- **Stanford ASD**: SRS Total

**Key Features**:
- **PCA on IG Scores**: Reduces 246 ROIs to 10 principal components before correlation analysis
- **Individual-level IG scores**: 246 ROIs per subject, median across timepoints
- **Correlation Analysis**: Between PCA components and behavioral measures
- **FDR Correction**: Benjamini-Hochberg correction for multiple behavioral metrics
- **Site-specific Analysis**: For multi-site datasets (ADHD-200)
- **Dataset-specific Loading**: ADHD-200 .pklz, CMI-HBN directory, ABIDE filtered sites, Stanford CSV
- **Proper NaN Handling**: Removes subjects with missing behavioral data

**Data Sources**:
- **ADHD-200**: `.pklz` files with NaN handling
- **CMI-HBN**: Directory with `run1` files, motion filtering (mean_fd < 0.5)
- **ABIDE**: Filtered sites (NYU, SDSU, STANFORD, etc.) with `acompcor` preprocessing
- **Stanford**: CSV files with SRS data and duplicate handling

### Feature Comparison (`scripts/feature_comparison.py`)

Compares feature attributions between cohorts:

- **Multiple Selection Modes**: Mean IG or rank-based selection
- **Similarity Metrics**: Jaccard index, Dice coefficient, cosine similarity
- **Statistical Analysis**: Overlap analysis with significance testing
- **Visualization**: Correlation plots and similarity metrics

## 🔧 Utility Modules

### Data Utils (`utils/data_utils.py`)

- `load_finetune_dataset()`: Load training data
- `reshape_data()`: Reshape neuroimaging data
- `remove_nans()`: Clean data with missing values
- `detect_roi_columns()`: Identify ROI columns in DataFrames

### Model Utils (`utils/model_utils.py`)

- `ConvNet`: Convolutional neural network for age prediction
- `train_regressor_w_embedding()`: Model training function
- `evaluate_model_performance()`: Comprehensive model evaluation
- `RMSELoss`: Custom loss function

### Statistical Utils (`utils/statistical_utils.py`)

- `benjamini_hochberg_correction()`: FDR correction
- `multiple_correlation_analysis()`: Multiple correlation testing
- `compute_effect_size()`: Effect size calculations
- `permutation_test()`: Non-parametric testing

### Plotting Utils (`utils/plotting_utils.py`)

- `plot_age_prediction()`: Age prediction scatter plots
- `plot_network_analysis()`: Network-level visualizations
- `plot_feature_importance()`: Feature importance plots
- `plot_correlation_matrix()`: Correlation heatmaps

### Feature Utils (`utils/feature_utils.py`)

- `compute_consensus_features_across_models()`: Consensus analysis
- `map_features_to_networks()`: Network mapping
- `create_feature_consensus_nifti()`: NIfTI file creation
- `analyze_feature_overlap()`: Overlap analysis

## 📈 Output Structure

The pipeline generates organized outputs in the `results/` directory:

```
results/
├── figures/
│   ├── network_analysis.png
│   ├── feature_importance.png
│   └── correlation_matrix.png
├── tables/
│   ├── feature_summary.csv
│   ├── network_summary.csv
│   └── analysis_summary.txt
├── network_analysis/
│   ├── consensus_features_normalized.nii.gz
│   └── network_comparison_across_cohorts.csv
└── brain_behavior/
    ├── cmihbn_td/
    │   └── cmihbn_td_correlations.csv
    └── adhd200_td/
        ├── adhd200_td_overall_correlations.csv
        └── site_specific_results/
```

## 🔬 Statistical Methods

### Multiple Comparison Correction

The pipeline implements several correction methods:

- **Benjamini-Hochberg (FDR)**: Default method for controlling false discovery rate
- **Bonferroni**: Conservative correction for family-wise error rate
- **Holm**: Step-down Bonferroni procedure

### Effect Size Calculations

- **Cohen's d**: Standardized mean difference
- **Hedges' g**: Bias-corrected Cohen's d
- **Bootstrap Confidence Intervals**: Non-parametric confidence intervals

### Correlation Analysis

- **Pearson**: Linear correlations
- **Spearman**: Rank-based correlations
- **Kendall's tau**: Alternative rank correlation

## 🎨 Visualization Standards

All plots follow consistent aesthetic standards:

- **Color Palette**: Professional, colorblind-friendly colors
- **Typography**: Arial font family for publication readiness
- **Resolution**: 300 DPI for high-quality figures
- **Formats**: PNG, PDF, and SVG support
- **Style**: Clean, minimal design with proper axis labels

## 📝 Usage Examples

### Example 1: Network Analysis

```python
from scripts.network_analysis import analyze_network_consensus

results = analyze_network_consensus(
    ig_dir="/path/to/ig_data",
    model_prefix="dev_age_model",
    num_models=5,
    yeo_atlas_path="/path/to/yeo_atlas.csv",
    percentile=95.0,
    output_dir="results/network_analysis"
)
```

### Example 2: Brain-Behavior Correlation

```python
from scripts.brain_behavior_correlation import analyze_cmihbn_td_cohort

results = analyze_cmihbn_td_cohort(
    data_dir="/path/to/data",
    output_dir="results/brain_behavior/cmihbn_td"
)
```

### Example 3: Feature Comparison

```python
from scripts.feature_comparison import FeatureComparison

comparator = FeatureComparison(top_fraction=0.5, selection_mode="mean_IG")
results = comparator.compare_cohorts(
    file_a="cohort1.csv",
    file_b="cohort2.csv",
    output_dir="results/comparison"
)
```

## 🧪 Testing

Run tests to verify functionality:

```bash
# Test utility functions
python -m pytest tests/test_utils.py

# Test individual modules
python -c "from utils import data_utils, model_utils, plotting_utils; print('All modules imported successfully')"
```

## 📚 Dependencies

### Core Dependencies

- **numpy**: Numerical computing
- **pandas**: Data manipulation
- **scipy**: Scientific computing
- **scikit-learn**: Machine learning
- **torch**: Deep learning framework
- **matplotlib**: Plotting
- **seaborn**: Statistical visualization

### Neuroimaging

- **nilearn**: Neuroimaging analysis
- **nibabel**: Neuroimaging file I/O

### Statistical Analysis

- **statsmodels**: Statistical models
- **captum**: Feature attribution

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Follow PEP8 standards**: Use `black` and `flake8` for formatting
4. **Add tests**: Ensure new code is tested
5. **Update documentation**: Add docstrings and update README
6. **Submit a pull request**

### Code Style

- Follow PEP8 guidelines
- Use type hints for function parameters and returns
- Write comprehensive docstrings
- Use meaningful variable names
- Keep functions focused and modular

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- SCSNL Age Prediction Team
- Contributors to the original analysis scripts
- Open-source neuroimaging and machine learning communities

## 📞 Support

For questions, issues, or contributions:

1. **Check the documentation**: Review this README and docstrings
2. **Search existing issues**: Look for similar problems
3. **Create a new issue**: Provide detailed description and error messages
4. **Contact the team**: [Add contact information]

## 🔄 Version History

- **v1.0.0**: Initial refactored release with modular architecture
- **v0.x.x**: Original analysis scripts (legacy)

---

**Note**: This is a research tool. Please ensure you have appropriate permissions and ethical approvals for your data and analyses.
