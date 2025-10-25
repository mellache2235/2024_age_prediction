# Age Prediction Analysis Pipeline

A comprehensive pipeline for brain age prediction analysis using pre-trained models and existing data files. This repository provides tools for feature attribution, network-level analysis, brain-behavior correlations, and statistical analysis.

## 🎯 **Quick Start - Complete Workflow**

**Run the complete pipeline with these commands on HPC:**

```bash
# Navigate to scripts directory
cd /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/scripts/

# Step 1: Convert Excel count data to CSV format
# Converts Excel files from original repository to CSV files in test directory
python convert_count_data.py

# Step 2: Run network analysis on individual datasets
# Creates radar plots for each dataset using count data
python network_analysis_yeo.py --process_all

# Step 3: Run shared network analysis across cohorts
# Creates radar plots using minimum overlap count for TD, ADHD, and ASD
python network_analysis_yeo.py --process_shared

# Step 4: Create region tables (individual + shared + overlap)
# Generates CSV tables with brain regions and counts
python create_region_tables.py \
  --config /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/config.yaml \
  --output_dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/region_tables

# Step 5: Generate brain age prediction plots
# Note: .npz files should be in results/brain_age_predictions/npz_files/
# TD Cohorts (2x2 layout: HCP-Dev, NKI, CMI-HBN TD, ADHD200 TD)
python plot_brain_age_td_cohorts.py \
  --output_dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_age_plots

# ADHD Cohorts (1x2 layout: CMI-HBN ADHD, ADHD200 ADHD)
python plot_brain_age_adhd_cohorts.py \
  --output_dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_age_plots

# ASD Cohorts (1x2 layout: ABIDE ASD, Stanford ASD)
python plot_brain_age_asd_cohorts.py \
  --output_dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_age_plots

# Step 6 (Optional): Run brain-behavior correlation analysis
python comprehensive_brain_behavior_analysis.py --dataset nki_rs_td
python comprehensive_brain_behavior_analysis.py --dataset adhd200_adhd
python comprehensive_brain_behavior_analysis.py --dataset abide_asd
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
- **Individual Tables**: `{dataset_name}_region_table.csv` - One per dataset
- **Shared Tables**: 
  - `shared_regions_TD.csv` - Shared across TD cohorts
  - `shared_regions_ADHD.csv` - Shared across ADHD cohorts
  - `shared_regions_ASD.csv` - Shared across ASD cohorts
- **Overlap Tables**:
  - `overlap_regions_TD.csv` - Minimum count overlap for TD
  - `overlap_regions_ADHD.csv` - Minimum count overlap for ADHD
  - `overlap_regions_ASD.csv` - Minimum count overlap for ASD
- **Format**: CSV with columns: Brain Regions, Subdivision, (ID) Region Label, Count

### **Step 5: Brain Age Plots**
- **Location**: `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_age_plots/`
- **Files**:
  - `td_cohorts_combined_scatter.png` - 2x2 subplot (4 core TD datasets)
  - `adhd_cohorts_combined_scatter.png` - 1x2 subplot (2 ADHD datasets)
  - `asd_cohorts_combined_scatter.png` - 1x2 subplot (2 ASD datasets)
- **Input**: .npz files in `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_age_predictions/npz_files/`

### **Step 6: Brain-Behavior Analysis (Optional)**
- **Location**: `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior_analysis/`
- **Files**: Correlation results with FDR correction, scatter plots, correlation matrices

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
- **Styling**: Light blue fill, dark blue lines
- **Network Mapping**: Yeo 17-network atlas
- **Individual Analysis**: Uses count data for each dataset separately
- **Shared Analysis**: Uses minimum overlap count across cohorts
- **Format**: PNG only

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