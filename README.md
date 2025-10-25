# Age Prediction Analysis Pipeline

A comprehensive pipeline for brain age prediction analysis using pre-trained models and existing data files. This repository provides tools for feature attribution, network-level analysis, brain-behavior correlations, and statistical analysis.

## 🎯 **Quick Start**

**Run the complete pipeline with these commands on HPC:**

```bash
# Navigate to scripts directory
cd /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/scripts/

# 1. Convert Excel count data to CSV format
python convert_count_data.py

# 2. Create region tables (individual + shared + overlap)
python create_region_tables.py \
  --config /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/config.yaml \
  --output_dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/region_tables

# 3. Generate brain age prediction plots (subplot format)
# Note: .npz files should be in results/brain_age_predictions/npz_files/
python plot_brain_age_td_cohorts.py \
  --output_dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_age_plots

python plot_brain_age_adhd_cohorts.py \
  --output_dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_age_plots

python plot_brain_age_asd_cohorts.py \
  --output_dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_age_plots

# 4. Run network analysis (individual datasets)
python network_analysis_yeo.py \
  --config /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/config.yaml

# 5. Run shared network analysis (across cohorts)
python network_analysis_yeo.py --process_shared

# 6. Run brain-behavior analysis (optional)
python comprehensive_brain_behavior_analysis.py --dataset nki_rs_td
python comprehensive_brain_behavior_analysis.py --dataset adhd200_adhd
python comprehensive_brain_behavior_analysis.py --dataset abide_asd
```

## 📁 **Output Files**

All results are saved to: `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/`

### **Brain Age Plots**
- **TD Cohorts**: 2x2 subplot layout (4 core datasets) - `.png`
  - HCP-Dev, NKI, CMI-HBN TD, ADHD200 TD
- **ADHD Cohorts**: 1x2 subplot layout (2 datasets) - `.png`
  - CMI-HBN ADHD, ADHD200 ADHD
- **ASD Cohorts**: 1x2 subplot layout (2 datasets) - `.png`
  - ABIDE ASD, Stanford ASD
- **Location**: `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_age_plots/`
- **Input Files**: .npz files in `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_age_predictions/npz_files/`

### **Region Tables**
- **Individual**: One table per dataset (CSV format)
- **Shared**: Combined tables for TD, ADHD, ASD cohorts
- **Overlap**: Minimum count tables for shared regions
- **Location**: `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/region_tables/`

### **Network Analysis**
- **Individual**: Radar charts for each dataset (PNG format)
- **Shared**: Radar charts for TD, ADHD, ASD cohorts (minimum overlap count)
- **CSV Results**: Network-level aggregated data
- **Location**: `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/network_analysis_yeo/`
  - Individual datasets: `network_analysis_yeo/{dataset_name}/`
  - Shared TD: `network_analysis_yeo/shared_TD/`
  - Shared ADHD: `network_analysis_yeo/shared_ADHD/`
  - Shared ASD: `network_analysis_yeo/shared_ASD/`

### **Brain-Behavior Analysis**
- **Correlation Results**: FDR-corrected statistics
- **Plots**: Scatter plots and correlation matrices
- **Location**: `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior_analysis/`

## 🔧 **Configuration**

The pipeline uses `config.yaml` for all file paths and parameters:

**Config file location**: `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/config.yaml`

Key paths configured:
```yaml
network_analysis:
  count_data:
    dev: "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/figures/dev/ig_files/top_50_consensus_features_hcp_dev_aging.xlsx"
    nki: "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/figures/nki/ig_files/top_50_consensus_features_nki_cog_dev_aging.xlsx"
    # ... other datasets
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
- **TD Layout**: 2x2 (4 core datasets only)
- **ADHD/ASD Layout**: 1x2 (2 datasets each)

### **Network Analysis Plots**
- **Type**: Radar charts (polar area plots)
- **Styling**: Light blue fill, dark blue lines
- **Network Mapping**: Yeo 17-network atlas
- **Individual Analysis**: Uses count data for each dataset
- **Shared Analysis**: Uses minimum overlap count across cohorts
- **Format**: PNG only

## 📂 **Repository Structure**

```
2024_age_prediction_test/
├── config.yaml                    # Configuration file
├── scripts/                       # Analysis scripts
│   ├── convert_count_data.py     # Convert Excel to CSV
│   ├── create_region_tables.py   # Create region tables
│   ├── plot_brain_age_td_cohorts.py      # TD brain age plots (2x2)
│   ├── plot_brain_age_adhd_cohorts.py    # ADHD brain age plots (1x2)
│   ├── plot_brain_age_asd_cohorts.py     # ASD brain age plots (1x2)
│   ├── network_analysis_yeo.py   # Network analysis (individual + shared)
│   ├── create_polar_network_plots.py     # Radar chart plotting functions
│   ├── plot_network_analysis.py  # Network comparison plots
│   ├── comprehensive_brain_behavior_analysis.py # Brain-behavior correlations
│   ├── cosine_similarity_analysis.py     # Similarity analysis
│   └── feature_comparison.py     # Feature comparison tools
├── utils/                         # Utility modules
│   ├── plotting_utils.py         # Plotting functions
│   ├── count_data_utils.py       # Data processing
│   └── statistical_utils.py      # Statistical analysis
└── results/                       # Output directory
    ├── brain_age_predictions/    # Brain age prediction data
    │   └── npz_files/           # .npz files with predicted/actual ages
    ├── brain_age_plots/          # Brain age scatter plots (PNG)
    ├── region_tables/            # Region importance tables (CSV)
    ├── network_analysis_yeo/     # Network analysis results
    │   ├── {dataset_name}/      # Individual dataset results
    │   ├── shared_TD/           # Shared TD analysis
    │   ├── shared_ADHD/         # Shared ADHD analysis
    │   └── shared_ASD/          # Shared ASD analysis
    ├── count_data/               # Converted CSV files
    └── brain_behavior_analysis/  # Brain-behavior correlation results
```

## 🚀 **Key Features**

- **Modular Design**: Clean, organized scripts with reusable utilities
- **Publication-Ready Plots**: Standardized visualization with consistent aesthetics
- **PNG Output**: Single format for all plots (easy to use and share)
- **Statistical Rigor**: FDR correction for multiple comparisons
- **Network Analysis**: Yeo atlas-based network-level analysis with radar charts
- **Shared Analysis**: Minimum overlap count approach for cross-cohort comparisons
- **Organized Structure**: Clear directory organization for all results

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

### **Network Plot Differences**
If shared and individual network plots look different:
- Check CSV files exist: `/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/count_data/`
- Review logging output to see which datasets are being processed
- Ensure all 4 TD datasets have CSV files for shared analysis

### **Import Errors**
If you get module import errors:
- Ensure you're running from the `scripts/` directory
- Check that `utils/` directory is in the parent directory
- Verify Python path includes the repository root

---

**Ready to run!** All scripts are configured with full HPC paths. Simply copy and paste the commands from the Quick Start section.