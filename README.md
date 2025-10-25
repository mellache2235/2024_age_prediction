# Age Prediction Analysis Pipeline

A comprehensive pipeline for brain age prediction analysis using pre-trained models and existing data files. This repository provides tools for feature attribution, network-level analysis, brain-behavior correlations, and statistical analysis.

## 🎯 **Quick Start**

**Run the complete pipeline with these commands:**

```bash
# Navigate to scripts directory
cd scripts/

# 1. Convert Excel count data to CSV format
python convert_count_data.py

# 2. Create region tables (individual + shared + overlap)
python create_region_tables.py --config ../config.yaml --output_dir ../results/region_tables

# 3. Generate brain age prediction plots (subplot format)
python plot_brain_age_td_cohorts.py --npz_dir . --output_dir ../results/brain_age_plots
python plot_brain_age_adhd_cohorts.py --npz_dir . --output_dir ../results/brain_age_plots
python plot_brain_age_asd_cohorts.py --npz_dir . --output_dir ../results/brain_age_plots

# 4. Run network analysis (individual datasets)
python network_analysis_yeo.py --config ../config.yaml

# 5. Run shared network analysis (across cohorts)
python network_analysis_yeo.py --process_shared

# 6. Run brain-behavior analysis
python comprehensive_brain_behavior_analysis.py --dataset nki_rs_td
python comprehensive_brain_behavior_analysis.py --dataset adhd200_adhd
python comprehensive_brain_behavior_analysis.py --dataset abide_asd
```

## 📁 **Output Files**

### **Brain Age Plots**
- **TD Cohorts**: 2x3 subplot layout (6 datasets) - `.png`, `.pdf`, `.svg`
- **ADHD Cohorts**: 1x2 subplot layout (2 datasets) - `.png`, `.pdf`, `.svg`
- **ASD Cohorts**: 1x2 subplot layout (2 datasets) - `.png`, `.pdf`, `.svg`

### **Region Tables**
- **Individual**: One table per dataset
- **Shared**: Combined tables for TD, ADHD, ASD cohorts
- **Overlap**: Minimum count tables for shared regions

### **Network Analysis**
- **Individual**: Radar charts for each dataset
- **Shared**: Radar charts for TD, ADHD, ASD cohorts
- **CSV Results**: Network-level aggregated data

### **Brain-Behavior Analysis**
- **Correlation Results**: FDR-corrected statistics
- **Plots**: Scatter plots and correlation matrices

## 🔧 **Configuration**

The pipeline uses `config.yaml` for all file paths and parameters:

```yaml
# Key paths (automatically configured for HPC)
network_analysis:
  count_data:
    dev: "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/figures/dev/ig_files/top_50_consensus_features_hcp_dev_aging.xlsx"
    nki: "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/figures/nki/ig_files/top_50_consensus_features_nki_cog_dev_aging.xlsx"
    # ... other datasets
  yeo_atlas_path: "/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/feature_attribution/csv_files/subregion_func_network_Yeo_updated_yz.csv"
  roi_labels_path: "/oak/stanford/groups/menon/projects/cdla/2021_hcp_earlypsychosis/scripts/restfmri/classify/CNN1dPyTorch/brainnetome_roi_labels.txt"
```

## 📊 **Available Datasets**

### **TD Cohorts**
- `dev` - HCP-Dev
- `nki` - NKI
- `adhd200_td` - ADHD200 TD
- `cmihbn_td` - CMI-HBN TD
- `abide_td` - ABIDE TD
- `stanford_td` - Stanford TD

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
- **Formats**: PNG, PDF, SVG (for Affinity Designer editing)

### **Network Analysis Plots**
- **Radar Charts**: Polar area plots with filled regions
- **Consistent Styling**: Light blue fill, dark blue lines
- **Network Mapping**: Yeo 17-network atlas
- **Shared Analysis**: Minimum count approach across cohorts

## 📂 **Repository Structure**

```
2024_age_prediction/
├── config.yaml                    # Configuration file
├── scripts/                       # Analysis scripts
│   ├── convert_count_data.py     # Convert Excel to CSV
│   ├── create_region_tables.py   # Create region tables
│   ├── plot_brain_age_*_cohorts.py # Brain age plots
│   ├── network_analysis_yeo.py   # Network analysis
│   └── comprehensive_brain_behavior_analysis.py # Brain-behavior
├── utils/                         # Utility modules
│   ├── plotting_utils.py         # Plotting functions
│   ├── count_data_utils.py       # Data processing
│   └── statistical_utils.py      # Statistical analysis
└── results/                       # Output directory
    ├── brain_age_plots/          # Brain age scatter plots
    ├── region_tables/            # Region importance tables
    ├── network_analysis_yeo/     # Network analysis results
    └── count_data/               # Converted CSV files
```

## 🚀 **Key Features**

- **Modular Design**: Clean, organized scripts with reusable utilities
- **Publication-Ready Plots**: Standardized visualization with consistent aesthetics
- **Multiple Output Formats**: PNG, PDF, SVG for different use cases
- **Statistical Rigor**: FDR correction for multiple comparisons
- **Network Analysis**: Yeo atlas-based network-level analysis
- **Comprehensive Documentation**: Detailed docstrings and usage examples

## 📝 **Notes**

- **HPC Environment**: Scripts are configured for Stanford HPC paths
- **File Formats**: Supports both Excel and CSV input files
- **Error Handling**: Graceful handling of missing files with clear logging
- **Scalable**: Easy to add new datasets or modify analysis parameters

## 🔍 **Troubleshooting**

### **Missing CSV Files**
If you see warnings about missing CSV files, run:
```bash
python convert_count_data.py
```

### **Missing Excel Files**
If Excel files are missing, check the paths in `config.yaml` and ensure files exist at the specified locations.

### **Network Plot Differences**
If shared and individual network plots look different, it's likely due to missing CSV files for some datasets. Check the logging output to see which datasets are being processed.

---

**Ready to run!** All scripts are configured and tested. Simply follow the Quick Start commands above to generate all results and plots.