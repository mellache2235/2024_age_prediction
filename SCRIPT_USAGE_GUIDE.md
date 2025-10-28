# Script Usage Guide

This guide provides clear instructions on how to specify datasets and available options for each script in the age prediction pipeline.

## 📊 Available Datasets

All scripts that work with datasets support these options:

| Dataset Name | Description |
|--------------|-------------|
| `nki_rs_td` | NKI-RS TD cohort |
| `adhd200_adhd` | ADHD-200 ADHD cohort |
| `cmihbn_adhd` | CMI-HBN ADHD cohort |
| `adhd200_td` | ADHD-200 TD cohort |
| `cmihbn_td` | CMI-HBN TD cohort |
| `abide_asd` | ABIDE ASD cohort |
| `stanford_asd` | Stanford ASD cohort |

## 🧠 Brain Age Prediction

### `scripts/brain_age_prediction.py`

**Purpose**: Train brain age models and test on external datasets

**Usage**:
```bash
# Complete brain age prediction analysis (all datasets)
python scripts/brain_age_prediction.py --config config.yaml

# Train models only
python scripts/brain_age_prediction.py --hcp_dev_dir /path/to/hcp_dev_data --output_dir results/training
```

**Options**:
- `--config`: Path to configuration file
- `--hcp_dev_dir`: Directory containing HCP-Dev training data
- `--output_dir`: Output directory for results (default: results/brain_age_prediction)
- `--num_folds`: Number of cross-validation folds (default: 5)

## 🔍 Integrated Gradients Computation

### `scripts/compute_integrated_gradients.py`

**Purpose**: Compute Integrated Gradients for brain age models

**Usage**:
```bash
# Compute IG for specific dataset
python scripts/compute_integrated_gradients.py --dataset nki_rs_td --fold 0

# With custom percentile threshold
python scripts/compute_integrated_gradients.py --dataset adhd200_adhd --fold 0 --percentile 80
```

**Options**:
- `--dataset`: Dataset to compute IG for (required, see available options above)
- `--fold`: HCP-Dev model fold to use (0-4, default: 0)
- `--model_dir`: Directory containing trained models (default: results/brain_age_models)
- `--roi_labels`: Path to ROI labels file
- `--percentile`: Percentile threshold for top features (default: 50, range: 0-100)

**Output**: Creates IG scores CSV and count data CSV files

## 📈 Count Data Generation

### `scripts/generate_count_data.py`

**Purpose**: Generate count data from Integrated Gradients scores

**Usage**:
```bash
# Generate count data from IG scores
python scripts/generate_count_data.py \
  --ig_csv results/integrated_gradients/nki_rs_td/nki_rs_td_features_IG_convnet_regressor_trained_on_hcp_dev_fold_0.csv \
  --output results/count_data/nki_rs_td_count_data.csv \
  --percentile 50

# With Yeo atlas region names
python scripts/generate_count_data.py \
  --ig_csv results/integrated_gradients/adhd200_adhd/adhd200_adhd_features_IG_convnet_regressor_trained_on_hcp_dev_fold_0.csv \
  --output results/count_data/adhd200_adhd_count_data.csv \
  --atlas_file /path/to/yeo_atlas.csv \
  --percentile 80
```

**Options**:
- `--ig_csv`: Path to IG scores CSV file (required)
- `--percentile`: Percentile threshold for top features (default: 50, range: 0-100)
- `--output`: Output CSV file path for count data (required)
- `--atlas_file`: Path to Yeo atlas CSV file for region names (optional)
- `--no_absolute`: Don't use absolute values of IG scores (default: use absolute values)

## 📊 Region Tables Creation

### `scripts/create_region_tables.py`

**Purpose**: Create comprehensive tables for regions of importance

**Usage**:
```bash
# Create all region tables
python scripts/create_region_tables.py

# With custom parameters
python scripts/create_region_tables.py --top_n 100 --output_dir results/custom_tables
```

**Options**:
- `--config`: Path to configuration file (default: config.yaml)
- `--output_dir`: Output directory for tables (default: results/region_tables)
- `--top_n`: Number of top regions to include (default: 50)

**Output**: Creates individual and shared region tables for TD, ADHD, ASD cohorts

## 🧠 Brain-Behavior Analysis

### Enhanced Brain-Behavior Scripts (Pre-configured, Zero Arguments)

**Purpose**: PCA-based brain-behavior correlation analysis using IG scores to predict behavioral measures. All paths pre-configured - just run!

#### TD Cohorts

**`scripts/run_nki_brain_behavior_enhanced.py`**
- **Dataset**: NKI-RS TD
- **Behavioral Measures**: CAARS (Conners Adult ADHD Rating Scale)
- **Usage**: `python scripts/run_nki_brain_behavior_enhanced.py`

**`scripts/run_adhd200_brain_behavior_enhanced.py`**
- **Dataset**: ADHD-200 TD Subset (NYU)
- **Behavioral Measures**: Inattentive, Hyperactive/Impulsive scores
- **Usage**: `python scripts/run_adhd200_brain_behavior_enhanced.py`

**`scripts/run_cmihbn_brain_behavior_enhanced.py`**
- **Dataset**: CMI-HBN TD
- **Behavioral Measures**: C3SR (Conners 3 Self-Report)
- **Usage**: `python scripts/run_cmihbn_brain_behavior_enhanced.py`

#### ADHD Cohorts

**`scripts/run_adhd200_adhd_brain_behavior_enhanced.py`**
- **Dataset**: ADHD-200 ADHD
- **Behavioral Measures**: Inattentive, Hyperactive/Impulsive scores
- **Usage**: `python scripts/run_adhd200_adhd_brain_behavior_enhanced.py`

**`scripts/run_cmihbn_adhd_brain_behavior_enhanced.py`**
- **Dataset**: CMI-HBN ADHD
- **Behavioral Measures**: C3SR (Conners 3 Self-Report)
- **Special Features**: Uses clinical diagnosis CSV (Diagnosis_ClinicianConsensus.csv) to identify ADHD subjects
- **Usage**: `python scripts/run_cmihbn_adhd_brain_behavior_enhanced.py`

#### ASD Cohorts

**`scripts/run_stanford_asd_brain_behavior_enhanced.py`**
- **Dataset**: Stanford ASD
- **Behavioral Measures**: SRS Total Score T-Score (Social Responsiveness Scale)
- **Usage**: `python scripts/run_stanford_asd_brain_behavior_enhanced.py`

**`scripts/run_abide_asd_brain_behavior_enhanced.py`**
- **Dataset**: ABIDE ASD
- **Behavioral Measures**: ADOS (Autism Diagnostic Observation Schedule)
  - `ados_total` - ADOS Total Score
  - `ados_social` - ADOS Social Score
  - `ados_comm` - ADOS Communication Score
- **Usage**: `python scripts/run_abide_asd_brain_behavior_enhanced.py`

**All scripts output**:
- Elbow plot (optimal PC selection)
- Scatter plots (6×6 inch, predicted vs actual behavioral scores) - PNG + TIFF + AI formats
- Plot format: Uses "r =" for Spearman correlation (not "ρ =")
- Linear regression results (Spearman r, p-values, R²)
- PC importance rankings
- PC loadings (top brain regions per PC)

### `scripts/comprehensive_brain_behavior_analysis.py`

**Purpose**: Comprehensive brain-behavior correlation analysis with PCA and FDR correction

**Usage**:
```bash
# Run comprehensive analysis for all datasets
python scripts/comprehensive_brain_behavior_analysis.py \
  --config config.yaml \
  --output_dir results/brain_behavior

# Analyze specific dataset
python scripts/comprehensive_brain_behavior_analysis.py \
  --dataset nki_rs_td \
  --ig_dir results/integrated_gradients/nki_rs_td \
  --behavioral_data /path/to/nki_behavioral_data.csv \
  --output_dir results/nki_rs_td_behavior
```

**Options**:
- `--config`: Path to configuration file (for comprehensive analysis)
- `--dataset`: Specific dataset to analyze (see available options above)
- `--ig_dir`: Directory containing IG scores CSV files
- `--behavioral_data`: Path to behavioral data file
- `--output_dir`: Output directory for results (default: results/brain_behavior)

## 📊 Network Analysis

### `scripts/network_analysis_yeo.py`

**Purpose**: Network-level analysis using Yeo atlas grouping

**Usage**:
```bash
# Process all datasets
python scripts/network_analysis_yeo.py --process_all

# Process specific dataset
python scripts/network_analysis_yeo.py --dataset nki_rs_td
```

**Options**:
- `--dataset`: Specific dataset to process
- `--process_all`: Process all datasets defined in config
- `--config`: Path to configuration file (default: config.yaml)
- `--output_dir`: Output directory for results

## 🔗 Cosine Similarity Analysis

### `scripts/cosine_similarity_analysis.py`

**Purpose**: Compute cosine similarity between discovery and validation cohorts

**Usage**:
```bash
# Run all analyses
python scripts/cosine_similarity_analysis.py --analysis_type all --data_dir results/count_data/

# Run specific analysis types
python scripts/cosine_similarity_analysis.py --analysis_type discovery_vs_validation --data_dir results/count_data/
python scripts/cosine_similarity_analysis.py --analysis_type within_condition --data_dir results/count_data/
python scripts/cosine_similarity_analysis.py --analysis_type pooled_condition --data_dir results/count_data/
python scripts/cosine_similarity_analysis.py --analysis_type cross_condition --data_dir results/count_data/
```

**Options**:
- `--analysis_type`: Type of analysis to run (all, discovery_vs_validation, within_condition, pooled_condition, cross_condition)
- `--data_dir`: Directory containing count data CSV files

## 📊 Plotting Scripts

### Brain Age Correlation Plots

#### `scripts/plot_brain_age_correlations.py`

**Purpose**: Create brain age correlation plots and statistical comparisons

**Usage**:
```bash
# Create all brain age correlation plots
python scripts/plot_brain_age_correlations.py \
  --results_file results/brain_age_prediction_results.json \
  --output_dir results/figures/brain_age_correlations
```

**Options**:
- `--results_file`: Path to brain age prediction results JSON file (required)
- `--output_dir`: Output directory for plots (default: results/figures/brain_age_correlations)

### Brain Visualization Plots

#### `scripts/plot_brain_visualization.py`

**Purpose**: Create 3D brain surface plots and NIfTI visualizations

**Usage**:
```bash
# Create brain visualizations for all datasets
python scripts/plot_brain_visualization.py --config config.yaml

# With custom parameters
python scripts/plot_brain_visualization.py \
  --config config.yaml \
  --output_dir custom_brain_plots/ \
  --top_n 100
```

**Options**:
- `--config`: Path to configuration file (default: config.yaml)
- `--output_dir`: Output directory for plots (default: results/figures/brain_visualization)
- `--top_n`: Number of top features to visualize (default: 50)

### Network Analysis Plots

#### `scripts/plot_network_analysis.py`

**Purpose**: Create network analysis plots using Yeo atlas grouping

**Usage**:
```bash
# Create network analysis plots for all datasets
python scripts/plot_network_analysis.py --config config.yaml
```

**Options**:
- `--config`: Path to configuration file (default: config.yaml)
- `--output_dir`: Output directory for plots (default: results/figures/network_analysis)

### Brain-Behavior Analysis Plots

#### `scripts/plot_brain_behavior_analysis.py`

**Purpose**: Create brain-behavior correlation analysis plots

**Usage**:
```bash
# Create brain-behavior analysis plots
python scripts/plot_brain_behavior_analysis.py \
  --results_file results/brain_behavior_results.json \
  --output_dir results/figures/brain_behavior_analysis
```

**Options**:
- `--results_file`: Path to brain-behavior analysis results JSON file (required)
- `--output_dir`: Output directory for plots (default: results/figures/brain_behavior_analysis)

## 🔧 Getting Help

To see detailed help for any script, use the `--help` flag:

```bash
python scripts/compute_integrated_gradients.py --help
python scripts/generate_count_data.py --help
python scripts/plot_brain_age_correlations.py --help
```

## 📁 Expected File Structure

After running the pipeline, you should have this structure:

```
results/
├── brain_age_prediction/
│   ├── brain_age_prediction_results.json
│   └── comprehensive_brain_age_analysis.json
├── integrated_gradients/
│   ├── nki_rs_td/
│   ├── adhd200_adhd/
│   └── ...
├── count_data/
│   ├── nki_rs_td_count_data.csv
│   ├── adhd200_adhd_count_data.csv
│   └── ...
├── region_tables/
│   ├── nki_rs_td_top_regions.csv
│   ├── shared_td_regions.csv
│   └── ...
├── network_analysis/
│   ├── nki_rs_td_network_analysis.csv
│   └── ...
├── brain_behavior/
│   └── comprehensive_brain_behavior_results.json
└── figures/
    ├── brain_age_correlations/
    ├── brain_visualization/
    ├── network_analysis/
    └── brain_behavior_analysis/
```
