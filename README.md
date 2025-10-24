# Age Prediction Analysis Pipeline

A comprehensive, modular pipeline for brain age prediction analysis using deep learning and neuroimaging data. This repository provides tools for feature attribution, network-level analysis, brain-behavior correlations, and statistical analysis with proper multiple comparison correction.

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
│   └── feature_utils.py            # Feature attribution and analysis
│
├── scripts/                         # Analysis scripts
│   ├── network_analysis.py         # Network-level analysis
│   ├── brain_behavior_correlation.py # Brain-behavior correlations
│   ├── feature_comparison.py       # Feature comparison between cohorts
│   └── [legacy scripts]            # Original scripts (to be refactored)
│
├── results/                         # Output directory
│   ├── figures/                    # Generated plots
│   ├── tables/                     # Statistical tables
│   ├── network_analysis/           # Network analysis results
│   └── brain_behavior/             # Brain-behavior results
│
└── [data directories]              # Data files (not in repo)
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for model training)

### Setup

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

## 🚀 Quick Start

### 1. Configuration

Edit `config.yaml` to specify your data paths and analysis parameters:

```yaml
# Network Analysis Configuration
network_analysis:
  ig_dir: "/path/to/integrated_gradients"
  model_prefix: "dev_age_model"
  num_models: 5
  yeo_atlas_path: "/path/to/yeo_atlas.csv"
  percentile: 95.0

# Brain-Behavior Analysis
brain_behavior:
  data_dir: "/path/to/behavioral_data"
  analyze_cmihbn_td: true
  analyze_adhd200_td: true
  correction_method: "fdr_bh"
```

### 2. Run Complete Pipeline

```bash
# Run all analyses in correct order (simplified - no command line arguments needed)
python main.py
```

**Analysis Order:**
1. **Brain Age Prediction** → Train models and generate predictions
2. **Feature Attribution** → Compute Integrated Gradients (IG) for each individual
3. **Brain Visualization** → Create 3D brain surface plots from IG scores
4. **Network Analysis** → Network-level analysis using consensus IG data
5. **Brain-Behavior Analysis** → Correlate individual IG scores with behavioral measures

### 3. Individual Scripts

```bash
# Brain age prediction with nested CV
python scripts/brain_age_prediction.py

# Brain-behavior correlation with PCA
python scripts/comprehensive_brain_behavior_analysis.py

# Network analysis
python scripts/network_analysis.py

# Feature comparison
python scripts/feature_comparison.py

# Create region tables
python scripts/create_region_tables.py
```

## 📊 Analysis Modules

### Network Analysis (`scripts/network_analysis.py`)

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
- **CMI-HBN ADHD**: C3SR, C3SR_HY_T
- **ADHD-200 TD**: HY, IN (by site)
- **CMI-HBN TD**: C3SR, C3SR_HY_T
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
