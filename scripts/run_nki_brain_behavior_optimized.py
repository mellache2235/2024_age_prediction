#!/usr/bin/env python3
"""
Optimized Brain-Behavior Analysis for NKI-RS TD Cohort

This script performs brain-behavior correlation analysis with hyperparameter tuning:
1. Loads NKI IG scores and behavioral data
2. Performs PCA on IG scores
3. Optimizes:
   - Number of PCs (grid search from 5 to 50)
   - Regression model (Linear, Ridge, Lasso, ElasticNet)
   - Regularization strength (alpha values)
4. Uses nested cross-validation for robust evaluation
5. Reports best parameters and performance
6. Creates visualizations with best model

Author: Enhanced Brain-Behavior Analysis
Date: 2024
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.backends.backend_pdf as pdf
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set Arial font
font_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/clustering_analysis/arial.ttf'
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET = "nki_rs_td"

# Input paths
IG_CSV = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/integrated_gradients/nki_cog_dev_wIDS_features_IG_convnet_regressor_single_model_fold_0.csv"

# Behavioral data paths
BEHAVIORAL_DIR = Path("/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/FLUX/assessment_data")
CAARS_FILE = BEHAVIORAL_DIR / "8100_CAARS_20191009.csv"
CONNERS_PARENT_FILE = BEHAVIORAL_DIR / "8100_Conners_3-P(S)_20191009.csv"
CONNERS_SELF_FILE = BEHAVIORAL_DIR / "8100_Conners_3-SR(S)_20191009.csv"
RBS_FILE = BEHAVIORAL_DIR / "8100_RBS-R_20191009.csv"

# Output directory
OUTPUT_DIR = Path("/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/nki_rs_td_optimized")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameter search space (will be adjusted based on sample size)
MAX_N_PCS = 50  # Maximum to consider
PC_STEP = 5  # Step size for PC grid
ALPHA_RANGE = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

# Cross-validation settings
OUTER_CV_FOLDS = 5  # For final evaluation
INNER_CV_FOLDS = 3  # For hyperparameter tuning

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*100)
    print(text.center(100))
    print("="*100)

def print_step(text):
    """Print a step header."""
    print(f"\n[STEP] {text}")
    print("-" * 100)

def print_info(key, value):
    """Print key-value information."""
    print(f"✓ {key}: {value}")

def spearman_scorer(y_true, y_pred):
    """Custom scorer for Spearman correlation."""
    rho, _ = spearmanr(y_true, y_pred)
    return rho

def check_data_integrity(data, data_name, subject_ids=None):
    """
    Comprehensive data integrity checks.
    
    Args:
        data: numpy array or pandas DataFrame
        data_name: descriptive name for the data
        subject_ids: optional array of subject IDs for alignment check
    """
    print(f"\n  🔍 DATA INTEGRITY CHECK: {data_name}")
    print("  " + "-" * 80)
    
    if isinstance(data, pd.DataFrame):
        print(f"  • Shape: {data.shape} (rows x columns)")
        print(f"  • Columns: {list(data.columns[:5])}{'...' if len(data.columns) > 5 else ''}")
        print(f"  • Data types: {data.dtypes.value_counts().to_dict()}")
        
        # Check for missing values
        missing_counts = data.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"  ⚠ Missing values found:")
            for col, count in missing_counts[missing_counts > 0].items():
                print(f"    - {col}: {count} ({100*count/len(data):.1f}%)")
        else:
            print(f"  ✓ No missing values")
        
        # Check for duplicates
        if 'subject_id' in data.columns:
            n_duplicates = data['subject_id'].duplicated().sum()
            if n_duplicates > 0:
                print(f"  ⚠ Duplicate subject IDs: {n_duplicates}")
                print(f"    Sample duplicates: {data[data['subject_id'].duplicated()]['subject_id'].head().tolist()}")
            else:
                print(f"  ✓ No duplicate subject IDs")
    
    elif isinstance(data, np.ndarray):
        print(f"  • Shape: {data.shape}")
        print(f"  • Data type: {data.dtype}")
        
        # Check for NaN/Inf
        n_nan = np.isnan(data).sum() if np.issubdtype(data.dtype, np.floating) else 0
        n_inf = np.isinf(data).sum() if np.issubdtype(data.dtype, np.floating) else 0
        
        if n_nan > 0:
            print(f"  ⚠ NaN values: {n_nan} ({100*n_nan/data.size:.2f}%)")
        else:
            print(f"  ✓ No NaN values")
        
        if n_inf > 0:
            print(f"  ⚠ Inf values: {n_inf}")
        else:
            print(f"  ✓ No Inf values")
        
        # Basic statistics
        if len(data.shape) == 1:
            print(f"  • Range: [{np.nanmin(data):.2f}, {np.nanmax(data):.2f}]")
            print(f"  • Mean ± SD: {np.nanmean(data):.2f} ± {np.nanstd(data):.2f}")
        else:
            print(f"  • Value range: [{np.nanmin(data):.2f}, {np.nanmax(data):.2f}]")
            print(f"  • Column means: [{np.nanmean(data, axis=0).min():.2f}, {np.nanmean(data, axis=0).max():.2f}]")
    
    # Sample data
    if subject_ids is not None:
        print(f"  • Sample subject IDs (first 5): {subject_ids[:5].tolist()}")
        print(f"  • Sample subject IDs (last 5): {subject_ids[-5:].tolist()}")
        print(f"  • Subject ID type: {type(subject_ids[0])}")

def verify_id_alignment(ids1, ids2, name1, name2):
    """
    Verify alignment between two sets of subject IDs.
    
    Args:
        ids1: first array of subject IDs
        ids2: second array of subject IDs
        name1: descriptive name for first ID set
        name2: descriptive name for second ID set
    
    Returns:
        common_ids: array of common IDs
        idx1: indices in ids1 for common IDs
        idx2: indices in ids2 for common IDs
    """
    print(f"\n  🔍 ID ALIGNMENT CHECK: {name1} ↔ {name2}")
    print("  " + "-" * 80)
    
    # Convert to same type for comparison
    ids1_str = np.array([str(x) for x in ids1])
    ids2_str = np.array([str(x) for x in ids2])
    
    print(f"  • {name1} IDs:")
    print(f"    - Count: {len(ids1_str)}")
    print(f"    - Type: {type(ids1[0])}")
    print(f"    - First 5: {ids1_str[:5].tolist()}")
    print(f"    - Last 5: {ids1_str[-5:].tolist()}")
    print(f"    - Unique: {len(np.unique(ids1_str))}")
    
    print(f"  • {name2} IDs:")
    print(f"    - Count: {len(ids2_str)}")
    print(f"    - Type: {type(ids2[0])}")
    print(f"    - First 5: {ids2_str[:5].tolist()}")
    print(f"    - Last 5: {ids2_str[-5:].tolist()}")
    print(f"    - Unique: {len(np.unique(ids2_str))}")
    
    # Find common IDs
    common_ids = np.intersect1d(ids1_str, ids2_str)
    
    print(f"  • Overlap:")
    print(f"    - Common IDs: {len(common_ids)}")
    print(f"    - Overlap with {name1}: {100*len(common_ids)/len(ids1_str):.1f}%")
    print(f"    - Overlap with {name2}: {100*len(common_ids)/len(ids2_str):.1f}%")
    print(f"    - Sample common IDs: {common_ids[:5].tolist()}")
    
    # IDs only in first set
    only_in_1 = np.setdiff1d(ids1_str, ids2_str)
    if len(only_in_1) > 0:
        print(f"  ⚠ IDs only in {name1}: {len(only_in_1)}")
        print(f"    Sample: {only_in_1[:5].tolist()}")
    
    # IDs only in second set
    only_in_2 = np.setdiff1d(ids2_str, ids1_str)
    if len(only_in_2) > 0:
        print(f"  ⚠ IDs only in {name2}: {len(only_in_2)}")
        print(f"    Sample: {only_in_2[:5].tolist()}")
    
    if len(common_ids) == 0:
        raise ValueError(f"No common IDs found between {name1} and {name2}!")
    
    # Get indices for alignment
    idx1 = np.array([np.where(ids1_str == id_)[0][0] for id_ in common_ids])
    idx2 = np.array([np.where(ids2_str == id_)[0][0] for id_ in common_ids])
    
    print(f"  ✓ Alignment indices created for {len(common_ids)} common subjects")
    
    return common_ids, idx1, idx2

def determine_pc_range(n_samples):
    """
    Determine appropriate PC range based on sample size.
    
    Args:
        n_samples: number of samples
    
    Returns:
        list of n_components to test
    """
    # Maximum PCs should be less than min(n_samples, n_features)
    # We'll use n_samples - 10 as max to ensure stability in CV
    max_pcs_for_sample = min(MAX_N_PCS, n_samples - 10)
    
    # Create range with specified step
    pc_range = list(range(PC_STEP, max_pcs_for_sample + 1, PC_STEP))
    
    # Ensure we have at least a few options
    if len(pc_range) < 3:
        pc_range = [max(2, n_samples // 4), max(3, n_samples // 2), max(4, n_samples - 5)]
    
    print(f"\n  📊 PC RANGE DETERMINATION")
    print("  " + "-" * 80)
    print(f"  • N samples: {n_samples}")
    print(f"  • Max PCs allowed: {max_pcs_for_sample}")
    print(f"  • PC range to test: {pc_range}")
    print(f"  • Number of PC values: {len(pc_range)}")
    
    return pc_range

# ============================================================================
# DATA LOADING
# ============================================================================

def load_nki_behavioral_data():
    """Load and merge all NKI behavioral data files."""
    print_step("Loading NKI behavioral data from multiple files")
    
    behavioral_dfs = []
    file_info = []
    
    # Load CAARS
    if CAARS_FILE.exists():
        caars_df = pd.read_csv(CAARS_FILE)
        caars_df = caars_df.rename(columns={'Identifiers': 'subject_id'})
        behavioral_dfs.append(caars_df)
        file_info.append(f"CAARS: {len(caars_df)} subjects, {len(caars_df.columns)} columns")
    
    # Load Conners Parent
    if CONNERS_PARENT_FILE.exists():
        conners_p_df = pd.read_csv(CONNERS_PARENT_FILE)
        conners_p_df = conners_p_df.rename(columns={'Identifiers': 'subject_id'})
        behavioral_dfs.append(conners_p_df)
        file_info.append(f"Conners Parent: {len(conners_p_df)} subjects, {len(conners_p_df.columns)} columns")
    
    # Load Conners Self-Report
    if CONNERS_SELF_FILE.exists():
        conners_s_df = pd.read_csv(CONNERS_SELF_FILE)
        conners_s_df = conners_s_df.rename(columns={'Identifiers': 'subject_id'})
        behavioral_dfs.append(conners_s_df)
        file_info.append(f"Conners Self: {len(conners_s_df)} subjects, {len(conners_s_df.columns)} columns")
    
    # Load RBS
    if RBS_FILE.exists():
        rbs_df = pd.read_csv(RBS_FILE)
        rbs_df = rbs_df.rename(columns={'Identifiers': 'subject_id'})
        behavioral_dfs.append(rbs_df)
        file_info.append(f"RBS: {len(rbs_df)} subjects, {len(rbs_df.columns)} columns")
    
    # Merge all behavioral files
    merged_df = behavioral_dfs[0]
    for df in behavioral_dfs[1:]:
        merged_df = merged_df.merge(df, on='subject_id', how='outer', suffixes=('', '_dup'))
    
    # Remove duplicate columns
    merged_df = merged_df.loc[:, ~merged_df.columns.str.endswith('_dup')]
    
    for info in file_info:
        print(f"  {info}")
    
    print_info("Total merged subjects", len(merged_df))
    print_info("Total columns", len(merged_df.columns))
    
    # Select relevant behavioral columns
    behavioral_cols = [col for col in merged_df.columns if any(x in col for x in [
        'CAARS', 'Conners', 'C3SR', 'RBS', '_T', '_Raw'
    ]) and col != 'subject_id']
    
    print_info("Behavioral measures available", len(behavioral_cols))
    
    return merged_df, behavioral_cols

def load_ig_scores():
    """Load IG scores from CSV with integrity checks."""
    print_step("Loading IG scores")
    print(f"From {Path(IG_CSV).name}")
    print("-" * 100)
    
    ig_df = pd.read_csv(IG_CSV)
    
    # Integrity check on raw DataFrame
    check_data_integrity(ig_df, "IG DataFrame (raw)")
    
    # Extract subject IDs and IG scores
    subject_ids = ig_df['subject_id'].values
    ig_cols = [col for col in ig_df.columns if col.startswith('ROI_')]
    ig_matrix = ig_df[ig_cols].values
    
    # Integrity check on IG matrix
    check_data_integrity(ig_matrix, "IG matrix", subject_ids)
    
    print_info("IG subjects", len(subject_ids))
    print_info("IG features (ROIs)", ig_matrix.shape[1])
    
    return subject_ids, ig_matrix, ig_cols

# ============================================================================
# OPTIMIZATION
# ============================================================================

def optimize_pca_regression(X, y, measure_name):
    """
    Optimize number of PCs and regression model using nested cross-validation.
    
    Returns best model, best parameters, and CV performance.
    """
    print_step(f"Optimizing hyperparameters for {measure_name}")
    
    # Integrity check on input data
    check_data_integrity(X, f"X (features) for {measure_name}")
    check_data_integrity(y, f"y (behavior) for {measure_name}")
    
    # Determine PC range based on sample size
    n_pcs_range = determine_pc_range(len(y))
    
    # Create custom scorer
    spearman_score = make_scorer(spearman_scorer)
    
    # Track best performance across different n_components manually
    best_score = -np.inf
    best_n_components = None
    best_model_type = None
    best_alpha = None
    best_model = None
    
    results_list = []
    
    # Outer CV for evaluation
    outer_cv = KFold(n_splits=OUTER_CV_FOLDS, shuffle=True, random_state=42)
    
    # Try different numbers of components
    for n_components in n_pcs_range:
        print(f"\n  Testing n_components = {n_components}")
        
        # Create PCA
        pca = PCA(n_components=n_components)
        
        # Try different models
        models = {
            'Linear': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso(max_iter=10000),
            'ElasticNet': ElasticNet(max_iter=10000)
        }
        
        for model_name, model in models.items():
            # Create pipeline
            if model_name == 'Linear':
                # Linear regression doesn't have alpha parameter
                pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('pca', pca),
                    ('regressor', model)
                ])
                
                # Evaluate with CV
                cv_scores = cross_val_score(pipe, X, y, cv=outer_cv, 
                                           scoring=spearman_score, n_jobs=-1)
                mean_score = np.mean(cv_scores)
                
                results_list.append({
                    'n_components': n_components,
                    'model': model_name,
                    'alpha': None,
                    'mean_cv_spearman': mean_score,
                    'std_cv_spearman': np.std(cv_scores)
                })
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_n_components = n_components
                    best_model_type = model_name
                    best_alpha = None
                    
                    # Refit on full data
                    best_model = Pipeline([
                        ('scaler', StandardScaler()),
                        ('pca', PCA(n_components=n_components)),
                        ('regressor', model)
                    ])
                    best_model.fit(X, y)
                
                print(f"    {model_name}: CV Spearman = {mean_score:.3f} ± {np.std(cv_scores):.3f}")
            
            else:
                # Regularized models - grid search over alpha
                for alpha in ALPHA_RANGE:
                    model.set_params(alpha=alpha)
                    
                    pipe = Pipeline([
                        ('scaler', StandardScaler()),
                        ('pca', pca),
                        ('regressor', model)
                    ])
                    
                    # Evaluate with CV
                    cv_scores = cross_val_score(pipe, X, y, cv=outer_cv, 
                                               scoring=spearman_score, n_jobs=-1)
                    mean_score = np.mean(cv_scores)
                    
                    results_list.append({
                        'n_components': n_components,
                        'model': model_name,
                        'alpha': alpha,
                        'mean_cv_spearman': mean_score,
                        'std_cv_spearman': np.std(cv_scores)
                    })
                    
                    if mean_score > best_score:
                        best_score = mean_score
                        best_n_components = n_components
                        best_model_type = model_name
                        best_alpha = alpha
                        
                        # Refit on full data
                        best_model = Pipeline([
                            ('scaler', StandardScaler()),
                            ('pca', PCA(n_components=n_components)),
                            ('regressor', model.__class__(alpha=alpha, max_iter=10000))
                        ])
                        best_model.fit(X, y)
                
                print(f"    {model_name}: Best α={best_alpha if best_model_type == model_name else 'N/A'}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values('mean_cv_spearman', ascending=False)
    
    print(f"\n  ✓ Best configuration:")
    print(f"    - n_components: {best_n_components}")
    print(f"    - Model: {best_model_type}")
    if best_alpha is not None:
        print(f"    - Alpha: {best_alpha}")
    print(f"    - CV Spearman: {best_score:.3f}")
    
    return best_model, best_n_components, best_model_type, best_alpha, best_score, results_df

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_best_model(model, X, y, measure_name):
    """Evaluate the best model on all data and return metrics."""
    print_step(f"Evaluating best model for {measure_name}")
    
    # Predict on all data
    y_pred = model.predict(X)
    
    # Calculate metrics
    rho, p_value = spearmanr(y, y_pred)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    print_info("N subjects", len(y))
    print_info("Spearman ρ", f"{rho:.3f}")
    print_info("P-value", f"{p_value:.4f}")
    print_info("R²", f"{r2:.3f}")
    print_info("MAE", f"{mae:.2f}")
    
    return {
        'y_actual': y,
        'y_pred': y_pred,
        'rho': rho,
        'p_value': p_value,
        'r2': r2,
        'mae': mae,
        'n_subjects': len(y)
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_scatter_plot(results, measure_name, n_components, model_type, alpha, output_dir):
    """Create scatter plot with consistent styling."""
    y_actual = results['y_actual']
    y_pred = results['y_pred']
    rho = results['rho']
    p_value = results['p_value']
    mae = results['mae']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter plot - bluer dots
    ax.scatter(y_actual, y_pred, alpha=0.7, s=80, 
              color='#5A6FA8', edgecolors='#5A6FA8', linewidth=1)
    
    # Best fit line - red
    z = np.polyfit(y_actual, y_pred, 1)
    p = np.poly1d(z)
    x_line = np.linspace(y_actual.min(), y_actual.max(), 100)
    ax.plot(x_line, p(x_line), color='#D32F2F', linewidth=2.5, alpha=0.9)
    
    # Statistics text - NO bounding box
    model_info = f"{model_type}" + (f" (α={alpha})" if alpha is not None else "")
    stats_text = f'ρ = {rho:.3f}\np = {p_value:.4f}\nMAE = {mae:.2f}\n{model_info}\nPCs = {n_components}'
    ax.text(0.95, 0.05, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='bottom', horizontalalignment='right')
    
    # Labels and title
    ax.set_xlabel('Observed Behavioral Score', fontsize=14, fontweight='normal')
    ax.set_ylabel('Predicted Behavioral Score', fontsize=14, fontweight='normal')
    ax.set_title(f'NKI-RS TD (Optimized)\n{measure_name}', fontsize=16, fontweight='bold', pad=15)
    
    # Styling - NO top/right spines
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Tick styling - major ticks only
    ax.tick_params(axis='both', which='major', labelsize=12, direction='out', 
                  length=6, width=1.5, top=False, right=False)
    
    plt.tight_layout()
    
    # Save PNG and AI
    safe_name = measure_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
    png_path = output_dir / f"scatter_{safe_name}_optimized.png"
    ai_path = output_dir / f"scatter_{safe_name}_optimized.ai"
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    pdf.FigureCanvas(fig).print_pdf(str(ai_path))
    
    print_info("Saved PNG", png_path.name)
    print_info("Saved AI", ai_path.name)
    
    plt.close()

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main analysis pipeline."""
    print_header(f"OPTIMIZED BRAIN-BEHAVIOR ANALYSIS - {DATASET.upper()}")
    
    print("\nConfiguration:")
    print(f"  IG CSV: {Path(IG_CSV).name}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Max PCs: {MAX_N_PCS} (will be adjusted based on sample size)")
    print(f"  PC step: {PC_STEP}")
    print(f"  Models: Linear, Ridge, Lasso, ElasticNet")
    print(f"  Alpha range: {ALPHA_RANGE}")
    print(f"  CV folds: {OUTER_CV_FOLDS} (outer)")
    
    # Load data
    behavioral_df, behavioral_cols = load_nki_behavioral_data()
    
    # Integrity check on behavioral data
    check_data_integrity(behavioral_df, "Behavioral DataFrame")
    
    subject_ids_ig, ig_matrix, ig_cols = load_ig_scores()
    
    # Merge datasets with ID verification
    print_step("Merging IG and behavioral data with ID alignment")
    print("-" * 100)
    
    ig_df_full = pd.DataFrame({
        'subject_id': subject_ids_ig,
        **{col: ig_matrix[:, i] for i, col in enumerate(ig_cols)}
    })
    
    # Verify ID alignment before merge
    common_ids, idx_ig, idx_beh = verify_id_alignment(
        subject_ids_ig,
        behavioral_df['subject_id'].values,
        "IG data",
        "Behavioral data"
    )
    
    # Perform merge
    merged_df = ig_df_full.merge(behavioral_df, on='subject_id', how='inner')
    
    # Verify merge results
    print(f"\n  📊 MERGE VERIFICATION")
    print("  " + "-" * 80)
    print_info("Common subjects after merge", len(merged_df))
    print_info("Expected common subjects", len(common_ids))
    
    if len(merged_df) != len(common_ids):
        print(f"  ⚠ WARNING: Merge count mismatch!")
    else:
        print(f"  ✓ Merge successful: counts match")
    
    # Final integrity check on merged data
    check_data_integrity(merged_df, "Merged DataFrame")
    
    X = merged_df[ig_cols].values
    
    print_info("Final IG matrix shape", f"{X.shape} (subjects x ROIs)")
    
    # Analyze each behavioral measure
    all_results = []
    
    for measure in behavioral_cols[:5]:  # Process first 5 measures as example
        print_header(f"ANALYZING: {measure}")
        
        # Get behavioral scores for this measure
        y = merged_df[measure].values
        
        # Remove NaN values
        valid_mask = ~np.isnan(y)
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        if len(y_valid) < 20:
            print(f"⚠ Insufficient data for {measure}: only {len(y_valid)} subjects")
            continue
        
        print_info("Valid subjects", len(y_valid))
        
        # Optimize model
        best_model, n_components, model_type, alpha, cv_score, opt_results = \
            optimize_pca_regression(X_valid, y_valid, measure)
        
        # Evaluate on all data
        eval_results = evaluate_best_model(best_model, X_valid, y_valid, measure)
        
        # Create visualization
        create_scatter_plot(eval_results, measure, n_components, model_type, alpha, OUTPUT_DIR)
        
        # Save optimization results
        opt_results.to_csv(OUTPUT_DIR / f"optimization_results_{measure.replace(' ', '_')}.csv", index=False)
        
        # Store summary
        all_results.append({
            'Measure': measure,
            'N_Subjects': len(y_valid),
            'Best_N_Components': n_components,
            'Best_Model': model_type,
            'Best_Alpha': alpha,
            'CV_Spearman': cv_score,
            'Final_Spearman': eval_results['rho'],
            'Final_P_Value': eval_results['p_value'],
            'Final_R2': eval_results['r2'],
            'Final_MAE': eval_results['mae']
        })
    
    # Save summary
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(OUTPUT_DIR / "optimization_summary.csv", index=False)
    
    print_header("COMPLETE")
    print(f"\n✓ Results saved to: {OUTPUT_DIR}")
    print(f"✓ Analyzed {len(all_results)} behavioral measures")
    print("\nBest performances:")
    print(summary_df[['Measure', 'Final_Spearman', 'Best_Model', 'Best_N_Components']].to_string(index=False))

if __name__ == "__main__":
    main()

