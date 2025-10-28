#!/usr/bin/env python3
"""
Enhanced brain-behavior analysis for NKI-RS TD with all paths pre-configured.
Includes: Elbow plot, Linear Regression, Scatter plots, PC importance, PC loadings.

Just run: python run_nki_brain_behavior_enhanced.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, make_scorer
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Add to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from logging_utils import (print_section_header, print_step, print_success, 
                           print_warning, print_error, print_info, print_completion)
from plot_styles import create_single_scatter_plot, get_dataset_title, setup_arial_font

# Setup Arial font globally
setup_arial_font()

# ============================================================================
# PRE-CONFIGURED PATHS (NO ARGUMENTS NEEDED)
# ============================================================================
DATASET = "nki_rs_td"
IG_CSV = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/integrated_gradients/nki_cog_dev_wIDS_features_IG_convnet_regressor_single_model_fold_0.csv"
BEHAVIORAL_DIR = "/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/FLUX/assessment_data"
OUTPUT_DIR = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/nki_rs_td"

# ============================================================================
# OPTIMIZATION SETTINGS (Set OPTIMIZE=True to enable)
# ============================================================================
OPTIMIZE = False  # Set to True for hyperparameter tuning (Ridge/Lasso/ElasticNet, optimal PCs)
# If True: Tests multiple models, PCs, and regularization strengths
# If False: Uses standard LinearRegression with 80% variance threshold (faster)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_nki_ig_scores(ig_csv):
    """Load NKI IG scores."""
    print_step("Loading IG scores", f"From {Path(ig_csv).name}")
    
    df = pd.read_csv(ig_csv)
    
    # Drop Unnamed: 0 if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    # Identify subject ID column
    id_col = None
    for col in ['subject_id', 'id', 'ID', 'Subject_ID']:
        if col in df.columns:
            id_col = col
            break
    
    if id_col is None:
        raise ValueError("No subject ID column found in IG CSV")
    
    # Standardize to 'subject_id'
    if id_col != 'subject_id':
        df = df.rename(columns={id_col: 'subject_id'})
    
    # Convert subject IDs to string
    df['subject_id'] = df['subject_id'].astype(str)
    
    # ROI columns are all columns except subject_id
    roi_cols = [col for col in df.columns if col != 'subject_id']
    
    print_info(f"IG subjects: {len(df)}")
    print_info(f"IG features (ROIs): {len(roi_cols)}")
    print_info(f"Sample ROI columns: {roi_cols[:3]}...")
    
    return df, roi_cols


def load_nki_behavioral_data(behavioral_dir):
    """Load NKI behavioral data (CAARS and Conners 3)."""
    print_step("Loading NKI behavioral data", f"From {Path(behavioral_dir).name}")
    
    behavioral_dir = Path(behavioral_dir)
    
    # Look for multiple behavioral files
    behavioral_files = []
    for pattern in ['*CAARS*.csv', '*Conners*.csv', '*RBS*.csv']:
        behavioral_files.extend(behavioral_dir.glob(pattern))
    
    if not behavioral_files:
        raise ValueError(f"No behavioral files found in {behavioral_dir}")
    
    print_info(f"Found {len(behavioral_files)} behavioral files:")
    for f in behavioral_files:
        print_info(f"  • {f.name}")
    
    # Load and merge all behavioral files
    all_dfs = []
    for bf in behavioral_files:
        df = pd.read_csv(bf)
        
        # Identify subject ID column
        id_col = None
        for col in df.columns:
            if 'id' in col.lower() or 'subject' in col.lower():
                id_col = col
                break
        
        if id_col is None:
            print_warning(f"No subject ID column in {bf.name}, skipping")
            continue
        
        # Standardize to 'subject_id'
        if id_col != 'subject_id':
            df = df.rename(columns={id_col: 'subject_id'})
        
        # Convert subject IDs to string
        df['subject_id'] = df['subject_id'].astype(str)
        
        all_dfs.append(df)
    
    # Merge all dataframes on subject_id
    if not all_dfs:
        raise ValueError("No valid behavioral files found")
    
    merged_df = all_dfs[0]
    for df in all_dfs[1:]:
        merged_df = pd.merge(merged_df, df, on='subject_id', how='outer')
    
    # Auto-detect behavioral columns (CAARS, Conners, RBS)
    behavioral_cols = [col for col in merged_df.columns if 
                       'CAARS' in col.upper() or 
                       'CONNERS' in col.upper() or
                       'RBS' in col.upper() or
                       ('TOTAL' in col.upper() and ('INATTENTION' in col.upper() or 'HYPERACTIVITY' in col.upper())) or
                       ('T-SCORE' in col.upper() or 'T_SCORE' in col.upper())]
    
    # Exclude subject_id if it got included
    behavioral_cols = [col for col in behavioral_cols if col != 'subject_id']
    
    if not behavioral_cols:
        raise ValueError("No behavioral columns found")
    
    print_info(f"Total behavioral subjects: {len(merged_df)}")
    print_info(f"Behavioral measures: {len(behavioral_cols)}")
    print_info(f"Sample columns: {behavioral_cols[:5]}...")
    
    return merged_df, behavioral_cols


def merge_data(ig_df, behavioral_df):
    """Merge IG and behavioral data by subject ID."""
    print_step("Merging data", "Matching subject IDs")
    
    print_info(f"IG subjects: {len(ig_df)}")
    print_info(f"Behavioral subjects: {len(behavioral_df)}")
    
    # Remove duplicates in behavioral data (keep first)
    behavioral_df = behavioral_df.drop_duplicates(subset='subject_id', keep='first')
    print_info(f"Behavioral subjects after deduplication: {len(behavioral_df)}")
    
    # Merge on subject_id
    merged = pd.merge(ig_df, behavioral_df, on='subject_id', how='inner')
    
    common_subjects = len(merged)
    print_success(f"Merged: {common_subjects} subjects with both IG and behavioral data")
    
    if common_subjects < 10:
        raise ValueError(f"Insufficient overlap: only {common_subjects} common subjects")
    
    return merged


def create_elbow_plot(pca, output_dir):
    """Create elbow plot to determine optimal number of PCs."""
    print_step("Creating elbow plot", "Determining optimal number of PCs")
    
    # Get explained variance
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: Scree plot
    ax1.plot(range(1, len(explained_var) + 1), explained_var * 100, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Principal Component', fontsize=11)
    ax1.set_ylabel('Variance Explained (%)', fontsize=11)
    ax1.set_title('Scree Plot', fontsize=12, fontweight='bold')
    ax1.grid(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(axis='both', which='major', labelsize=10, direction='out', length=4, width=1)
    
    # Plot 2: Cumulative variance
    ax2.plot(range(1, len(cumulative_var) + 1), cumulative_var * 100, 'ro-', linewidth=2, markersize=8)
    ax2.axhline(y=80, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='80% threshold')
    ax2.set_xlabel('Number of Components', fontsize=11)
    ax2.set_ylabel('Cumulative Variance Explained (%)', fontsize=11)
    ax2.set_title('Cumulative Variance', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(axis='both', which='major', labelsize=10, direction='out', length=4, width=1)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir) / 'elbow_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_success(f"Elbow plot saved: {output_path.name}")
    
    # Determine optimal number of PCs (80% variance threshold)
    optimal_pcs = np.argmax(cumulative_var >= 0.80) + 1
    print_info(f"Optimal PCs (80% variance): {optimal_pcs}")
    print_info(f"Variance explained by {optimal_pcs} PCs: {cumulative_var[optimal_pcs-1]*100:.2f}%")
    
    return optimal_pcs


def perform_linear_regression(pca_scores, behavioral_scores, behavioral_name, output_dir):
    """Perform linear regression using all PCs to predict behavioral scores."""
    print_step(f"Linear regression for {behavioral_name}", "Using all PCs as predictors")
    
    # Remove NaNs
    valid_mask = ~np.isnan(behavioral_scores)
    X = pca_scores[valid_mask]
    y = behavioral_scores[valid_mask]
    
    if len(y) < 10:
        print_warning(f"Insufficient valid data for {behavioral_name}: {len(y)} subjects")
        return None
    
    n_features = X.shape[1]
    n_samples = len(y)
    
    # Fit linear regression on ALL data
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict on ALL data
    y_pred = model.predict(X)
    
    # Calculate Spearman correlation
    rho, p_value = spearmanr(y, y_pred)
    
    # Calculate R² on all data
    from sklearn.metrics import r2_score
    r2 = r2_score(y, y_pred)
    
    # Format p-value for console output
    if p_value < 0.001:
        p_str = "< 0.001"
    else:
        p_str = f"= {p_value:.4f}"
    
    print_info(f"N subjects: {len(y)}")
    print_info(f"N features (PCs): {n_features}")
    print_info(f"Spearman ρ = {rho:.3f}, p {p_str}")
    print_info(f"R² = {r2:.3f}")
    
    # Create scatter plot
    create_scatter_plot(y, y_pred, rho, p_value, behavioral_name, DATASET, output_dir)
    
    # Get PC importance (absolute coefficients)
    pc_importance = np.abs(model.coef_)
    pc_ranks = np.argsort(pc_importance)[::-1]
    
    return {
        'behavioral_measure': behavioral_name,
        'n_subjects': len(y),
        'n_features': n_features,
        'spearman_rho': rho,
        'p_value': p_value,
        'r2': r2,
        'pc_importance': pc_importance,
        'pc_ranks': pc_ranks
    }


def create_scatter_plot(y_actual, y_pred, rho, p_value, behavioral_name, dataset_name, output_dir):
    """Create scatter plot using centralized styling."""
    # Format p-value
    p_str = "< 0.001" if p_value < 0.001 else f"= {p_value:.3f}"
    
    # Create stats text
    stats_text = f"ρ = {rho:.3f}\np {p_str}"
    
    # Get standardized title
    title = get_dataset_title(dataset_name)
    
    # Create safe filename
    safe_name = behavioral_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
    save_path = Path(output_dir) / f'scatter_{safe_name}'
    
    # Use centralized plotting function (handles PNG + TIFF + AI export)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    
    from plot_styles import create_standardized_scatter
    create_standardized_scatter(
        ax, y_actual, y_pred,
        title=title,
        xlabel='Observed Behavioral Score',
        ylabel='Predicted Behavioral Score',
        stats_text=stats_text,
        is_subplot=False
    )
    
    # Save with centralized export (PNG + TIFF + AI)
    from plot_styles import DPI, FIGURE_FACECOLOR
    import matplotlib.backends.backend_pdf as pdf_backend
    
    plt.tight_layout()
    
    png_path = save_path.with_suffix('.png')
    tiff_path = save_path.with_suffix('.tiff')
    ai_path = save_path.with_suffix('.ai')
    
    plt.savefig(png_path, dpi=DPI, bbox_inches='tight', facecolor=FIGURE_FACECOLOR, edgecolor='none')
    plt.savefig(tiff_path, dpi=DPI, bbox_inches='tight', facecolor=FIGURE_FACECOLOR, edgecolor='none',
               format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
    pdf_backend.FigureCanvas(fig).print_pdf(str(ai_path))
    
    plt.close()
    
    print(f"  ✓ Saved: {png_path.name} + {tiff_path.name} + {ai_path.name}")


def get_pc_loadings(pca, roi_cols, n_top=10):
    """Get top contributing brain regions for each PC."""
    print_step("Extracting PC loadings", f"Top {n_top} regions per PC")
    
    loadings = pca.components_
    
    pc_loadings_dict = {}
    for i in range(loadings.shape[0]):
        # Get absolute loadings for this PC
        abs_loadings = np.abs(loadings[i, :])
        top_indices = np.argsort(abs_loadings)[::-1][:n_top]
        
        pc_loadings_dict[f'PC{i+1}'] = [
            (roi_cols[idx], abs_loadings[idx]) for idx in top_indices
        ]
    
    return pc_loadings_dict


def save_results(results_list, pc_loadings_dict, output_dir):
    """Save all results to CSV files."""
    print_step("Saving results", "Writing CSV files")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save regression results
    results_df = pd.DataFrame([
        {
            'Behavioral_Measure': r['behavioral_measure'],
            'N_Subjects': r['n_subjects'],
            'N_Features': r['n_features'],
            'Spearman_Rho': r['spearman_rho'],
            'P_Value': r['p_value'],
            'R2': r['r2']
        }
        for r in results_list if r is not None
    ])
    results_path = output_dir / 'linear_regression_results.csv'
    results_df.to_csv(results_path, index=False)
    print_success(f"Regression results: {results_path.name}")
    
    # 2. Save PC importance
    for r in results_list:
        if r is not None:
            pc_importance_df = pd.DataFrame({
                'PC': [f'PC{i+1}' for i in range(len(r['pc_importance']))],
                'Importance': r['pc_importance'],
                'Rank': [np.where(r['pc_ranks'] == i)[0][0] + 1 for i in range(len(r['pc_importance']))]
            })
            safe_name = r['behavioral_measure'].replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
            pc_imp_path = output_dir / f'pc_importance_{safe_name}.csv'
            pc_importance_df.to_csv(pc_imp_path, index=False)
            print_success(f"PC importance: {pc_imp_path.name}")
    
    # 3. Save PC loadings
    for pc_name, loadings in pc_loadings_dict.items():
        loadings_df = pd.DataFrame(loadings, columns=['Brain_Region', 'Loading'])
        loadings_path = output_dir / f'{pc_name}_loadings.csv'
        loadings_df.to_csv(loadings_path, index=False)
        print_success(f"PC loadings: {loadings_path.name}")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    print_section_header("ENHANCED BRAIN-BEHAVIOR ANALYSIS - NKI-RS TD")
    
    print_info(f"IG CSV:          {IG_CSV}")
    print_info(f"Behavioral DIR:  {BEHAVIORAL_DIR}")
    print_info(f"Output:          {OUTPUT_DIR}")
    print()
    
    # Create output directory if it doesn't exist
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Load data
        ig_df, roi_cols = load_nki_ig_scores(IG_CSV)
        behavioral_df, caars_cols = load_nki_behavioral_data(BEHAVIORAL_DIR)
        
        # 2. Merge data
        merged_df = merge_data(ig_df, behavioral_df)
        
        # Extract IG matrix and behavioral scores
        ig_matrix = merged_df[roi_cols].values
        
        print_info(f"IG matrix shape: {ig_matrix.shape} (subjects x ROIs)")
        print()
        
        # 3. Perform PCA with many components
        print_step("Performing PCA", "Using 50 components for elbow plot")
        scaler = StandardScaler()
        ig_scaled = scaler.fit_transform(ig_matrix)
        
        pca = PCA(n_components=min(50, ig_matrix.shape[0], ig_matrix.shape[1]))
        pca_scores = pca.fit_transform(ig_scaled)
        
        print_info(f"PCA scores shape: {pca_scores.shape}")
        print_info(f"Variance explained by first 3 PCs: {pca.explained_variance_ratio_[:3].sum()*100:.2f}%")
        print()
        
        # 4. Create elbow plot and determine optimal PCs
        optimal_pcs = create_elbow_plot(pca, OUTPUT_DIR)
        print()
        
        # Use optimal number of PCs for regression
        pca_scores_optimal = pca_scores[:, :optimal_pcs]
        print_info(f"Using {optimal_pcs} PCs for linear regression")
        print()
        
        # 5. Get PC loadings
        pc_loadings_dict = get_pc_loadings(pca, roi_cols, n_top=10)
        print()
        
        # 6. Perform linear regression for each behavioral measure
        print_section_header("LINEAR REGRESSION RESULTS")
        results_list = []
        for caars_col in caars_cols:
            print()
            print_step(f"Analyzing {caars_col}", "Predicted vs Actual Behavioral Scores")
            behavioral_scores = pd.to_numeric(merged_df[caars_col], errors='coerce').values
            result = perform_linear_regression(pca_scores_optimal, behavioral_scores, caars_col, OUTPUT_DIR)
            if result is not None:
                results_list.append(result)
                # Print summary
                if result['p_value'] < 0.001:
                    p_display = "< 0.001"
                else:
                    p_display = f"= {result['p_value']:.4f}"
                print_success(f"✓ ρ = {result['spearman_rho']:.3f}, p {p_display}")
        
        print()
        
        # 7. Save all results
        save_results(results_list, pc_loadings_dict, OUTPUT_DIR)
        
        print()
        print_completion("NKI-RS TD Enhanced Analysis Complete!")
        print_info(f"Results saved to: {OUTPUT_DIR}")
        
    except Exception as e:
        print()
        print_error(f"Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
