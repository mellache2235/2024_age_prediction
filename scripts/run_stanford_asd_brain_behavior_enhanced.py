#!/usr/bin/env python3
"""
Enhanced brain-behavior analysis for Stanford ASD with all paths pre-configured.
Includes: Elbow plot, Linear Regression, Scatter plots, PC importance, PC loadings.

Just run: python run_stanford_asd_brain_behavior_enhanced.py
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
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Add to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from logging_utils import (print_section_header, print_step, print_success, 
                           print_warning, print_error, print_info, print_completion)
from plot_styles import create_standardized_scatter, get_dataset_title, setup_arial_font, DPI, FIGURE_FACECOLOR

# Setup Arial font globally
setup_arial_font()

# ============================================================================
# PRE-CONFIGURED PATHS (NO ARGUMENTS NEEDED)
# ============================================================================
DATASET = "stanford_asd"
IG_CSV = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/integrated_gradients/stanford_asd_features_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS.csv"
SRS_FILE = "/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/prepare_data/stanford_autism/SRS_data_20230925.csv"
OUTPUT_DIR = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/stanford_asd"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_stanford_ig_scores(ig_csv):
    """Load Stanford ASD IG scores."""
    print_step("Loading IG scores", f"From {Path(ig_csv).name}")
    
    df = pd.read_csv(ig_csv)
    
    # Drop Unnamed: 0 if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    # Identify subject ID column
    id_col = None
    for col in ['subject_id', 'id', 'ID', 'Subject_ID', 'record_id']:
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


def load_srs_data(srs_file):
    """Load Stanford SRS behavioral data."""
    print_step("Loading SRS behavioral data", f"From {Path(srs_file).name}")
    
    # Load SRS file - handle potential whitespace in column names
    srs_df = pd.read_csv(srs_file)
    
    # Strip whitespace from column names
    srs_df.columns = srs_df.columns.str.strip()
    
    print_info(f"Total rows: {len(srs_df)}")
    print_info(f"First few columns: {list(srs_df.columns[:5])}")
    
    # Identify subject ID column (case-insensitive, flexible matching)
    id_col = None
    for col in srs_df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['record', 'subject', 'participant', 'pid']) and \
           any(keyword in col_lower for keyword in ['id', 'pid']):
            id_col = col
            break
    
    if id_col is None:
        print_error(f"No subject ID column found in SRS file")
        print_info(f"All columns: {list(srs_df.columns)}")
        raise ValueError(f"No subject ID column found in SRS file. Available columns: {list(srs_df.columns)}")
    
    print_info(f"Using ID column: {id_col}")
    
    # Drop duplicates (keep last as specified)
    srs_df = srs_df.drop_duplicates(subset=[id_col], keep='last')
    
    # Convert ID to string
    srs_df[id_col] = srs_df[id_col].astype(str)
    
    # Rename to subject_id for consistency
    if id_col != 'subject_id':
        srs_df = srs_df.rename(columns={id_col: 'subject_id'})
    
    # Check if srs_total_score_standard exists, or find alternative
    if 'srs_total_score_standard' not in srs_df.columns:
        print_warning("Column 'srs_total_score_standard' not found in SRS file")
        
        # Try to find SRS total score column (various naming conventions)
        srs_cols = []
        for col in srs_df.columns:
            col_lower = col.lower()
            # Look for columns with 'total' and either 'score' or 't-score' or 'tscore'
            if 'total' in col_lower and any(x in col_lower for x in ['score', 't-score', 'tscore', 't_score']):
                # Prioritize columns that also mention 'srs'
                if 'srs' in col_lower:
                    srs_cols.insert(0, col)  # Add to front
                else:
                    srs_cols.append(col)
        
        if srs_cols:
            print_info(f"Found SRS total score columns: {srs_cols}")
            print_info(f"Using: {srs_cols[0]}")
            srs_df = srs_df.rename(columns={srs_cols[0]: 'srs_total_score_standard'})
        else:
            print_error("No SRS total score column found")
            print_info(f"All columns: {list(srs_df.columns)}")
            raise ValueError(f"No SRS total score column found. Available columns: {list(srs_df.columns)}")
    
    print_info(f"SRS subjects: {len(srs_df)}")
    print_info(f"SRS score column: srs_total_score_standard (renamed from original if needed)")
    
    return srs_df


def merge_data(ig_df, srs_df):
    """Merge IG and SRS data by subject ID."""
    print_step("Merging data", "Matching subject IDs")
    
    print_info(f"IG subjects: {len(ig_df)}")
    print_info(f"SRS subjects: {len(srs_df)}")
    
    # Merge on subject_id
    merged = pd.merge(ig_df, srs_df, on='subject_id', how='inner')
    
    common_subjects = len(merged)
    print_success(f"Merged: {common_subjects} subjects with both IG and SRS data")
    
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


def perform_linear_regression(pca_scores, srs_scores, output_dir):
    """Perform linear regression using all PCs to predict SRS total score."""
    print_step("Linear regression for SRS Total Score", "Using all PCs as predictors")
    
    # Remove NaNs
    valid_mask = ~np.isnan(srs_scores)
    X = pca_scores[valid_mask]
    y = srs_scores[valid_mask]
    
    if len(y) < 10:
        print_warning(f"Insufficient valid data: {len(y)} subjects")
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
    create_scatter_plot(y, y_pred, rho, p_value, "SRS Total Score", DATASET, output_dir)
    
    # Get PC importance (absolute coefficients)
    pc_importance = np.abs(model.coef_)
    pc_ranks = np.argsort(pc_importance)[::-1]
    
    return {
        'behavioral_measure': 'SRS_Total_Score',
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
    stats_text = f"r = {rho:.3f}\np {p_str}"
    
    # Get standardized title
    title = get_dataset_title(dataset_name)
    
    # Create safe filename
    safe_name = behavioral_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
    save_path = Path(output_dir) / f'scatter_{safe_name}'
    
    # Use centralized plotting function (handles PNG + TIFF + AI export)
    fig, ax = plt.subplots(figsize=(6, 6))
    
    create_standardized_scatter(
        ax, y_actual, y_pred,
        title=title,
        xlabel='Observed Behavioral Score',
        ylabel='Predicted Behavioral Score',
        stats_text=stats_text,
        is_subplot=False
    )
    
    # Save with centralized export (PNG + TIFF + AI)
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


def save_results(results, pc_loadings_dict, output_dir):
    """Save all results to CSV files."""
    print_step("Saving results", "Writing CSV files")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if results is None:
        print_warning("No results to save")
        return
    
    # 1. Save regression results
    results_df = pd.DataFrame([{
        'Behavioral_Measure': results['behavioral_measure'],
        'N_Subjects': results['n_subjects'],
        'N_Features': results['n_features'],
        'Spearman_Rho': results['spearman_rho'],
        'P_Value': results['p_value'],
        'R2': results['r2']
    }])
    results_path = output_dir / 'linear_regression_results.csv'
    results_df.to_csv(results_path, index=False)
    print_success(f"Regression results: {results_path.name}")
    
    # 2. Save PC importance
    pc_importance_df = pd.DataFrame({
        'PC': [f'PC{i+1}' for i in range(len(results['pc_importance']))],
        'Importance': results['pc_importance'],
        'Rank': [np.where(results['pc_ranks'] == i)[0][0] + 1 for i in range(len(results['pc_importance']))]
    })
    pc_imp_path = output_dir / f'pc_importance_SRS_Total_Score.csv'
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
    print_section_header("ENHANCED BRAIN-BEHAVIOR ANALYSIS - STANFORD ASD")
    
    print_info(f"IG CSV:     {IG_CSV}")
    print_info(f"SRS File:   {SRS_FILE}")
    print_info(f"Output:     {OUTPUT_DIR}")
    print()
    
    # Create output directory if it doesn't exist
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Load data
        ig_df, roi_cols = load_stanford_ig_scores(IG_CSV)
        srs_df = load_srs_data(SRS_FILE)
        
        # 2. Merge data
        merged_df = merge_data(ig_df, srs_df)
        
        # Extract IG matrix and SRS scores
        ig_matrix = merged_df[roi_cols].values
        srs_scores = merged_df['srs_total_score_standard'].values
        
        print_info(f"IG matrix shape: {ig_matrix.shape} (subjects x ROIs)")
        print_info(f"SRS scores shape: {srs_scores.shape}")
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
        
        # 6. Perform linear regression for SRS total score
        print_section_header("LINEAR REGRESSION RESULTS")
        print()
        result = perform_linear_regression(pca_scores_optimal, srs_scores, OUTPUT_DIR)
        
        if result is not None:
            # Print summary
            if result['p_value'] < 0.001:
                p_display = "< 0.001"
            else:
                p_display = f"= {result['p_value']:.4f}"
            print_success(f"✓ ρ = {result['spearman_rho']:.3f}, p {p_display}")
        
        print()
        
        # 7. Save all results
        save_results(result, pc_loadings_dict, OUTPUT_DIR)
        
        print()
        print_completion("Stanford ASD Brain-Behavior Analysis Complete!")
        print_info(f"Results saved to: {OUTPUT_DIR}")
        
    except Exception as e:
        print()
        print_error(f"Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

