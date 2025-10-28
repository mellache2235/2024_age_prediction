#!/usr/bin/env python3
"""
Enhanced brain-behavior analysis for ABIDE ASD with all paths pre-configured.
Includes: Elbow plot, Linear Regression, Scatter plots, PC importance, PC loadings.

Uses ADOS measures for behavioral correlation.

Just run: python run_abide_asd_brain_behavior_enhanced.py
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
DATASET = "abide_asd"
IG_CSV = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/integrated_gradients/abide_asd_features_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS.csv"
PKLZ_DIR = "/oak/stanford/groups/menon/deriveddata/public/abide/restfmri/timeseries/group_level/brainnetome/normz/"
OUTPUT_DIR = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/abide_asd"

# ABIDE sites to include (matching create_data script)
ABIDE_SITES = ['NYU', 'SDSU', 'STANFORD', 'Stanford', 'TCD-1', 'UM', 'USM', 'Yale']

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_abide_ig_scores(ig_csv):
    """Load ABIDE ASD IG scores."""
    print_step("Loading IG scores", f"From {Path(ig_csv).name}")
    
    df = pd.read_csv(ig_csv)
    
    # Drop Unnamed: 0 if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    # Identify subject ID column
    id_col = None
    for col in ['subjid', 'subject_id', 'id', 'ID', 'Subject_ID']:
        if col in df.columns:
            id_col = col
            break
    
    if id_col is None:
        raise ValueError("No subject ID column found in IG CSV")
    
    print_info(f"Using ID column: {id_col}")
    
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


def load_abide_behavioral_data(pklz_dir, sites):
    """Load ABIDE behavioral data from .pklz files (ASD subjects only)."""
    print_step("Loading ABIDE behavioral data", f"From {Path(pklz_dir).name}")
    
    pklz_dir = Path(pklz_dir)
    
    # Find all .pklz files matching the specified sites AND ending with 246ROIs.pklz
    all_files = os.listdir(pklz_dir)
    filtered_files = []
    for file_name in all_files:
        # Must contain one of the sites AND end with 246ROIs.pklz
        if '246ROIs.pklz' in file_name and any(site in file_name for site in sites):
            filtered_files.append(pklz_dir / file_name)
    
    if not filtered_files:
        raise ValueError(f"No 246ROIs.pklz files found for sites: {sites}")
    
    print_info(f"Found {len(filtered_files)} 246ROIs.pklz files for {len(sites)} sites")
    
    # Load and concatenate all dataframes
    appended_data = []
    for file_path in filtered_files:
        try:
            data = pd.read_pickle(file_path)
            # Remove NaN entries
            data = data[~pd.isna(data)]
            appended_data.append(data)
            print_info(f"  Loaded {len(data)} subjects from {file_path.name}")
        except Exception as e:
            print_warning(f"  Could not load {file_path.name}: {e}")
    
    if not appended_data:
        raise ValueError("No data loaded from .pklz files")
    
    # Concatenate all data
    combined_df = pd.concat(appended_data, ignore_index=True)
    
    # Convert label to string if not already
    combined_df['label'] = combined_df['label'].astype(str)
    
    print_info(f"Total subjects loaded: {len(combined_df)}")
    print_info(f"Unique label values: {combined_df['label'].unique()}")
    
    # Filter for ASD subjects only (label == 'asd' string)
    asd_df = combined_df[combined_df['label'] == 'asd'].copy()
    
    print_info(f"ASD subjects (label='asd'): {len(asd_df)}")
    
    # Filter for developmental age (age <= 21)
    if 'age' in asd_df.columns:
        asd_df['age'] = pd.to_numeric(asd_df['age'], errors='coerce')
        asd_df = asd_df[asd_df['age'] <= 21]
        print_info(f"ASD subjects age <= 21: {len(asd_df)}")
    
    # Reset index
    asd_df = asd_df.reset_index(drop=True)
    
    # Identify and rename subject ID column
    id_col = None
    for col in ['subjid', 'subject_id', 'id', 'ID', 'Subject_ID']:
        if col in asd_df.columns:
            id_col = col
            break
    
    if id_col is None:
        print_warning("No standard subject ID column found in .pklz data")
        print_info(f"Available columns: {list(asd_df.columns)}")
        raise ValueError(f"No subject ID column found in .pklz files")
    
    print_info(f"Using ID column from .pklz: {id_col}")
    
    # Standardize to subject_id
    if id_col != 'subject_id':
        asd_df = asd_df.rename(columns={id_col: 'subject_id'})
    
    # Ensure subject_id is string
    asd_df['subject_id'] = asd_df['subject_id'].astype(str)
    
    # Check for specific ADOS columns (case-insensitive)
    # Looking for: ados_total, ados_social, ados_comm
    target_ados = ['ados_total', 'ados_social', 'ados_comm']
    ados_cols = []
    
    print_info(f"Looking for ADOS columns: {target_ados}")
    print_info(f"Available columns in .pklz data: {list(asd_df.columns)}")
    
    for target in target_ados:
        # Try exact match (lowercase)
        if target in asd_df.columns:
            ados_cols.append(target)
            print_info(f"  Found: {target}")
        else:
            # Try case-insensitive match
            found = False
            for col in asd_df.columns:
                if col.lower() == target:
                    ados_cols.append(col)
                    print_info(f"  Found: {col} (matched {target})")
                    found = True
                    break
            if not found:
                print_warning(f"  NOT FOUND: {target}")
    
    # Check for any ADOS-related columns
    all_ados_cols = [col for col in asd_df.columns if 'ados' in col.lower()]
    
    if not ados_cols:
        print_warning(f"None of the target ADOS columns found: {target_ados}")
        if all_ados_cols:
            print_info(f"Available ADOS columns: {all_ados_cols}")
            # Check if any have non-null values
            for col in all_ados_cols[:5]:  # Check first 5
                non_null = asd_df[col].notna().sum()
                print_info(f"  {col}: {non_null} non-null values")
        else:
            print_warning("No ADOS columns found in data at all")
            print_info(f"All columns: {list(asd_df.columns)}")
    else:
        print_info(f"ADOS behavioral measures found: {ados_cols}")
        # Check how many non-null values each has
        for col in ados_cols:
            non_null = asd_df[col].notna().sum()
            print_info(f"  {col}: {non_null} non-null values out of {len(asd_df)}")
    
    return asd_df, ados_cols


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
    
    if len(results_df) > 0:
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
    print_section_header("ENHANCED BRAIN-BEHAVIOR ANALYSIS - ABIDE ASD")
    
    print_info(f"IG CSV:     {IG_CSV}")
    print_info(f"PKLZ DIR:   {PKLZ_DIR}")
    print_info(f"Sites:      {', '.join(ABIDE_SITES)}")
    print_info(f"Output:     {OUTPUT_DIR}")
    print()
    
    # Create output directory if it doesn't exist
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Load data
        ig_df, roi_cols = load_abide_ig_scores(IG_CSV)
        behavioral_df, ados_cols = load_abide_behavioral_data(PKLZ_DIR, ABIDE_SITES)
        
        # 2. Merge data
        merged_df = merge_data(ig_df, behavioral_df)
        
        # Debug: Check if ADOS columns are in merged data
        print_info(f"Merged data columns: {list(merged_df.columns)}")
        if ados_cols:
            for col in ados_cols:
                if col in merged_df.columns:
                    non_null = merged_df[col].notna().sum()
                    print_info(f"  {col} in merged data: {non_null} non-null values")
                else:
                    print_warning(f"  {col} NOT in merged data!")
        
        # Extract IG matrix
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
        
        # 6. Perform linear regression for each ADOS measure
        print_section_header("LINEAR REGRESSION RESULTS")
        results_list = []
        
        if not ados_cols:
            print_warning("No ADOS/behavioral columns found - skipping regression")
        else:
            for ados_col in ados_cols:
                print()
                print_step(f"Analyzing {ados_col}", "Predicted vs Actual Behavioral Scores")
                
                # Check if column exists in merged data
                if ados_col not in merged_df.columns:
                    print_warning(f"Column '{ados_col}' not found in merged data - skipping")
                    continue
                
                # Extract column and ensure it's a Series, then convert to numeric
                try:
                    col_data = merged_df[ados_col]
                    if isinstance(col_data, pd.DataFrame):
                        # If it's a DataFrame (shouldn't be), take the first column
                        col_data = col_data.iloc[:, 0]
                    behavioral_scores = pd.to_numeric(col_data, errors='coerce').values
                except Exception as e:
                    print_warning(f"Error processing column '{ados_col}': {e}")
                    continue
                
                result = perform_linear_regression(pca_scores_optimal, behavioral_scores, ados_col, OUTPUT_DIR)
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
        print_completion("ABIDE ASD Brain-Behavior Analysis Complete!")
        print_info(f"Results saved to: {OUTPUT_DIR}")
        
    except Exception as e:
        print()
        print_error(f"Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

