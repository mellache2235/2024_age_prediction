#!/usr/bin/env python3
"""
Enhanced brain-behavior analysis for CMI-HBN TD with all paths pre-configured.
Includes: Elbow plot, Linear Regression, Scatter plots, PC importance, PC loadings.

Just run: python run_cmihbn_brain_behavior_enhanced.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.backends.backend_pdf as pdf
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Set Arial font
font_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/clustering_analysis/arial.ttf'
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()

# Add utils to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
from logging_utils import (print_section_header, print_step, print_success, 
                           print_warning, print_error, print_info, print_completion)

# ============================================================================
# PRE-CONFIGURED PATHS (NO ARGUMENTS NEEDED)
# ============================================================================
DATASET = "cmihbn_td"
PKLZ_DIR = "/oak/stanford/groups/menon/deriveddata/public/cmihbn/restfmri/timeseries/group_level/brainnetome/normz"
IG_CSV = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/integrated_gradients/cmihbn_td_features_all_sites_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS_single_model_predictions.csv"
C3SR_FILE = "/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/prepare_data/adhd/C3SR.csv"
OUTPUT_DIR = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/cmihbn_td"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_cmihbn_pklz_data(pklz_dir, c3sr_file):
    """Load CMI-HBN .pklz files (run1 only) and merge with C3SR behavioral data."""
    print_step("Loading CMI-HBN TD data", f"From {Path(pklz_dir).name}")
    
    # Load all run1 .pklz files
    pklz_files = [f for f in Path(pklz_dir).glob('*.pklz') if 'run1' in f.name]
    
    if not pklz_files:
        raise ValueError(f"No run1 .pklz files found in {pklz_dir}")
    
    print_info(f"Found {len(pklz_files)} run1 .pklz files")
    
    # Load and concatenate
    data_list = []
    for pklz_file in pklz_files:
        data_new = pd.read_pickle(pklz_file)
        data_list.append(data_new)
    
    data = pd.concat(data_list, ignore_index=True)
    print_info(f"Loaded {len(data)} total subjects from {len(pklz_files)} files")
    
    # Convert subject_id to string
    data['subject_id'] = data['subject_id'].astype(str)
    
    # Remove duplicates
    data = data.drop_duplicates(subset='subject_id', keep='first')
    print_info(f"Total subjects after deduplication: {len(data)}")
    
    # CMI-HBN specific filtering
    # Check if label column exists
    if 'label' not in data.columns:
        print_warning("No 'label' column found. Assuming all subjects are TD.")
        td_data = data
    else:
        # Debug: Check label values before filtering
        print_info(f"Label column dtype: {data['label'].dtype}")
        unique_labels = data['label'].unique()
        print_info(f"Unique label values: {unique_labels}")
        print_info(f"Non-null labels: {data['label'].notna().sum()}")
        
        # Filter for TD subjects based on label format
        # Check if labels are strings (e.g., 'td', 'asd') or numeric (0, 1)
        if data['label'].dtype == 'object':
            # String labels: filter for 'td' (case-insensitive)
            td_data = data[data['label'].str.lower() == 'td']
            print_info(f"TD subjects (label='td'): {len(td_data)}")
        else:
            # Numeric labels: filter for 0
            # First remove 'pending' labels if they exist
            data = data[data['label'] != 'pending']
            print_info(f"After removing 'pending' labels: {len(data)}")
            
            # Convert label to numeric
            data['label'] = pd.to_numeric(data['label'], errors='coerce')
            
            # Filter out NaN labels
            data = data[~data['label'].isna()]
            print_info(f"Valid subjects (non-NaN labels): {len(data)}")
            
            # Filter out label == 99
            data = data[data['label'] != 99]
            print_info(f"Valid subjects (label != 99): {len(data)}")
            
            # Filter for TD subjects (label == 0)
            td_data = data[data['label'] == 0]
            print_info(f"TD subjects (label=0): {len(td_data)}")
    
    # Filter by mean_fd < 0.5
    td_data = td_data[td_data['mean_fd'] < 0.5]
    print_info(f"After mean_fd < 0.5 filter: {len(td_data)}")
    
    # Load C3SR behavioral data
    print_step("Loading C3SR behavioral data", f"From {Path(c3sr_file).name}")
    
    if not Path(c3sr_file).exists():
        raise ValueError(f"C3SR file not found: {c3sr_file}")
    
    c3sr = pd.read_csv(c3sr_file)
    
    # Identify subject ID column in C3SR
    id_col = None
    for col in c3sr.columns:
        if 'id' in col.lower() or 'identifier' in col.lower():
            id_col = col
            break
    
    if id_col is None:
        raise ValueError("No subject ID column found in C3SR CSV")
    
    # Process C3SR subject IDs (take first 12 characters)
    c3sr['subject_id'] = c3sr[id_col].apply(lambda x: str(x)[:12])
    
    # Identify C3SR/Conners behavioral columns
    # Look for various patterns: C3SR_HY_T, C3SR_IN_T, or Conners subscales
    behavioral_cols = []
    
    # First try C3SR naming
    for col in c3sr.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in ['c3sr_hy_t', 'c3sr_in_t', 'hyperactivity', 'inattention']):
            behavioral_cols.append(col)
    
    # If no C3SR columns, look for Conners subscales (T-scores)
    if not behavioral_cols:
        for col in c3sr.columns:
            if ('_t' in col.lower() or 't_score' in col.lower() or 'tscore' in col.lower()) and \
               any(keyword in col.lower() for keyword in ['hyperactiv', 'inattent', 'adhd']):
                behavioral_cols.append(col)
    
    if not behavioral_cols:
        print_warning("No behavioral columns found in C3SR/Conners file")
        print_info(f"Available columns: {list(c3sr.columns)[:10]}...")
        return td_data, []
    
    print_info(f"Behavioral columns: {behavioral_cols}")
    
    # Merge TD data with C3SR
    c3sr_subset = c3sr[['subject_id'] + behavioral_cols]
    merged = pd.merge(td_data, c3sr_subset, on='subject_id', how='inner')
    
    print_info(f"Subjects with C3SR data: {len(merged)}")
    
    # Convert behavioral columns to numeric
    for col in behavioral_cols:
        merged[col] = pd.to_numeric(merged[col], errors='coerce')
    
    # Debug: Check how many subjects have data for each measure
    for col in behavioral_cols:
        non_null = merged[col].notna().sum()
        print_info(f"  {col}: {non_null} non-null values")
    
    # DON'T drop rows with NaN - we'll handle each behavioral measure separately
    # Just return the data with behavioral columns (some may have NaN)
    print_success(f"Final TD subjects: {len(merged)}")
    print_info(f"Note: Each behavioral measure will be analyzed separately with available data")
    
    return merged, behavioral_cols


def load_cmihbn_ig_scores(ig_csv):
    """Load CMI-HBN IG scores."""
    print_step("Loading IG scores", f"From {Path(ig_csv).name}")
    
    df = pd.read_csv(ig_csv)
    
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
    
    return df, roi_cols


def merge_data(pklz_data, ig_df, behavioral_cols):
    """Merge PKLZ and IG data by subject ID."""
    print_step("Merging data", "Matching subject IDs")
    
    # Select relevant columns from pklz_data
    pklz_cols = ['subject_id'] + behavioral_cols
    pklz_subset = pklz_data[pklz_cols]
    
    print_info(f"PKLZ subjects: {len(pklz_subset)}")
    print_info(f"IG subjects: {len(ig_df)}")
    
    # Merge on subject_id
    merged = pd.merge(ig_df, pklz_subset, on='subject_id', how='inner')
    
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
    """Create scatter plot of predicted vs actual behavioral scores."""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Scatter plot - match brain age plot colors (darker blue/purple)
    ax.scatter(y_actual, y_pred, alpha=0.7, s=80, color='#5A6FA8', edgecolors='#5A6FA8', linewidth=1)
    
    # Add best fit line - match brain age plot (red)
    z = np.polyfit(y_actual, y_pred, 1)
    p = np.poly1d(z)
    x_line = np.linspace(y_actual.min(), y_actual.max(), 100)
    ax.plot(x_line, p(x_line), color='#D32F2F', linewidth=2.5, alpha=0.9)
    
    # Format p-value
    if p_value < 0.001:
        p_str = "< 0.001"
    else:
        p_str = f"= {p_value:.3f}"
    
    # Add statistics text (bottom right, NO bounding box)
    stats_text = f"R = {rho:.3f}\nP {p_str}"
    ax.text(0.95, 0.05, stats_text, transform=ax.transAxes,
            fontsize=14, verticalalignment='bottom', horizontalalignment='right')
    
    # Labels and title
    ax.set_xlabel('Observed Behavioral Score', fontsize=14, fontweight='normal')
    ax.set_ylabel('Predicted Behavioral Score', fontsize=14, fontweight='normal')
    
    # Use dataset name as title (e.g., "CMIHBN-TD")
    title = dataset_name.replace('_', '-').upper()
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    
    # Styling - clean minimal style, NO top/right spines
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Ensure all ticks are present on both axes
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', labelsize=12, direction='out', 
                  length=6, width=1.5, top=False, right=False)
    ax.tick_params(axis='both', which='minor', direction='out', 
                  length=3, width=1, top=False, right=False)
    
    plt.tight_layout()
    
    # Save PNG and AI
    safe_name = behavioral_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
    output_path = Path(output_dir) / f'scatter_{safe_name}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Save as .ai file
    ai_path = Path(output_dir) / f'scatter_{safe_name}.ai'
    pdf.FigureCanvas(fig).print_pdf(str(ai_path))
    
    plt.close()


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
    print_section_header("ENHANCED BRAIN-BEHAVIOR ANALYSIS - CMI-HBN TD")
    
    print_info(f"PKLZ DIR:   {PKLZ_DIR}")
    print_info(f"IG CSV:     {IG_CSV}")
    print_info(f"C3SR FILE:  {C3SR_FILE}")
    print_info(f"Output:     {OUTPUT_DIR}")
    print()
    
    # Create output directory if it doesn't exist
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Load data
        pklz_data, behavioral_cols = load_cmihbn_pklz_data(PKLZ_DIR, C3SR_FILE)
        
        if not behavioral_cols:
            raise ValueError("No behavioral columns found")
        
        ig_df, roi_cols = load_cmihbn_ig_scores(IG_CSV)
        
        # 2. Merge data
        merged_df = merge_data(pklz_data, ig_df, behavioral_cols)
        
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
        
        # 6. Perform linear regression for each behavioral measure
        print_section_header("LINEAR REGRESSION RESULTS")
        results_list = []
        for behavioral_col in behavioral_cols:
            print()
            print_step(f"Analyzing {behavioral_col}", "Predicted vs Actual Behavioral Scores")
            behavioral_scores = merged_df[behavioral_col].values
            result = perform_linear_regression(pca_scores_optimal, behavioral_scores, behavioral_col, OUTPUT_DIR)
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
        print_completion("CMI-HBN TD Enhanced Analysis Complete!")
        print_info(f"Results saved to: {OUTPUT_DIR}")
        
    except Exception as e:
        print()
        print_error(f"Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
