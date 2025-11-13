#!/usr/bin/env python3
"""
ADHD-200 Brain-Behavior Analysis Using Integrated Gradient Features (NYU site only)

This script:
1. Loads pre-computed IG features from CSV
2. Merges with ADHD-200 behavioral data (Hyperactivity or Inattention scores)
3. Performs PCA + LinearRegression to predict behavior from IG features
4. Generates publication-ready scatter plots

Focuses on NYU site, excludes Peking.

Usage:
    python adhd200_brain_behavior_ig_analysis.py --measure HY   # Hyperactivity
    python adhd200_brain_behavior_ig_analysis.py --measure IN   # Inattention
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf
from matplotlib import font_manager
import numpy as np
import pandas as pd
import scipy.stats as spss
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Constants
N_PCA_COMPONENTS = 50
RANDOM_SEED = 0

# File paths
FONT_PATH = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/clustering_analysis/arial.ttf'
IG_CSV = 'adhd200_adhd_features_all_sites_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS_single_model_predictions.csv'
PKLZ_PATH = '/oak/stanford/groups/menon/deriveddata/public/adhd200/restfmri/timeseries/group_level/brainnetome/normz/adhd200_run-rest_brainnetome_mean_regMov-6param_wmcsf_dt1_bpf008-09_normz_246ROIs.pklz'
ACTUAL_AGES_NPZ = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/adhd200_updated/actual_adhd200_adhd_ages_oct25.npz'
PREDICTED_AGES_NPZ = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/adhd200_updated/predicted_adhd200_adhd_ages_oct25.npz'
OUTPUT_DIR = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/adhd200_adhd_ig_analysis'

# Setup Arial font
if font_manager.fontManager.addfont(FONT_PATH):
    prop = font_manager.FontProperties(fname=FONT_PATH)
    plt.rcParams['font.family'] = prop.get_name()


def perform_brain_behavior_regression(features_df_site, measure_col, site_name, output_dir):
    """
    Perform PCA + LinearRegression to predict behavioral measure from IG features.
    
    Args:
        features_df_site: DataFrame with IG features for specific site
        site_name: Name of site (e.g., 'NYU')
        output_dir: Directory to save outputs
    """
    # Drop metadata columns
    features_df_clean = features_df_site.drop(
        ['actual_age', 'brain_age', 'site', 'subject_id'],
        axis=1,
        errors='ignore'
    )
    
    # Separate features (246 ROIs) and target
    X = features_df_clean.iloc[:, :246]
    Y = np.asarray(features_df_clean[measure_col])
    
    # Remove any remaining NaNs
    valid_mask = ~np.isnan(Y)
    X = X[valid_mask]
    Y = Y[valid_mask]
    
    if len(Y) < 10:
        print(f"  ⚠︎ Insufficient subjects for {site_name} ({len(Y)}); skipping.")
        return
    
    print(f"\n{'='*60}")
    print(f"Analyzing {site_name}: N={len(Y)} subjects")
    print(f"{'='*60}")
    
    # Standardize features
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    
    # PCA dimensionality reduction
    n_components = min(N_PCA_COMPONENTS, X_scaled.shape[0] - 1, X_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"  PCA: {X.shape[1]} ROIs → {n_components} components")
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Linear regression
    model = LinearRegression()
    model.fit(X_pca, Y)
    predictions = model.predict(X_pca)
    
    # Compute statistics
    corr, pvalue = spss.spearmanr(Y, predictions)
    r_squared = corr ** 2
    
    print(f"  Spearman ρ = {corr:.3f}")
    print(f"  R² = {r_squared:.3f}")
    print(f"  P-value = {pvalue:.4f}")
    
    # Format p-value for plot
    if pvalue < 0.001:
        p_text = r"$\mathit{P} < 0.001$"
    else:
        p_text = rf"$\mathit{{P}} = {pvalue:.3f}$"
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=300)
    
    import seaborn as sns
    sns.set_style("white")
    sns.regplot(
        x=Y,
        y=predictions,
        ci=None,
        scatter_kws={'color': 'navy', 'alpha': 0.6, 's': 40, 'edgecolor': 'w', 'linewidth': 0.5},
        line_kws={'color': 'red', 'linewidth': 2},
        ax=ax
    )
    
    ax.text(
        0.95, 0.05,
        f"$\mathit{{R}}$ = {corr:.3f}\n{p_text}",
        transform=ax.transAxes,
        horizontalalignment='right',
        verticalalignment='bottom',
        fontsize=12
    )
    
    ax.spines[['right', 'top']].set_visible(False)
    measure_name = "Hyperactivity" if measure_col == "HY" else "Inattention"
    ax.set_xlabel(f"Observed {measure_name}", fontsize=15, labelpad=10)
    ax.set_ylabel(f"Predicted {measure_name}", fontsize=15, labelpad=10)
    ax.set_title(f"Brain-Behavior Analysis: ADHD-200 {site_name}", fontsize=14, pad=10)
    plt.tight_layout(pad=1.2)
    
    # Save outputs
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    png_path = os.path.join(output_dir, f'adhd200_features_{measure_col}_{site_name}_scatter.png')
    ai_path = os.path.join(output_dir, f'adhd200_features_{measure_col}_{site_name}_scatter.ai')
    
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    pdf.FigureCanvas(fig).print_pdf(ai_path)
    
    print(f"  ✓ Saved: {png_path}")
    print(f"  ✓ Saved: {ai_path}")
    
    plt.close(fig)
    
    # Save predictions
    results_df = pd.DataFrame({
        'Observed_HY': Y,
        'Predicted_HY': predictions
    })
    csv_path = os.path.join(output_dir, f'adhd200_features_HY_{site_name}_predictions.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved: {csv_path}")


def perform_bag_behavior_correlation(features_df_site, actual_ages, predicted_ages, site_name, output_dir):
    """
    Correlate Brain Age Gap with Hyperactivity scores.
    
    Alignment: assumes actual_ages and predicted_ages are in the same row order
    as the IG CSV (from fold_0.bin).
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure subject_id column exists for merging
    if 'subject_id' not in features_df_site.columns:
        print("  ⚠︎ No subject_id column; cannot align BAG with behavior.")
        return
    
    n_subjects = len(features_df_site)
    
    # Check alignment
    if len(actual_ages) != n_subjects or len(predicted_ages) != n_subjects:
        print(f"  ⚠︎ Age count mismatch: {len(actual_ages)} ages, {n_subjects} subjects")
        # Truncate to minimum
        min_n = min(len(actual_ages), len(predicted_ages), n_subjects)
        actual_ages = actual_ages[:min_n]
        predicted_ages = predicted_ages[:min_n]
        features_df_site = features_df_site.iloc[:min_n].copy()
        print(f"  Truncated to {min_n} subjects")
    
    # Compute Brain Age Gap
    bag = predicted_ages - actual_ages
    features_df_site['BAG'] = bag
    
    # Get HY scores
    if 'HY' not in features_df_site.columns:
        print("  ⚠︎ No HY column; cannot correlate BAG with behavior.")
        return
    
    valid_mask = ~np.isnan(features_df_site['HY']) & ~np.isnan(features_df_site['BAG'])
    bag_clean = features_df_site.loc[valid_mask, 'BAG'].values
    hy_clean = features_df_site.loc[valid_mask, 'HY'].values
    
    if len(bag_clean) < 10:
        print(f"  ⚠︎ Insufficient subjects for BAG analysis ({len(bag_clean)})")
        return
    
    # Compute correlation
    corr, pvalue = spss.spearmanr(bag_clean, hy_clean)
    
    print(f"\n{'='*60}")
    print(f"BAG-Behavior Correlation ({site_name}): N={len(bag_clean)}")
    print(f"  Spearman ρ = {corr:.3f}, p = {pvalue:.4f}")
    print(f"{'='*60}")
    
    if pvalue < 0.05:
        # Create scatter plot
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=300)
        sns.set_style("white")
        sns.regplot(
            x=bag_clean, y=hy_clean, ci=None,
            scatter_kws={'color': 'navy', 'alpha': 0.6, 's': 40, 'edgecolor': 'w', 'linewidth': 0.5},
            line_kws={'color': 'red', 'linewidth': 2},
            ax=ax
        )
        
        p_text = r"$\mathit{P} < 0.001$" if pvalue < 0.001 else rf"$\mathit{{P}} = {pvalue:.3f}$"
        ax.text(0.95, 0.05, f"$\mathit{{R}}$ = {corr:.3f}\n{p_text}",
               transform=ax.transAxes, ha='right', va='bottom', fontsize=12)
        
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_xlabel("Brain Age Gap", fontsize=15, labelpad=10)
        ax.set_ylabel("Hyperactivity", fontsize=15, labelpad=10)
        ax.set_title(f"BAG-Behavior: ADHD-200 {site_name}", fontsize=14, pad=10)
        plt.tight_layout(pad=1.2)
        
        png_path = os.path.join(output_dir, f'adhd200_BAG_HY_{site_name}_scatter.png')
        ai_path = os.path.join(output_dir, f'adhd200_BAG_HY_{site_name}_scatter.ai')
        
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
        pdf.FigureCanvas(fig).print_pdf(ai_path)
        plt.close(fig)
        
        print(f"  ✓ Saved BAG plot: {png_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ADHD-200 Brain-Behavior Analysis')
    parser.add_argument('--measure', choices=['HY', 'IN'], default='HY',
                       help='Behavioral measure: HY=Hyperactivity, IN=Inattention')
    args = parser.parse_args()
    
    measure_map = {
        'HY': 'Hyper/Impulsive',
        'IN': 'Inattentive'
    }
    measure_label = measure_map[args.measure]
    
    print("="*80)
    print(f"ADHD-200 Brain-Behavior Analysis (IG Features, NYU Site, {args.measure})")
    print("="*80)
    
    # Load IG features from pre-computed CSV
    print(f"\nLoading IG features from: {IG_CSV}")
    features_df = pd.read_csv(IG_CSV)
    features_df['subject_id'] = features_df['subject_id'].astype('str')
    
    # Load brain ages
    print(f"Loading brain ages from NPZ files...")
    actual_ages_data = np.load(ACTUAL_AGES_NPZ)
    predicted_ages_data = np.load(PREDICTED_AGES_NPZ)
    
    actual_ages = actual_ages_data['actual']
    predicted_ages = predicted_ages_data['predicted']
    
    print(f"  Actual ages: {len(actual_ages)} subjects")
    print(f"  Predicted ages: {len(predicted_ages)} subjects")
    print(f"  IG features: {len(features_df)} subjects")
    
    # Load PKLZ data to get site information
    print(f"Loading behavioral data from: {PKLZ_PATH}")
    hy_data = np.load(PKLZ_PATH, allow_pickle=True)
    hy_data['subject_id'] = hy_data['subject_id'].astype('str')
    hy_data['site'] = hy_data['site'].astype('str')
    hy_data = hy_data.drop_duplicates(subset='subject_id', keep='first')
    hy_data = hy_data.reset_index()
    
    # Align ages with IG features (assume same fold_0.bin order)
    min_n = min(len(actual_ages), len(predicted_ages), len(features_df))
    features_df = features_df.iloc[:min_n].copy()
    features_df['actual_age'] = actual_ages[:min_n]
    features_df['brain_age'] = predicted_ages[:min_n]
    
    # Merge site and behavioral scores
    features_df['site'] = features_df['subject_id'].map(hy_data.set_index('subject_id')['site'])
    features_df[args.measure] = features_df['subject_id'].map(hy_data.set_index('subject_id')[measure_label])
    features_df[args.measure] = features_df[args.measure].astype(float)
    
    # Remove missing scores
    features_df = features_df[features_df[args.measure] != -999.]
    
    print(f"\nTotal subjects with IG + {args.measure} + ages: {len(features_df)}")
    print(f"  NYU: {len(features_df[features_df['site'] == 'NYU'])}")
    print(f"  Peking: {len(features_df[features_df['site'] == 'Peking'])}")
    
    # Analyze NYU only
    features_df_nyu = features_df.loc[features_df['site'] == 'NYU', :].copy()
    
    if not features_df_nyu.empty:
        # Brain-behavior regression (IG features → behavior)
        perform_brain_behavior_regression(features_df_nyu, args.measure, 'NYU', OUTPUT_DIR)
        
        # BAG-behavior correlation
        perform_bag_behavior_correlation(
            features_df_nyu,
            features_df_nyu['actual_age'].values,
            features_df_nyu['brain_age'].values,
            'NYU',
            OUTPUT_DIR
        )
    else:
        print("  ✗ No NYU subjects found.")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

