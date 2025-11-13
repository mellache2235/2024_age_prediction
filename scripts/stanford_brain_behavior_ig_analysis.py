#!/usr/bin/env python3
"""
Stanford ASD Brain-Behavior Analysis Using Integrated Gradient Features

Loads pre-computed IG features and performs PCA + regression to predict
SRS behavioral scores.
"""

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf
from matplotlib import font_manager
import numpy as np
import os
import pandas as pd
import scipy.stats as spss
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# Constants
N_PCA_COMPONENTS = 50
RANDOM_SEED = 0
FONT_PATH = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/clustering_analysis/arial.ttf'

# File paths
IG_CSV = 'stanford_asd_features_IG_convnet_regressor_trained_on_hcp_dev_wIDS.csv'
SRS_CSV = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/prepare_data/stanford/SRS_data_20230925.csv'
ACTUAL_AGES_NPZ = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/stanford_updated/actual_stanford_asd_ages_oct25.npz'
PREDICTED_AGES_NPZ = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/stanford_updated/predicted_stanford_asd_ages_oct25.npz'
OUTPUT_DIR = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/stanford_asd_ig_analysis'

# Setup font
try:
    font_manager.fontManager.addfont(FONT_PATH)
    prop = font_manager.FontProperties(fname=FONT_PATH)
    plt.rcParams['font.family'] = prop.get_name()
except:
    pass


def perform_bag_behavior_correlation(merged_df, srs_cols, output_dir):
    """Correlate Brain Age Gap with SRS scores."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    if 'BAG' not in merged_df.columns:
        print("  ⚠︎ No BAG column")
        return
    
    print(f"\n{'='*60}")
    print(f"BAG-SRS Correlations: N={len(merged_df)}")
    print(f"{'='*60}")
    
    for srs_col in srs_cols:
        if srs_col not in merged_df.columns:
            continue
        
        valid_mask = ~np.isnan(merged_df['BAG']) & ~np.isnan(merged_df[srs_col])
        bag_clean = merged_df.loc[valid_mask, 'BAG'].values
        srs_clean = merged_df.loc[valid_mask, srs_col].values
        
        if len(bag_clean) < 10:
            continue
        
        corr, pvalue = spss.spearmanr(bag_clean, srs_clean)
        print(f"  {srs_col}: ρ={corr:.3f}, p={pvalue:.4f}")
        
        if pvalue < 0.05:
            import seaborn as sns
            fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=300)
            sns.set_style("white")
            sns.regplot(
                x=bag_clean, y=srs_clean, ci=None,
                scatter_kws={'color': 'navy', 'alpha': 0.6, 's': 40, 'edgecolor': 'w', 'linewidth': 0.5},
                line_kws={'color': 'red', 'linewidth': 2},
                ax=ax
            )
            
            p_text = r"$\mathit{P} < 0.001$" if pvalue < 0.001 else rf"$\mathit{{P}} = {pvalue:.3f}$"
            ax.text(0.95, 0.05, f"$\mathit{{R}}$ = {corr:.3f}\n{p_text}",
                   transform=ax.transAxes, ha='right', va='bottom', fontsize=12)
            
            ax.spines[['right', 'top']].set_visible(False)
            ax.set_xlabel("Brain Age Gap", fontsize=15, labelpad=10)
            ax.set_ylabel(f"{srs_col}", fontsize=15, labelpad=10)
            ax.set_title("Stanford ASD BAG-Behavior", fontsize=14, pad=10)
            plt.tight_layout(pad=1.2)
            
            safe_name = srs_col.replace('/', '_').replace(' ', '_')
            png_path = os.path.join(output_dir, f'stanford_BAG_{safe_name}_scatter.png')
            plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"    ✓ Saved: {png_path}")


def perform_brain_behavior_regression(features_df, srs_df, output_dir):
    """
    Perform PCA + LinearRegression to predict SRS scores from IG features.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Merge features with SRS by subject_id
    merged = features_df.merge(srs_df, on='subject_id', how='inner')
    
    if len(merged) < 10:
        print(f"  ⚠︎ Insufficient overlap: {len(merged)} subjects")
        return
    
    print(f"\nMerged: {len(merged)} subjects with IG + SRS data")
    
    # Get SRS columns
    srs_cols = [col for col in srs_df.columns if 'SRS' in col and col != 'subject_id']
    
    if not srs_cols:
        print("  ⚠︎ No SRS columns found")
        return
    
    print(f"  SRS measures: {len(srs_cols)}")
    
    # Extract ROI features
    roi_cols = [col for col in features_df.columns if col != 'subject_id'][:246]
    X = merged[roi_cols].values
    
    # Process each SRS measure
    for srs_col in srs_cols:
        if srs_col not in merged.columns:
            continue
            
        Y = merged[srs_col].values
        
        # Remove NaNs
        valid_mask = ~np.isnan(Y) & np.all(~np.isnan(X), axis=1)
        X_clean = X[valid_mask]
        Y_clean = Y[valid_mask]
        
        if len(Y_clean) < 10:
            continue
        
        # Standardize + PCA
        sc = StandardScaler()
        X_scaled = sc.fit_transform(X_clean)
        
        n_components = min(N_PCA_COMPONENTS, X_scaled.shape[0] - 1, X_scaled.shape[1])
        pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
        X_pca = pca.fit_transform(X_scaled)
        
        # Regression
        model = LinearRegression()
        model.fit(X_pca, Y_clean)
        predictions = model.predict(X_pca)
        
        # Statistics
        corr, pvalue = spss.spearmanr(Y_clean, predictions)
        
        print(f"\n  {srs_col}: ρ={corr:.3f}, p={pvalue:.4f}, N={len(Y_clean)}")
        
        if pvalue < 0.05:
            # Create scatter plot
            fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=300)
            sns.set_style("white")
            sns.regplot(
                x=Y_clean, y=predictions, ci=None,
                scatter_kws={'color': 'navy', 'alpha': 0.6, 's': 40, 'edgecolor': 'w', 'linewidth': 0.5},
                line_kws={'color': 'red', 'linewidth': 2},
                ax=ax
            )
            
            p_text = r"$\mathit{P} < 0.001$" if pvalue < 0.001 else rf"$\mathit{{P}} = {pvalue:.3f}$"
            ax.text(0.95, 0.05, f"$\mathit{{R}}$ = {corr:.3f}\n{p_text}",
                   transform=ax.transAxes, ha='right', va='bottom', fontsize=12)
            
            ax.spines[['right', 'top']].set_visible(False)
            ax.set_xlabel(f"Observed {srs_col}", fontsize=15, labelpad=10)
            ax.set_ylabel(f"Predicted {srs_col}", fontsize=15, labelpad=10)
            ax.set_title("Stanford ASD Brain-Behavior", fontsize=14, pad=10)
            plt.tight_layout(pad=1.2)
            
            safe_name = srs_col.replace('/', '_').replace(' ', '_')
            png_path = os.path.join(output_dir, f'stanford_{safe_name}_scatter.png')
            ai_path = os.path.join(output_dir, f'stanford_{safe_name}_scatter.ai')
            
            plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
            pdf.FigureCanvas(fig).print_pdf(ai_path)
            plt.close(fig)
            
            print(f"    ✓ Saved: {png_path}")


if __name__ == '__main__':
    print("="*80)
    print("Stanford ASD Brain-Behavior Analysis (IG Features)")
    print("="*80)
    
    # Load IG features
    print(f"\nLoading IG features from: {IG_CSV}")
    features_df = pd.read_csv(IG_CSV)
    features_df['subject_id'] = features_df['subject_id'].astype(str)
    
    # Load brain ages
    print(f"Loading brain ages...")
    actual_ages_data = np.load(ACTUAL_AGES_NPZ)
    predicted_ages_data = np.load(PREDICTED_AGES_NPZ)
    
    actual_ages = actual_ages_data['actual']
    predicted_ages = predicted_ages_data['predicted']
    
    # Align ages with IG
    min_n = min(len(actual_ages), len(predicted_ages), len(features_df))
    features_df = features_df.iloc[:min_n].copy()
    features_df['actual_age'] = actual_ages[:min_n]
    features_df['brain_age'] = predicted_ages[:min_n]
    features_df['BAG'] = features_df['brain_age'] - features_df['actual_age']
    
    # Load SRS behavioral data
    print(f"Loading SRS from: {SRS_CSV}")
    srs_df = pd.read_csv(SRS_CSV)
    srs_df['subject_id'] = srs_df['subject_id'].astype(str)
    
    # Merge
    merged = features_df.merge(srs_df, on='subject_id', how='inner')
    srs_cols = [col for col in srs_df.columns if 'SRS' in col and col != 'subject_id']
    
    print(f"\nMerged: {len(merged)} subjects with IG + SRS + ages")
    
    # Perform analyses
    perform_brain_behavior_regression(merged, srs_df, OUTPUT_DIR)
    perform_bag_behavior_correlation(merged, srs_cols, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

