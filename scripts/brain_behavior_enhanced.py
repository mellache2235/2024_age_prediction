#!/usr/bin/env python3
"""
Enhanced brain-behavior analysis with:
- Elbow plot for optimal PC selection
- Linear regression using all PCs to predict behavioral scores
- Scatter plots of predicted vs actual behavioral scores
- PC importance ranking (which PCs contribute most to predictions)
- PC loadings (which brain regions contribute to each PC)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Add utils
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir.parent / 'utils'))

from plotting_utils import save_figure, setup_fonts
from logging_utils import (print_section_header, print_step, print_success, 
                           print_info, print_warning, print_completion)


def find_elbow(variance_ratio):
    """Find elbow point in scree plot."""
    diffs = np.diff(variance_ratio)
    diffs2 = np.diff(diffs)
    threshold = np.abs(diffs2).mean()
    elbow_idx = np.where(np.abs(diffs2) < threshold)[0]
    
    if len(elbow_idx) > 0:
        return min(elbow_idx[0] + 2, len(variance_ratio))
    return min(10, len(variance_ratio))


def create_elbow_plot(pca, output_path):
    """Create elbow/scree plot."""
    setup_fonts()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    n_components = len(pca.explained_variance_ratio_)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    elbow = find_elbow(pca.explained_variance_ratio_)
    
    ax.plot(range(1, n_components + 1), pca.explained_variance_ratio_ * 100, 
            'bo-', linewidth=2, markersize=8, label='Individual')
    ax.plot(range(1, n_components + 1), cumvar * 100, 
            'ro-', linewidth=2, markersize=8, label='Cumulative')
    ax.axvline(elbow, color='green', linestyle='--', linewidth=2, 
               label=f'Elbow (PC{elbow}: {cumvar[elbow-1]*100:.1f}%)')
    
    ax.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
    ax.set_ylabel('Variance Explained (%)', fontsize=12, fontweight='bold')
    ax.set_title('PCA Scree Plot', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', direction='out', length=4)
    
    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close()
    
    return elbow


def predict_behavior_with_pcs(pca_scores, behavioral_scores, behavior_name):
    """
    Use all PCs in linear regression to predict behavioral scores.
    
    Returns:
        Dict with predictions, correlation, PC importances
    """
    # Train linear regression
    model = LinearRegression()
    model.fit(pca_scores, behavioral_scores)
    
    # Predict
    predicted = model.predict(pca_scores)
    
    # Calculate correlation between predicted and actual
    r, p = spearmanr(predicted, behavioral_scores)
    
    # Get PC importances (absolute coefficients)
    importances = np.abs(model.coef_)
    importance_order = np.argsort(importances)[::-1]
    
    # Cross-validation R²
    cv_scores = cross_val_score(model, pca_scores, behavioral_scores, 
                                cv=5, scoring='r2')
    
    return {
        'predicted': predicted,
        'actual': behavioral_scores,
        'spearman_r': r,
        'p_value': p,
        'coefficients': model.coef_,
        'intercept': model.intercept_,
        'importances': importances,
        'importance_order': importance_order,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std()
    }


def create_prediction_scatter(predicted, actual, behavior_name, r, p, output_path):
    """Create scatter plot of predicted vs actual behavioral scores."""
    setup_fonts()
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    ax.scatter(actual, predicted, color='#1f77b4', 
              edgecolors='#1f77b4', alpha=0.7, s=50)
    
    # Regression line
    z = np.polyfit(actual, predicted, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(actual.min(), actual.max(), 100)
    ax.plot(x_line, p_line(x_line), 'k-', alpha=0.8, linewidth=2)
    
    # Statistics
    p_text = f"< 0.001" if p < 0.001 else f"= {p:.3f}"
    ax.text(0.95, 0.05,
            f"$\\rho$ = {r:.3f}\n$\\mathit{{P}}$ {p_text}",
            transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax.set_xlabel(f'Actual {behavior_name}', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Predicted {behavior_name}', fontsize=12, fontweight='bold')
    ax.set_title(f'Brain-Behavior Prediction: {behavior_name}', fontsize=14, fontweight='bold')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', direction='out', length=4)
    
    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close()


def get_pc_loadings(pca, roi_names, n_top=10):
    """Get top contributing regions for each PC."""
    loadings_data = []
    
    for i in range(pca.n_components_):
        pc_loadings = pca.components_[i]
        abs_loadings = np.abs(pc_loadings)
        top_indices = np.argsort(abs_loadings)[-n_top:][::-1]
        
        row = {'PC': f'PC{i+1}'}
        for j, idx in enumerate(top_indices, 1):
            row[f'Region_{j}'] = roi_names[idx]
            row[f'Loading_{j}'] = pc_loadings[idx]
            row[f'Abs_Loading_{j}'] = abs_loadings[idx]
        
        loadings_data.append(row)
    
    return pd.DataFrame(loadings_data)


def save_pc_importance(importances, importance_order, output_path):
    """Save PC importance ranking."""
    df = pd.DataFrame({
        'PC': [f'PC{i+1}' for i in importance_order],
        'Importance': importances[importance_order],
        'Rank': range(1, len(importances) + 1)
    })
    
    df.to_csv(output_path, index=False)
    print_success(f"PC importance saved to {output_path}")
    
    return df


# Main function would load data, perform analysis, and create all outputs
print("Enhanced brain-behavior analysis module loaded.")
print("This script provides functions for:")
print("  • Elbow plot creation")
print("  • Linear regression with all PCs")
print("  • Prediction scatter plots")
print("  • PC importance ranking")
print("  • PC loadings (top brain regions)")
