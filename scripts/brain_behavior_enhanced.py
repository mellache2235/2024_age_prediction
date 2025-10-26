#!/usr/bin/env python3
"""
Enhanced brain-behavior analysis with:
- Elbow plot for optimal PC selection
- Scatter plots for PC-behavioral correlations
- PC loadings to show contributing regions
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Add utils
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir.parent / 'utils'))

from plotting_utils import save_figure, setup_fonts
from logging_utils import print_section_header, print_step, print_success, print_info

def find_elbow(variance_ratio):
    """Find elbow point in scree plot using the elbow method."""
    # Calculate the rate of change (second derivative)
    diffs = np.diff(variance_ratio)
    diffs2 = np.diff(diffs)
    
    # Find the point where the rate of change stabilizes
    # (first point where second derivative is close to 0)
    threshold = np.abs(diffs2).mean()
    elbow_idx = np.where(np.abs(diffs2) < threshold)[0]
    
    if len(elbow_idx) > 0:
        return elbow_idx[0] + 2  # +2 because of two np.diff operations
    return min(10, len(variance_ratio))  # Default to 10 or max available

def create_elbow_plot(pca, output_path):
    """Create elbow/scree plot for PCA variance."""
    setup_fonts()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    n_components = len(pca.explained_variance_ratio_)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    
    # Find elbow
    elbow = find_elbow(pca.explained_variance_ratio_)
    
    # Plot variance
    ax.plot(range(1, n_components + 1), pca.explained_variance_ratio_ * 100, 
            'bo-', linewidth=2, markersize=8, label='Individual')
    ax.plot(range(1, n_components + 1), cumvar * 100, 
            'ro-', linewidth=2, markersize=8, label='Cumulative')
    
    # Mark elbow
    ax.axvline(elbow, color='green', linestyle='--', linewidth=2, 
               label=f'Elbow (PC{elbow})')
    
    ax.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
    ax.set_ylabel('Variance Explained (%)', fontsize=12, fontweight='bold')
    ax.set_title('PCA Scree Plot - Elbow Method', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    save_figure(fig, output_path)
    plt.close()
    
    return elbow

def create_scatter_plot(pc_scores, behavioral_scores, pc_num, behavior_name, output_path):
    """Create scatter plot of PC vs behavioral scores."""
    setup_fonts()
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # Calculate Spearman correlation
    r, p = spearmanr(pc_scores, behavioral_scores)
    
    # Scatter plot
    ax.scatter(pc_scores, behavioral_scores, color='#1f77b4', 
              edgecolors='#1f77b4', alpha=0.7, s=50)
    
    # Regression line
    z = np.polyfit(pc_scores, behavioral_scores, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(pc_scores.min(), pc_scores.max(), 100)
    ax.plot(x_line, p_line(x_line), 'k-', alpha=0.8, linewidth=2)
    
    # Statistics in bottom right
    p_text = f"< 0.001" if p < 0.001 else f"= {p:.3f}"
    ax.text(0.95, 0.05,
            f"$\\rho$ = {r:.3f}\n$\\mathit{{P}}$ {p_text}",
            transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax.set_xlabel(f'PC{pc_num} Scores', fontsize=12, fontweight='bold')
    ax.set_ylabel(behavior_name, fontsize=12, fontweight='bold')
    ax.set_title(f'PC{pc_num} vs {behavior_name}', fontsize=14, fontweight='bold')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    save_figure(fig, output_path)
    plt.close()
    
    return r, p

def get_top_regions_per_pc(pca, roi_names, n_top=10):
    """Get top contributing regions for each PC."""
    loadings = pca.components_
    top_regions = {}
    
    for i, pc_loadings in enumerate(loadings):
        # Get absolute loadings
        abs_loadings = np.abs(pc_loadings)
        # Get top indices
        top_indices = np.argsort(abs_loadings)[-n_top:][::-1]
        
        top_regions[f'PC{i+1}'] = [
            {
                'region': roi_names[idx],
                'loading': pc_loadings[idx],
                'abs_loading': abs_loadings[idx]
            }
            for idx in top_indices
        ]
    
    return top_regions

# This would be integrated into the main brain_behavior scripts
# For now, this shows the structure of the enhancements needed

