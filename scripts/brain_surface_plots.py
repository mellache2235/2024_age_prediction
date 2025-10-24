#!/usr/bin/env python3
"""
Create brain surface plots for feature visualization.

This script generates 3D brain surface plots with colored overlays showing
feature importance, brain-behavior associations, or other quantitative measures.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import argparse

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / 'utils'))

from plotting_utils import setup_fonts, save_figure

try:
    from nilearn import plotting, surface, datasets
    from nilearn.image import load_img, math_img
    import nibabel as nib
    NILEARN_AVAILABLE = True
except ImportError:
    NILEARN_AVAILABLE = False
    print("Warning: nilearn not available. Install with: pip install nilearn")


def create_brain_surface_plot(feature_values: np.ndarray,
                            roi_indices: List[int],
                            atlas_nifti_path: str,
                            title: str = "Brain Feature Visualization",
                            colormap: str = 'Reds',
                            vmin: Optional[float] = None,
                            vmax: Optional[float] = None,
                            threshold: float = 0.0,
                            save_path: Optional[str] = None,
                            views: List[str] = ['lateral', 'medial'],
                            hemispheres: List[str] = ['left', 'right']) -> Optional[plt.Figure]:
    """
    Create brain surface plots with feature overlays.
    
    Args:
        feature_values (np.ndarray): Feature values to visualize
        roi_indices (List[int]): ROI indices corresponding to feature values
        atlas_nifti_path (str): Path to atlas NIfTI file
        title (str): Plot title
        colormap (str): Colormap for visualization
        vmin (float, optional): Minimum value for colormap
        vmax (float, optional): Maximum value for colormap
        threshold (float): Threshold for feature values
        save_path (str, optional): Path to save the figure
        views (List[str]): Views to show ('lateral', 'medial')
        hemispheres (List[str]): Hemispheres to show ('left', 'right')
        
    Returns:
        plt.Figure: The created brain surface plot
    """
    if not NILEARN_AVAILABLE:
        print("Error: nilearn not available. Cannot create brain surface plots.")
        return None
    
    try:
        # Load atlas
        atlas_img = load_img(atlas_nifti_path)
        atlas_data = atlas_img.get_fdata()
        
        # Create feature map
        feature_map = np.zeros_like(atlas_data)
        
        for roi_idx, value in zip(roi_indices, feature_values):
            if value > threshold:
                # Find voxels belonging to this ROI (ROIs are typically 1-indexed)
                roi_mask = atlas_data == (roi_idx + 1)
                feature_map[roi_mask] = value
        
        # Create NIfTI image from feature map
        feature_img = nib.Nifti1Image(feature_map, atlas_img.affine, atlas_img.header)
        
        # Set colormap limits
        if vmin is None:
            vmin = np.min(feature_values[feature_values > threshold])
        if vmax is None:
            vmax = np.max(feature_values)
        
        # Create brain surface plot
        fig = plt.figure(figsize=(16, 12), dpi=300)
        
        # Create subplots for different views
        n_views = len(views)
        n_hemispheres = len(hemispheres)
        
        plot_idx = 1
        for view in views:
            for hemisphere in hemispheres:
                ax = fig.add_subplot(n_views, n_hemispheres, plot_idx, projection='3d')
                
                # Create surface plot
                display = plotting.plot_stat_map(
                    feature_img,
                    display_mode='z',
                    cut_coords=1,
                    cmap=colormap,
                    vmin=vmin,
                    vmax=vmax,
                    threshold=threshold,
                    title=f"{hemisphere.capitalize()} {view.capitalize()} View",
                    axes=ax
                )
                
                plot_idx += 1
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            save_figure(fig, save_path, formats=['png', 'pdf'])
        
        return fig
        
    except Exception as e:
        print(f"Error creating brain surface plot: {e}")
        return None


def create_consensus_brain_surface_plot(consensus_counts: Dict[int, int],
                                      atlas_nifti_path: str,
                                      title: str = "Consensus Feature Map",
                                      normalize: bool = True,
                                      top_percentage: float = 0.2,
                                      save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Create brain surface plot from consensus feature counts.
    
    Args:
        consensus_counts (Dict[int, int]): Feature consensus counts
        atlas_nifti_path (str): Path to atlas NIfTI file
        title (str): Plot title
        normalize (bool): Whether to normalize counts
        top_percentage (float): Percentage of top features to show
        save_path (str, optional): Path to save the figure
        
    Returns:
        plt.Figure: The created brain surface plot
    """
    if not NILEARN_AVAILABLE:
        print("Error: nilearn not available. Cannot create brain surface plots.")
        return None
    
    # Get top features
    sorted_features = sorted(consensus_counts.items(), key=lambda x: x[1], reverse=True)
    n_top = max(1, int(len(sorted_features) * top_percentage))
    top_features = sorted_features[:n_top]
    
    # Extract values and indices
    roi_indices = [f[0] for f in top_features]
    feature_values = np.array([f[1] for f in top_features])
    
    # Normalize if requested
    if normalize and np.max(feature_values) > 0:
        feature_values = (feature_values / np.max(feature_values)) * 100
    
    # Create brain surface plot
    return create_brain_surface_plot(
        feature_values=feature_values,
        roi_indices=roi_indices,
        atlas_nifti_path=atlas_nifti_path,
        title=title,
        colormap='Reds',
        threshold=0.0,
        save_path=save_path
    )


def create_brain_behavior_surface_plot(correlation_values: np.ndarray,
                                     roi_indices: List[int],
                                     atlas_nifti_path: str,
                                     behavior_name: str = "Behavior",
                                     title: str = "Brain-Behavior Association",
                                     save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Create brain surface plot showing brain-behavior correlations.
    
    Args:
        correlation_values (np.ndarray): Correlation values
        roi_indices (List[int]): ROI indices
        atlas_nifti_path (str): Path to atlas NIfTI file
        behavior_name (str): Name of behavioral measure
        title (str): Plot title
        save_path (str, optional): Path to save the figure
        
    Returns:
        plt.Figure: The created brain surface plot
    """
    # Use absolute correlation values for visualization
    abs_correlations = np.abs(correlation_values)
    
    # Create brain surface plot
    return create_brain_surface_plot(
        feature_values=abs_correlations,
        roi_indices=roi_indices,
        atlas_nifti_path=atlas_nifti_path,
        title=f"{title}: {behavior_name}",
        colormap='RdBu_r',
        threshold=0.1,  # Only show correlations above 0.1
        save_path=save_path
    )


def create_multi_panel_brain_surface_plot(feature_data: Dict[str, np.ndarray],
                                        roi_indices: List[int],
                                        atlas_nifti_path: str,
                                        titles: Dict[str, str],
                                        save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Create multi-panel brain surface plots for different features.
    
    Args:
        feature_data (Dict[str, np.ndarray]): Dictionary of feature data
        roi_indices (List[int]): ROI indices
        atlas_nifti_path (str): Path to atlas NIfTI file
        titles (Dict[str, str]): Titles for each feature
        save_path (str, optional): Path to save the figure
        
    Returns:
        plt.Figure: Multi-panel brain surface plot
    """
    if not NILEARN_AVAILABLE:
        print("Error: nilearn not available. Cannot create brain surface plots.")
        return None
    
    n_features = len(feature_data)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16), dpi=300)
    axes = axes.flatten()
    
    try:
        # Load atlas
        atlas_img = load_img(atlas_nifti_path)
        atlas_data = atlas_img.get_fdata()
        
        for i, (feature_name, values) in enumerate(feature_data.items()):
            if i >= 4:  # Limit to 4 panels
                break
            
            # Create feature map
            feature_map = np.zeros_like(atlas_data)
            
            for roi_idx, value in zip(roi_indices, values):
                roi_mask = atlas_data == (roi_idx + 1)
                feature_map[roi_mask] = value
            
            # Create NIfTI image
            feature_img = nib.Nifti1Image(feature_map, atlas_img.affine, atlas_img.header)
            
            # Create surface plot
            display = plotting.plot_stat_map(
                feature_img,
                display_mode='z',
                cut_coords=1,
                cmap='Reds',
                threshold=0.0,
                title=titles.get(feature_name, feature_name),
                axes=axes[i]
            )
        
        # Hide unused subplots
        for i in range(n_features, 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            save_figure(fig, save_path, formats=['png', 'pdf'])
        
        return fig
        
    except Exception as e:
        print(f"Error creating multi-panel brain surface plot: {e}")
        return None


def main():
    """Main function for creating brain surface plots."""
    parser = argparse.ArgumentParser(
        description="Create brain surface plots for feature visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create consensus brain surface plot
  python brain_surface_plots.py \\
    --consensus_file results/consensus_data.npz \\
    --atlas_nifti /path/to/atlas.nii.gz \\
    --output_dir results/figures

  # Create brain-behavior surface plot
  python brain_surface_plots.py \\
    --correlation_file results/correlations.npz \\
    --atlas_nifti /path/to/atlas.nii.gz \\
    --behavior_name "Hyperactivity" \\
    --output_dir results/figures
        """
    )
    
    parser.add_argument("--consensus_file", type=str,
                       help="Path to consensus data file")
    parser.add_argument("--correlation_file", type=str,
                       help="Path to correlation data file")
    parser.add_argument("--atlas_nifti", type=str, required=True,
                       help="Path to atlas NIfTI file")
    parser.add_argument("--behavior_name", type=str, default="Behavior",
                       help="Name of behavioral measure")
    parser.add_argument("--output_dir", type=str, default="results/figures",
                       help="Output directory for figures")
    parser.add_argument("--plot_type", type=str, 
                       choices=['consensus', 'brain_behavior', 'both'],
                       default='both', help="Type of plot to create")
    parser.add_argument("--top_percentage", type=float, default=0.2,
                       help="Percentage of top features to show")
    
    args = parser.parse_args()
    
    if not NILEARN_AVAILABLE:
        print("Error: nilearn is required for brain surface plots.")
        print("Install with: pip install nilearn")
        sys.exit(1)
    
    # Setup fonts
    setup_fonts()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.plot_type in ['consensus', 'both']:
        if args.consensus_file:
            try:
                # Load consensus data
                consensus_data = np.load(args.consensus_file, allow_pickle=True)
                consensus_counts = consensus_data['consensus_counts'].item()
                
                # Create consensus brain surface plot
                fig = create_consensus_brain_surface_plot(
                    consensus_counts=consensus_counts,
                    atlas_nifti_path=args.atlas_nifti,
                    title="Consensus Feature Map",
                    top_percentage=args.top_percentage,
                    save_path=os.path.join(args.output_dir, "consensus_brain_surface.png")
                )
                
                if fig:
                    print(f"Consensus brain surface plot created and saved to: {args.output_dir}")
                
            except Exception as e:
                print(f"Error creating consensus brain surface plot: {e}")
        else:
            print("Warning: --consensus_file required for consensus plots")
    
    if args.plot_type in ['brain_behavior', 'both']:
        if args.correlation_file:
            try:
                # Load correlation data
                correlation_data = np.load(args.correlation_file, allow_pickle=True)
                correlations = correlation_data['correlations']
                roi_indices = correlation_data['roi_indices']
                
                # Create brain-behavior surface plot
                fig = create_brain_behavior_surface_plot(
                    correlation_values=correlations,
                    roi_indices=roi_indices,
                    atlas_nifti_path=args.atlas_nifti,
                    behavior_name=args.behavior_name,
                    save_path=os.path.join(args.output_dir, f"{args.behavior_name}_brain_surface.png")
                )
                
                if fig:
                    print(f"Brain-behavior surface plot created and saved to: {args.output_dir}")
                
            except Exception as e:
                print(f"Error creating brain-behavior surface plot: {e}")
        else:
            print("Warning: --correlation_file required for brain-behavior plots")


if __name__ == "__main__":
    main()
