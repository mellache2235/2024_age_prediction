#!/usr/bin/env python3
"""
Create radar charts for network analysis similar to the provided example.

This script generates radar charts showing brain region importance scores
across different networks or features.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / 'utils'))

from plotting_utils import setup_fonts, save_figure


def create_radar_chart(values: np.ndarray, 
                      labels: List[str],
                      title: str = "Network Analysis",
                      figsize: Tuple[int, int] = (10, 10),
                      color: str = 'turquoise',
                      alpha: float = 0.3,
                      line_color: str = 'darkblue',
                      line_width: float = 2.0,
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a radar chart for network analysis.
    
    Args:
        values (np.ndarray): Values for each axis
        labels (List[str]): Labels for each axis
        title (str): Chart title
        figsize (Tuple[int, int]): Figure size
        color (str): Fill color for the polygon
        alpha (float): Transparency of the fill
        line_color (str): Color of the polygon outline
        line_width (float): Width of the polygon outline
        save_path (str, optional): Path to save the figure
        
    Returns:
        plt.Figure: The created radar chart
    """
    # Number of variables
    N = len(labels)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Add the first value at the end to close the polygon
    values_plot = np.concatenate((values, [values[0]]))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    
    # Plot the polygon
    ax.plot(angles, values_plot, color=line_color, linewidth=line_width, label='Data')
    ax.fill(angles, values_plot, color=color, alpha=alpha)
    
    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    
    # Set y-axis limits
    ax.set_ylim(0, np.max(values) * 1.1)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Set title
    ax.set_title(title, size=16, fontweight='bold', pad=20)
    
    # Remove y-axis labels for cleaner look
    ax.set_yticklabels([])
    
    # Add value annotations at each point
    for angle, value, label in zip(angles[:-1], values, labels):
        ax.annotate(f'{value:.1f}', 
                   xy=(angle, value), 
                   xytext=(5, 5), 
                   textcoords='offset points',
                   fontsize=8,
                   ha='center')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_figure(fig, save_path, formats=['png', 'pdf'])
    
    return fig


def create_network_radar_from_consensus(consensus_counts: Dict[int, int],
                                      roi_labels: List[str],
                                      network_mapping: Dict[str, List[int]],
                                      top_n: int = 15,
                                      output_path: str = "results/figures/network_radar.png") -> plt.Figure:
    """
    Create radar chart from consensus feature data.
    
    Args:
        consensus_counts (Dict[int, int]): Feature consensus counts
        roi_labels (List[str]): ROI labels
        network_mapping (Dict[str, List[int]]): Network to feature mapping
        top_n (int): Number of top features to include
        output_path (str): Output path for the figure
        
    Returns:
        plt.Figure: Radar chart figure
    """
    # Get top features
    sorted_features = sorted(consensus_counts.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:top_n]
    
    # Extract values and labels
    feature_indices = [f[0] for f in top_features]
    values = np.array([f[1] for f in top_features])
    
    # Create labels (ROI names or network assignments)
    labels = []
    for idx in feature_indices:
        if idx < len(roi_labels):
            # Find which network this feature belongs to
            network = 'Unknown'
            for net, features in network_mapping.items():
                if idx in features:
                    network = net
                    break
            
            # Create label with ROI name and network
            roi_name = roi_labels[idx] if idx < len(roi_labels) else f"ROI_{idx}"
            labels.append(f"{roi_name}_{network}")
        else:
            labels.append(f"ROI_{idx}")
    
    # Normalize values to 0-100 scale for better visualization
    if np.max(values) > 0:
        values = (values / np.max(values)) * 100
    
    # Create radar chart
    fig = create_radar_chart(
        values=values,
        labels=labels,
        title="Top Brain Regions by Consensus Count",
        color='lightblue',
        alpha=0.3,
        line_color='darkblue',
        save_path=output_path
    )
    
    return fig


def create_network_comparison_radar(network_scores: Dict[str, float],
                                  title: str = "Network-level Analysis",
                                  output_path: str = "results/figures/network_comparison_radar.png") -> plt.Figure:
    """
    Create radar chart comparing network-level scores.
    
    Args:
        network_scores (Dict[str, float]): Network scores
        title (str): Chart title
        output_path (str): Output path for the figure
        
    Returns:
        plt.Figure: Radar chart figure
    """
    # Sort networks by score
    sorted_networks = sorted(network_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Extract values and labels
    networks, scores = zip(*sorted_networks)
    values = np.array(scores)
    
    # Normalize values to 0-100 scale
    if np.max(values) > 0:
        values = (values / np.max(values)) * 100
    
    # Create radar chart
    fig = create_radar_chart(
        values=values,
        labels=list(networks),
        title=title,
        color='lightgreen',
        alpha=0.3,
        line_color='darkgreen',
        save_path=output_path
    )
    
    return fig


def main():
    """Main function for creating radar plots."""
    parser = argparse.ArgumentParser(
        description="Create radar charts for network analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create radar chart from consensus data
  python create_radar_plots.py \\
    --consensus_file results/consensus_data.npz \\
    --roi_labels /path/to/roi_labels.txt \\
    --output_dir results/figures

  # Create network comparison radar chart
  python create_radar_plots.py \\
    --network_scores results/network_scores.csv \\
    --output_dir results/figures
        """
    )
    
    parser.add_argument("--consensus_file", type=str,
                       help="Path to consensus data file")
    parser.add_argument("--network_scores", type=str,
                       help="Path to network scores CSV file")
    parser.add_argument("--roi_labels", type=str,
                       help="Path to ROI labels file")
    parser.add_argument("--output_dir", type=str, default="results/figures",
                       help="Output directory for figures")
    parser.add_argument("--top_n", type=int, default=15,
                       help="Number of top features to include in radar chart")
    
    args = parser.parse_args()
    
    # Setup fonts
    setup_fonts()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.consensus_file and args.roi_labels:
        # Create radar chart from consensus data
        try:
            # Load consensus data
            consensus_data = np.load(args.consensus_file, allow_pickle=True)
            consensus_counts = consensus_data['consensus_counts'].item()
            network_mapping = consensus_data.get('network_mapping', {}).item()
            
            # Load ROI labels
            with open(args.roi_labels, 'r') as f:
                roi_labels = [line.strip() for line in f.readlines()]
            
            # Create radar chart
            fig = create_network_radar_from_consensus(
                consensus_counts,
                roi_labels,
                network_mapping,
                top_n=args.top_n,
                output_path=os.path.join(args.output_dir, "network_radar.png")
            )
            
            print(f"Radar chart created and saved to: {args.output_dir}/network_radar.png")
            
        except Exception as e:
            print(f"Error creating radar chart from consensus data: {e}")
            sys.exit(1)
    
    elif args.network_scores:
        # Create network comparison radar chart
        try:
            # Load network scores
            network_df = pd.read_csv(args.network_scores)
            network_scores = dict(zip(network_df['network'], network_df['score']))
            
            # Create radar chart
            fig = create_network_comparison_radar(
                network_scores,
                output_path=os.path.join(args.output_dir, "network_comparison_radar.png")
            )
            
            print(f"Network comparison radar chart created and saved to: {args.output_dir}/network_comparison_radar.png")
            
        except Exception as e:
            print(f"Error creating network comparison radar chart: {e}")
            sys.exit(1)
    
    else:
        print("Error: Must specify either --consensus_file with --roi_labels or --network_scores")
        sys.exit(1)


if __name__ == "__main__":
    main()
