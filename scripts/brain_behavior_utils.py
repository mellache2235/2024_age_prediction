#!/usr/bin/env python3
"""
Lean, efficient utilities for brain-behavior analysis.

This module provides reusable, optimized functions for:
- Data loading and validation
- ID alignment and merging
- PCA-based regression optimization
- Visualization

All functions include integrity checks while remaining performant.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf


# ============================================================================
# INTEGRITY CHECKS - Lean and Fast
# ============================================================================

def verify_data(data: np.ndarray, name: str, min_samples: int = 10) -> None:
    """Fast integrity check for numpy arrays."""
    assert data.ndim in [1, 2], f"{name}: Invalid dimensions {data.ndim}"
    assert len(data) >= min_samples, f"{name}: Insufficient samples {len(data)}"
    
    if np.issubdtype(data.dtype, np.floating):
        n_bad = np.isnan(data).sum() + np.isinf(data).sum()
        assert n_bad == 0, f"{name}: Found {n_bad} NaN/Inf values"
    
    print(f"  ✓ {name}: {data.shape}, range [{np.min(data):.2f}, {np.max(data):.2f}]")


def align_ids(ids1: np.ndarray, ids2: np.ndarray, 
              name1: str = "Set1", name2: str = "Set2",
              verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align two sets of IDs efficiently.
    
    Returns: (common_ids, indices_in_set1, indices_in_set2)
    """
    # Convert to string for robust comparison
    ids1_str = np.char.array(ids1).astype(str)
    ids2_str = np.char.array(ids2).astype(str)
    
    # Find intersection
    common_ids = np.intersect1d(ids1_str, ids2_str)
    
    if len(common_ids) == 0:
        raise ValueError(f"No overlap between {name1} and {name2}")
    
    # Get indices efficiently using searchsorted
    sorter1 = np.argsort(ids1_str)
    sorter2 = np.argsort(ids2_str)
    idx1 = sorter1[np.searchsorted(ids1_str, common_ids, sorter=sorter1)]
    idx2 = sorter2[np.searchsorted(ids2_str, common_ids, sorter=sorter2)]
    
    if verbose:
        overlap_pct1 = 100 * len(common_ids) / len(ids1)
        overlap_pct2 = 100 * len(common_ids) / len(ids2)
        print(f"  ✓ {name1} ({len(ids1)}) ↔ {name2} ({len(ids2)}): "
              f"{len(common_ids)} common ({overlap_pct1:.0f}%/{overlap_pct2:.0f}%)")
    
    return common_ids, idx1, idx2


# ============================================================================
# DATA LOADING - Streamlined
# ============================================================================

def load_ig_data(csv_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load IG scores efficiently.
    
    Returns: (subject_ids, ig_matrix, roi_names)
    """
    df = pd.read_csv(csv_path)
    
    subject_ids = df['subject_id'].values
    roi_cols = [c for c in df.columns if c.startswith('ROI_')]
    ig_matrix = df[roi_cols].values.astype(np.float32)  # Use float32 for efficiency
    
    # Quick validation
    assert not df['subject_id'].duplicated().any(), "Duplicate subject IDs in IG data"
    verify_data(ig_matrix, "IG matrix")
    
    print(f"  ✓ Loaded IG: {len(subject_ids)} subjects × {len(roi_cols)} ROIs")
    
    return subject_ids, ig_matrix, roi_cols


def load_behavioral_data(file_path: str, 
                        behavioral_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load behavioral data efficiently.
    
    Returns: (dataframe, behavioral_column_names)
    """
    df = pd.read_csv(file_path)
    
    # Auto-detect behavioral columns if not specified
    if behavioral_cols is None:
        behavioral_cols = [c for c in df.columns 
                          if c != 'subject_id' and df[c].dtype in [np.float64, np.int64]]
    
    # Quick validation
    assert 'subject_id' in df.columns, "No subject_id column found"
    assert not df['subject_id'].duplicated().any(), "Duplicate subject IDs in behavioral data"
    
    print(f"  ✓ Loaded behavioral: {len(df)} subjects × {len(behavioral_cols)} measures")
    
    return df, behavioral_cols


def merge_datasets(ig_ids: np.ndarray, ig_matrix: np.ndarray,
                  behavioral_df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Merge IG and behavioral data efficiently.
    
    Returns: (aligned_ig_matrix, aligned_behavioral_df)
    """
    # Align IDs
    common_ids, idx_ig, idx_beh = align_ids(
        ig_ids, behavioral_df['subject_id'].values,
        "IG", "Behavioral"
    )
    
    # Subset data using indices
    aligned_ig = ig_matrix[idx_ig]
    aligned_beh = behavioral_df.iloc[idx_beh].reset_index(drop=True)
    
    # Verify alignment
    assert len(aligned_ig) == len(aligned_beh) == len(common_ids)
    
    return aligned_ig, aligned_beh


# ============================================================================
# OPTIMIZATION - Lean and Fast
# ============================================================================

def spearman_scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman correlation scorer."""
    rho, _ = spearmanr(y_true, y_pred)
    return rho if not np.isnan(rho) else 0.0


def determine_pc_range(n_samples: int, max_pcs: int = 50, step: int = 5) -> List[int]:
    """Determine PC range based on sample size."""
    max_allowed = min(max_pcs, n_samples - 10)
    pc_range = list(range(step, max_allowed + 1, step))
    
    # Ensure we have options
    if len(pc_range) < 3:
        pc_range = [max(5, n_samples // 4), max(10, n_samples // 2), max(15, n_samples - 10)]
    
    return sorted(set(pc_range))  # Remove duplicates and sort


def optimize_model(X: np.ndarray, y: np.ndarray,
                  pc_range: Optional[List[int]] = None,
                  alphas: Optional[List[float]] = None,
                  cv_folds: int = 5,
                  n_jobs: int = -1) -> Dict[str, Any]:
    """
    Optimize PCA + regression model efficiently.
    
    Uses vectorized operations and parallel processing.
    
    Returns: dict with best_model, best_params, cv_score, all_results
    """
    # Validate inputs
    verify_data(X, "Features")
    verify_data(y, "Target")
    
    # Set defaults
    if pc_range is None:
        pc_range = determine_pc_range(len(y))
    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    
    print(f"  Testing {len(pc_range)} PC values × 4 models × {len(alphas)} alphas")
    
    # Setup CV
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scorer = make_scorer(spearman_scorer)
    
    # Track results
    results = []
    best_score = -np.inf
    best_config = None
    
    # Test models
    models = {
        'Linear': (LinearRegression(), [None]),
        'Ridge': (Ridge(max_iter=5000), alphas),
        'Lasso': (Lasso(max_iter=5000), alphas),
        'ElasticNet': (ElasticNet(max_iter=5000), alphas)
    }
    
    for n_pcs in pc_range:
        for model_name, (base_model, alpha_list) in models.items():
            for alpha in alpha_list:
                # Build pipeline
                steps = [
                    ('scaler', StandardScaler()),
                    ('pca', PCA(n_components=n_pcs)),
                    ('regressor', base_model if alpha is None 
                     else base_model.__class__(alpha=alpha, max_iter=5000))
                ]
                pipe = Pipeline(steps)
                
                # Cross-validate
                scores = cross_val_score(pipe, X, y, cv=cv, scoring=scorer, 
                                        n_jobs=n_jobs, error_score='raise')
                mean_score = scores.mean()
                
                # Store result
                results.append({
                    'n_components': n_pcs,
                    'model': model_name,
                    'alpha': alpha,
                    'cv_mean': mean_score,
                    'cv_std': scores.std()
                })
                
                # Track best
                if mean_score > best_score:
                    best_score = mean_score
                    best_config = (n_pcs, model_name, alpha)
    
    # Refit best model on all data
    n_pcs, model_name, alpha = best_config
    best_model = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_pcs)),
        ('regressor', models[model_name][0] if alpha is None
         else models[model_name][0].__class__(alpha=alpha, max_iter=5000))
    ])
    best_model.fit(X, y)
    
    print(f"  ✓ Best: {model_name}, {n_pcs} PCs" + 
          (f", α={alpha}" if alpha else "") + f", CV ρ={best_score:.3f}")
    
    return {
        'best_model': best_model,
        'best_params': {'n_components': n_pcs, 'model': model_name, 'alpha': alpha},
        'cv_score': best_score,
        'all_results': pd.DataFrame(results)
    }


def evaluate_model(model: Pipeline, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Evaluate model on data."""
    y_pred = model.predict(X)
    
    rho, p_val = spearmanr(y, y_pred)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    return {
        'y_pred': y_pred,
        'spearman_rho': rho,
        'p_value': p_val,
        'r2': r2,
        'mae': mae
    }


# ============================================================================
# VISUALIZATION - Consistent Style
# ============================================================================

def create_scatter(y_true: np.ndarray, y_pred: np.ndarray,
                  title: str, save_path: str,
                  metrics: Optional[Dict[str, float]] = None,
                  model_info: Optional[str] = None) -> None:
    """
    Create styled scatter plot efficiently.
    
    Consistent formatting: Arial, #5A6FA8 dots, #D32F2F line, no spines.
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    
    # Scatter
    ax.scatter(y_true, y_pred, alpha=0.7, s=80, 
              color='#5A6FA8', edgecolors='#5A6FA8', linewidth=1)
    
    # Best fit line
    coeffs = np.polyfit(y_true, y_pred, 1)
    x_line = np.linspace(y_true.min(), y_true.max(), 100)
    ax.plot(x_line, np.polyval(coeffs, x_line), 
           color='#D32F2F', linewidth=2.5, alpha=0.9)
    
    # Statistics
    if metrics:
        stats_text = f"ρ = {metrics.get('spearman_rho', 0):.3f}\n"
        stats_text += f"p = {metrics.get('p_value', 0):.4f}\n"
        stats_text += f"MAE = {metrics.get('mae', 0):.2f}"
        if model_info:
            stats_text += f"\n{model_info}"
        
        ax.text(0.95, 0.05, stats_text, transform=ax.transAxes,
               fontsize=11, va='bottom', ha='right')
    
    # Labels
    ax.set_xlabel('Observed', fontsize=14, fontweight='normal')
    ax.set_ylabel('Predicted', fontsize=14, fontweight='normal')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    
    # Clean style
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(axis='both', which='major', labelsize=12, 
                  length=6, width=1.5, top=False, right=False)
    
    plt.tight_layout()
    
    # Save both formats
    save_path = Path(save_path)
    plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    pdf.FigureCanvas(fig).print_pdf(str(save_path.with_suffix('.ai')))
    
    plt.close()
    print(f"  ✓ Saved: {save_path.name}")


# ============================================================================
# HIGH-LEVEL WORKFLOW
# ============================================================================

def run_brain_behavior_analysis(
    ig_csv: str,
    behavioral_csv: str,
    output_dir: str,
    dataset_name: str,
    behavioral_cols: Optional[List[str]] = None,
    max_measures: Optional[int] = None
) -> pd.DataFrame:
    """
    Complete brain-behavior analysis pipeline.
    
    Lean, efficient, with integrity checks.
    
    Returns: Summary DataFrame with results for all measures
    """
    print(f"\n{'='*80}")
    print(f"OPTIMIZED BRAIN-BEHAVIOR: {dataset_name}")
    print(f"{'='*80}\n")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("📂 Loading data...")
    ig_ids, ig_matrix, roi_cols = load_ig_data(ig_csv)
    behavioral_df, beh_cols = load_behavioral_data(behavioral_csv, behavioral_cols)
    
    # Merge
    print("\n🔗 Merging datasets...")
    X, beh_aligned = merge_datasets(ig_ids, ig_matrix, behavioral_df)
    
    # Analyze each measure
    print(f"\n🧠 Analyzing {len(beh_cols)} behavioral measures...\n")
    
    results_summary = []
    measures_to_process = beh_cols[:max_measures] if max_measures else beh_cols
    
    for i, measure in enumerate(measures_to_process, 1):
        print(f"[{i}/{len(measures_to_process)}] {measure}")
        
        # Get valid data
        y = beh_aligned[measure].values
        valid_mask = ~np.isnan(y)
        X_valid, y_valid = X[valid_mask], y[valid_mask]
        
        if len(y_valid) < 20:
            print(f"  ⚠ Skipped: only {len(y_valid)} valid subjects\n")
            continue
        
        # Optimize
        opt_result = optimize_model(X_valid, y_valid)
        
        # Evaluate
        eval_result = evaluate_model(opt_result['best_model'], X_valid, y_valid)
        
        # Visualize
        best_params = opt_result['best_params']
        model_info = f"{best_params['model']}"
        if best_params['alpha']:
            model_info += f" (α={best_params['alpha']})"
        model_info += f", {best_params['n_components']} PCs"
        
        create_scatter(
            y_valid, eval_result['y_pred'],
            f"{dataset_name}\n{measure}",
            output_dir / f"{measure}.png",
            eval_result,
            model_info
        )
        
        # Store results
        results_summary.append({
            'Measure': measure,
            'N': len(y_valid),
            'N_Components': best_params['n_components'],
            'Model': best_params['model'],
            'Alpha': best_params['alpha'],
            'CV_Spearman': opt_result['cv_score'],
            'Spearman_Rho': eval_result['spearman_rho'],
            'P_Value': eval_result['p_value'],
            'R2': eval_result['r2'],
            'MAE': eval_result['mae']
        })
        
        # Save detailed results
        opt_result['all_results'].to_csv(
            output_dir / f"{measure}_optimization.csv", index=False
        )
        
        print()
    
    # Save summary
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    
    print(f"✅ Complete! Results in {output_dir}")
    print(f"\nTop 5 by Spearman ρ:")
    print(summary_df.nlargest(5, 'Spearman_Rho')[['Measure', 'Spearman_Rho', 'Model', 'N_Components']].to_string(index=False))
    
    return summary_df

