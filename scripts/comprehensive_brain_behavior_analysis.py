#!/usr/bin/env python3
"""
Comprehensive brain-behavior correlation analysis for all datasets.

This script performs brain-behavior correlation analysis using IG scores for:
- NKI-RS TD (hyperactivity, inattention)
- ADHD-200 ADHD (HY, IN, by site)
- CMI-HBN ADHD (HY, IN)
- ADHD-200 TD
- CMI-HBN TD
- ABIDE ASD
- Stanford ASD
- HCP-Dev (discovery cohort)
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import logging
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import random

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / 'utils'))

from data_utils import load_finetune_dataset_w_ids, reshape_data
from statistical_utils import (
    multiple_correlation_analysis,
    apply_multiple_comparison_correction,
    benjamini_hochberg_correction
)
from plotting_utils import (
    plot_correlation_matrix,
    plot_brain_behavior_scatter,
    save_figure,
    setup_fonts
)


class ComprehensiveBrainBehaviorAnalyzer:
    """
    Comprehensive brain-behavior correlation analyzer for multiple datasets.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize analyzer.
        
        Args:
            config (Dict): Configuration dictionary
        """
        self.config = config
        self.results = {}
        
    def load_ig_scores_from_csv(self, csv_path: str) -> Tuple[np.ndarray, List[str]]:
        """Load IG scores from CSV file."""
        ig_df = pd.read_csv(csv_path)
        
        # Get ROI columns (exclude metadata)
        roi_columns = [col for col in ig_df.columns if col.lower() not in 
                      ['subject_id', 'participant_id', 'subid', 'sub_id', 'id', 'unnamed: 0']]
        
        ig_scores = ig_df[roi_columns].values
        
        # Get subject IDs with consistent formatting
        if 'subject_id' in ig_df.columns:
            subject_ids = ig_df['subject_id'].astype(str).str.strip().tolist()
        elif 'id' in ig_df.columns:
            subject_ids = ig_df['id'].astype(str).str.strip().tolist()
        else:
            subject_ids = [f"subj_{i}" for i in range(len(ig_df))]
        
        return ig_scores, subject_ids
    
    def load_behavioral_data(self, dataset_name: str, data_path: str, behavioral_columns: List[str]) -> pd.DataFrame:
        """Load behavioral data for a specific dataset."""
        if dataset_name == 'ADHD-200_ADHD' or dataset_name == 'ADHD-200_TD':
            return self._load_adhd200_data(data_path, behavioral_columns)
        elif dataset_name == 'CMI-HBN_ADHD' or dataset_name == 'CMI-HBN_TD':
            return self._load_cmihbn_data(data_path, behavioral_columns)
        elif dataset_name == 'ABIDE_ASD':
            return self._load_abide_data(data_path, behavioral_columns)
        elif dataset_name == 'Stanford_ASD':
            return self._load_stanford_data(data_path, behavioral_columns)
        elif dataset_name == 'NKI-RS_TD':
            return self._load_nki_data(data_path, behavioral_columns)
        elif data_path.endswith('.bin'):
            # For external datasets with .bin files, load from .bin and create behavioral data
            return self._load_bin_behavioral_data(data_path, behavioral_columns)
        else:
            return pd.DataFrame()
    
    def _load_adhd200_data(self, data_path: str, behavioral_columns: List[str]) -> pd.DataFrame:
        """Load ADHD-200 data."""
        data = np.load(data_path, allow_pickle=True)
        behavioral_data = pd.DataFrame(data.item())
        
        # Remove NaNs
        for col in behavioral_columns:
            if col in behavioral_data.columns:
                behavioral_data = behavioral_data.dropna(subset=[col])
        
        # Add subject_id
        if 'subject_id' not in behavioral_data.columns:
            behavioral_data['subject_id'] = behavioral_data.get('id', range(len(behavioral_data)))
        
        return behavioral_data[behavioral_columns + ['subject_id']]
    
    def _load_cmihbn_data(self, data_dir: str, behavioral_columns: List[str]) -> pd.DataFrame:
        """Load CMI-HBN data with C3SR behavioral data."""
        from os import listdir
        from os.path import isfile, join
        
        # Load C3SR behavioral data
        c3sr = pd.read_csv('/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/prepare_data/adhd/C3SR.csv')
        c3sr['Identifiers'] = c3sr['Identifiers'].apply(lambda x: x[0:12]).astype('str')
        
        # Load imaging data (run1 files only)
        files = [f for f in listdir(data_dir) if isfile(join(data_dir, f)) and 'run1' in f]
        
        data = None
        for file in files:
            file_data = np.load(data_dir + file, allow_pickle=True)
            data = pd.concat([data, file_data]) if data is not None else file_data
        
        # Filter data
        data['label'] = data['label'].astype(str).astype(int)
        df = data[(data['label'] != 99) & (data['mean_fd'] < 0.5)]
        
        # Merge with C3SR
        df['id_short'] = df['id'].astype(str).apply(lambda x: x[0:12])
        merged_data = df.merge(c3sr, left_on='id_short', right_on='Identifiers', how='inner')
        merged_data['subject_id'] = merged_data['id']
        
        return merged_data[behavioral_columns + ['subject_id']]
    
    def _load_abide_data(self, data_dir: str, behavioral_columns: List[str]) -> pd.DataFrame:
        """Load ABIDE data."""
        final_sites = ['NYU', 'SDSU', 'STANFORD', 'Stanford', 'TCD-1', 'UM', 'USM', 'Yale']
        
        # Filter files
        all_files = os.listdir(data_dir)
        filtered_files = [f for f in all_files if 'acompcor' in f and any(site in f for site in final_sites)]
        
        # Load and concatenate data
        appended_data = []
        for file_name in filtered_files:
            data = np.load(os.path.join(data_dir, file_name), allow_pickle=True)
            data = data[~pd.isna(data)]
            appended_data.append(data)
        
        df = pd.concat(appended_data)
        df = df[(df['label'] == 'asd') & (df['age'] <= 21)]
        
        # Add subject_id
        if 'subject_id' not in df.columns:
            df['subject_id'] = df.get('id', range(len(df)))
        
        return df[behavioral_columns + ['subject_id']]
    
    def _load_stanford_data(self, data_dir: str, behavioral_columns: List[str]) -> pd.DataFrame:
        """Load Stanford data."""
        SRS_file = pd.read_csv(data_dir + 'SRS_data_20230925.csv', skiprows=[0])
        SRS_file = SRS_file.drop_duplicates(subset=['record_id'], keep='last')
        
        # Extract IDs and SRS scores
        ids_2 = SRS_file['record_id'].astype('str')
        srs_score = SRS_file['srs_total_score_standard']
        
        stanford_data = pd.DataFrame({
            'subject_id': ids_2,
            'SRS_Total': srs_score
        })
        
        return stanford_data[behavioral_columns + ['subject_id']]
    
    def _load_nki_data(self, data_path: str, behavioral_columns: List[str]) -> pd.DataFrame:
        """Load NKI-RS data from CAARS CSV."""
        behavioral_data = pd.read_csv(data_path)
        
        # Ensure consistent ID format (remove any leading/trailing characters)
        if 'subject_id' in behavioral_data.columns:
            behavioral_data['subject_id'] = behavioral_data['subject_id'].astype(str).str.strip()
        elif 'id' in behavioral_data.columns:
            behavioral_data['subject_id'] = behavioral_data['id'].astype(str).str.strip()
        else:
            behavioral_data['subject_id'] = [f"subj_{i}" for i in range(len(behavioral_data))]
        
        return behavioral_data[behavioral_columns + ['subject_id']]
    
    def _load_hcp_dev_data(self, data_path: str, behavioral_columns: List[str]) -> pd.DataFrame:
        """Load HCP-Dev data."""
        if data_path.endswith('.csv'):
            behavioral_data = pd.read_csv(data_path)
        else:  # .pklz
            data = np.load(data_path, allow_pickle=True)
            behavioral_data = pd.DataFrame(data.item())
        
        # Add subject_id
        if 'subject_id' not in behavioral_data.columns:
            behavioral_data['subject_id'] = behavioral_data.get('id', range(len(behavioral_data)))
        
        return behavioral_data[behavioral_columns + ['subject_id']]
    
    def _load_bin_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Load .bin file data and concatenate train/test for external datasets."""
        import pickle
        
        with open(data_path, "rb") as fp:
            data_dict = pickle.load(fp)
        
        # Concatenate train and test data
        X_combined = np.concatenate([data_dict["X_train"], data_dict["X_test"]], axis=0)
        Y_combined = np.concatenate([data_dict["Y_train"], data_dict["Y_test"]], axis=0)
        
        # Get IDs if available
        if "id_train" in data_dict and "id_test" in data_dict:
            ids_combined = data_dict["id_train"] + data_dict["id_test"]
        else:
            ids_combined = [f"subj_{i}" for i in range(len(X_combined))]
        
        return X_combined, Y_combined, ids_combined
    
    def _load_bin_behavioral_data(self, data_path: str, behavioral_columns: List[str]) -> pd.DataFrame:
        """Load behavioral data from .bin file for external datasets."""
        import pickle
        
        with open(data_path, "rb") as fp:
            data_dict = pickle.load(fp)
        
        # Get IDs if available
        if "id_train" in data_dict and "id_test" in data_dict:
            ids_combined = data_dict["id_train"] + data_dict["id_test"]
        else:
            ids_combined = [f"subj_{i}" for i in range(len(data_dict["X_train"]) + len(data_dict["X_test"]))]
        
        # Create behavioral data DataFrame with age as primary behavioral measure
        behavioral_data = pd.DataFrame({
            'subject_id': ids_combined,
            'age': np.concatenate([data_dict["Y_train"], data_dict["Y_test"]])
        })
        
        # Add other behavioral columns if they exist in the data_dict
        for col in behavioral_columns:
            if col in data_dict:
                behavioral_data[col] = np.concatenate([data_dict[col + "_train"], data_dict[col + "_test"]])
        
        return behavioral_data[behavioral_columns + ['subject_id']]
    
    def analyze_dataset(self, 
                       dataset_name: str,
                       ig_csv_path: str,
                       behavioral_data_path: str,
                       behavioral_columns: List[str],
                       site_specific: bool = False) -> Dict:
        """
        Analyze brain-behavior correlations for a single dataset.
        
        Args:
            dataset_name (str): Name of the dataset
            ig_csv_path (str): Path to CSV file containing IG scores
            behavioral_data_path (str): Path to behavioral data
            behavioral_columns (List[str]): Behavioral measures to analyze
            site_specific (bool): Whether to perform site-specific analysis
            
        Returns:
            Dict: Analysis results
        """
        # Load data
        ig_scores, subject_ids = self.load_ig_scores_from_csv(ig_csv_path)
        behavioral_data = self.load_behavioral_data(dataset_name, behavioral_data_path, behavioral_columns)
        
        # Match subjects
        matched_data = self._match_subjects(ig_scores, subject_ids, behavioral_data)
        if matched_data is None:
            return {}
        
        matched_ig_scores, matched_behavioral_data = matched_data
        
        # Perform analysis
        if site_specific and 'site' in matched_behavioral_data.columns:
            results = self._analyze_site_specific(matched_ig_scores, matched_behavioral_data, behavioral_columns)
        else:
            results = self._analyze_overall(matched_ig_scores, matched_behavioral_data, behavioral_columns)
        
        results.update({
            'dataset_name': dataset_name,
            'n_subjects': len(matched_ig_scores),
            'behavioral_measures': behavioral_columns
        })
        
        return results
    
    def _match_subjects(self, 
                       ig_scores: np.ndarray, 
                       subject_ids: List[str], 
                       behavioral_data: pd.DataFrame) -> Optional[Tuple[np.ndarray, pd.DataFrame]]:
        """
        Match subjects between IG scores and behavioral data.
        
        Args:
            ig_scores (np.ndarray): IG scores
            subject_ids (List[str]): Subject IDs for IG scores
            behavioral_data (pd.DataFrame): Behavioral data
            
        Returns:
            Optional[Tuple[np.ndarray, pd.DataFrame]]: Matched data or None
        """
        ig_df = pd.DataFrame({'subject_id': subject_ids, 'ig_scores': ig_scores.tolist()})
        merged_data = ig_df.merge(behavioral_data, on='subject_id', how='inner')
        
        if len(merged_data) == 0:
            return None
        
        matched_ig_scores = np.array(merged_data['ig_scores'].tolist())
        return matched_ig_scores, merged_data
    
    def _analyze_overall(self, 
                        ig_scores: np.ndarray, 
                        behavioral_data: pd.DataFrame, 
                        behavioral_columns: List[str]) -> Dict:
        """
        Perform overall correlation analysis.
        
        Args:
            ig_scores (np.ndarray): IG scores
            behavioral_data (pd.DataFrame): Behavioral data
            behavioral_columns (List[str]): Behavioral measures
            
        Returns:
            Dict: Analysis results
        """
        # Apply PCA to IG scores (246 ROIs -> reduced components)
        pca_results = self._apply_pca_to_ig_scores(ig_scores, n_components=10)
        pca_components = pca_results['pca_components']
        
        results = {
            'overall_correlations': {},
            'corrected_p_values': {},
            'pca_results': pca_results
        }
        
        # Compute correlations for each behavioral measure using PCA components
        for measure in behavioral_columns:
            if measure not in behavioral_data.columns:
                continue
                
            # Get valid data (non-NaN)
            valid_mask = ~pd.isna(behavioral_data[measure])
            if valid_mask.sum() < 10:  # Need minimum sample size
                continue
            
            valid_pca = pca_components[valid_mask]
            valid_behavior = behavioral_data[measure][valid_mask].values
            
            # Compute correlations for each PCA component
            correlations = []
            p_values = []
            
            for comp_idx in range(valid_pca.shape[1]):
                comp_scores = valid_pca[:, comp_idx]
                
                # Pearson correlation
                r, p = pearsonr(comp_scores, valid_behavior)
                correlations.append(r)
                p_values.append(p)
            
            # Apply FDR correction
            corrected_p, _, _, _ = benjamini_hochberg_correction(np.array(p_values))
            
            results['overall_correlations'][measure] = {
                'correlations': np.array(correlations),
                'p_values': np.array(p_values),
                'corrected_p_values': corrected_p,
                'n_subjects': valid_mask.sum(),
                'pca_components_used': valid_pca.shape[1]
            }
        
        return results
    
    def _apply_pca_to_ig_scores(self, ig_scores: np.ndarray, n_components: int = 10) -> Dict:
        """
        Apply PCA to IG scores (246 ROIs) to reduce dimensionality.
        
        Args:
            ig_scores (np.ndarray): IG scores of shape (n_subjects, 246_ROIs)
            n_components (int): Number of PCA components to retain
            
        Returns:
            Dict: PCA results including components, explained variance, etc.
        """
        from sklearn.decomposition import PCA
        
        logging.info(f"Applying PCA to IG scores: {ig_scores.shape} -> {n_components} components")
        
        # Initialize PCA
        pca = PCA(n_components=n_components)
        
        # Fit and transform
        pca_components = pca.fit_transform(ig_scores)
        
        # Get explained variance
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        logging.info(f"PCA explained variance: {explained_variance_ratio[:5]} (first 5 components)")
        logging.info(f"Cumulative variance: {cumulative_variance[-1]:.3f} (total)")
        
        return {
            'pca_components': pca_components,
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance': cumulative_variance,
            'n_components': n_components,
            'pca_model': pca
        }
    
    def _analyze_site_specific(self, 
                              ig_scores: np.ndarray, 
                              behavioral_data: pd.DataFrame, 
                              behavioral_columns: List[str]) -> Dict:
        """
        Perform site-specific correlation analysis.
        
        Args:
            ig_scores (np.ndarray): IG scores
            behavioral_data (pd.DataFrame): Behavioral data
            behavioral_columns (List[str]): Behavioral measures
            
        Returns:
            Dict: Analysis results
        """
        results = {
            'site_specific_correlations': {},
            'overall_correlations': {}
        }
        
        # Get unique sites
        unique_sites = behavioral_data['site'].unique()
        
        for site in unique_sites:
            site_mask = behavioral_data['site'] == site
            if site_mask.sum() < 10:  # Need minimum sample size
                continue
            
            site_ig = ig_scores[site_mask]
            site_behavioral = behavioral_data[site_mask]
            
            site_results = self._analyze_overall(site_ig, site_behavioral, behavioral_columns)
            results['site_specific_correlations'][site] = site_results
        
        # Also compute overall correlations
        results['overall_correlations'] = self._analyze_overall(ig_scores, behavioral_data, behavioral_columns)
        
        return results
    
    def create_visualizations(self, results: Dict, output_dir: str) -> None:
        """
        Create visualizations for brain-behavior correlations.
        
        Args:
            results (Dict): Analysis results
            output_dir (str): Output directory
        """
        setup_fonts()
        os.makedirs(output_dir, exist_ok=True)
        
        for dataset_name, dataset_results in results.items():
            if not dataset_results or 'overall_correlations' not in dataset_results:
                continue
            
            # Create correlation matrix plot
            correlations_data = {}
            for measure, measure_results in dataset_results['overall_correlations'].items():
                correlations_data[measure] = measure_results['correlations']
            
            if correlations_data:
                fig = plot_correlation_matrix(
                    correlations_data,
                    title=f"{dataset_name} Brain-Behavior Correlations",
                    save_path=os.path.join(output_dir, f"{dataset_name}_correlation_matrix.png")
                )
            
            # Create scatter plots for significant correlations
            self._create_significant_correlation_plots(dataset_results, dataset_name, output_dir)
    
    def _create_significant_correlation_plots(self, 
                                            dataset_results: Dict, 
                                            dataset_name: str, 
                                            output_dir: str) -> None:
        """
        Create scatter plots for significant brain-behavior correlations.
        
        Args:
            dataset_results (Dict): Dataset results
            dataset_name (str): Dataset name
            output_dir (str): Output directory
        """
        # Find significant correlations (FDR corrected p < 0.05)
        significant_rois = set()
        
        for measure, measure_results in dataset_results['overall_correlations'].items():
            corrected_p = measure_results['corrected_p_values']
            significant_rois.update(np.where(corrected_p < 0.05)[0])
        
        if not significant_rois:
            logging.info(f"No significant correlations found for {dataset_name}")
            return
        
        # Create plots for top significant ROIs
        for roi_idx in list(significant_rois)[:5]:  # Top 5 ROIs
            for measure, measure_results in dataset_results['overall_correlations'].items():
                if roi_idx < len(measure_results['correlations']):
                    r = measure_results['correlations'][roi_idx]
                    p = measure_results['corrected_p_values'][roi_idx]
                    
                    if p < 0.05:  # Significant
                        # Create scatter plot (placeholder - would need actual data)
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.text(0.5, 0.5, 
                               f"ROI {roi_idx} - {measure}\nR = {r:.3f}, p = {p:.3f}",
                               ha='center', va='center', fontsize=14)
                        ax.set_title(f"{dataset_name} - ROI {roi_idx} vs {measure}")
                        
                        save_figure(fig, 
                                  os.path.join(output_dir, f"{dataset_name}_ROI_{roi_idx}_{measure}.png"))
                        plt.close()


def run_comprehensive_brain_behavior_analysis(config: Dict, output_dir: str) -> Dict:
    """
    Run comprehensive brain-behavior analysis for all datasets.
    
    Args:
        config (Dict): Configuration dictionary
        output_dir (str): Output directory
        
    Returns:
        Dict: Analysis results
    """
    logging.info("Starting comprehensive brain-behavior analysis...")
    
    # Initialize analyzer
    analyzer = ComprehensiveBrainBehaviorAnalyzer(config)
    
    # Define datasets and their behavioral measures
    datasets_config = {
        'NKI-RS_TD': {
            'behavioral_columns': ['CAARS_36', 'CAARS_37'],
            'site_specific': False
        },
        'ADHD-200_ADHD': {
            'behavioral_columns': ['Hyper/Impulsive', 'Inattentive'],
            'site_specific': True
        },
                'CMI-HBN_ADHD': {
                    'behavioral_columns': ['C3SR', 'C3SR_HY_T'],
                    'site_specific': False
                },
        'ADHD-200_TD': {
            'behavioral_columns': ['Hyper/Impulsive', 'Inattentive'],
            'site_specific': True
        },
        'CMI-HBN_TD': {
            'behavioral_columns': ['C3SR', 'C3SR_HY_T'],
            'site_specific': False
        },
        'ABIDE_ASD': {
            'behavioral_columns': ['ADOS_Total', 'ADOS_Social', 'ADOS_Comm'],
            'site_specific': False
        },
        'Stanford_ASD': {
            'behavioral_columns': ['SRS_Total'],
            'site_specific': False
        },
    }
    
    # Get data paths from config
    data_paths = config.get('data_paths', {})
    
    results = {}
    
    # Analyze each dataset
    for dataset_name, dataset_config in datasets_config.items():
        # Get data paths for this dataset
        dataset_paths = data_paths.get(dataset_name.lower().replace('-', '_'), {})
        
        if not dataset_paths:
            logging.warning(f"No data paths configured for {dataset_name}")
            continue
        
        # Get IG CSV path and behavioral data path
        ig_csv_path = dataset_paths.get('ig_csv')
        behavioral_data_path = dataset_paths.get('behavioral_data')
        
        if not ig_csv_path:
            logging.warning(f"No IG CSV path for {dataset_name}")
            continue
        
        if not behavioral_data_path:
            logging.warning(f"No behavioral data path for {dataset_name}")
            continue
        
        # Analyze dataset
        dataset_results = analyzer.analyze_dataset(
            dataset_name=dataset_name,
            ig_csv_path=ig_csv_path,
            behavioral_data_path=behavioral_data_path,
            behavioral_columns=dataset_config['behavioral_columns'],
            site_specific=dataset_config['site_specific']
        )
        
        if dataset_results:
            results[dataset_name] = dataset_results
    
    # Create visualizations
    if results:
        analyzer.create_visualizations(results, output_dir)
    
    # Save results
    import json
    with open(os.path.join(output_dir, 'comprehensive_brain_behavior_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logging.info("Comprehensive brain-behavior analysis completed")
    return results


def main():
    """Main function for comprehensive brain-behavior analysis."""
    parser = argparse.ArgumentParser(
        description="Comprehensive brain-behavior correlation analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run comprehensive analysis
  python comprehensive_brain_behavior_analysis.py \\
    --config config.yaml \\
    --output_dir results/brain_behavior

  # Analyze specific dataset
  python comprehensive_brain_behavior_analysis.py \\
    --dataset ADHD-200_ADHD \\
    --ig_dir results/ig_scores \\
    --behavioral_data /path/to/behavioral_data.csv \\
    --output_dir results/adhd200_adhd
        """
    )
    
    parser.add_argument("--config", type=str,
                       help="Path to configuration file")
    parser.add_argument("--dataset", type=str,
                       help="Specific dataset to analyze")
    parser.add_argument("--ig_dir", type=str,
                       help="Directory containing IG scores")
    parser.add_argument("--behavioral_data", type=str,
                       help="Path to behavioral data file")
    parser.add_argument("--output_dir", type=str, default="results/brain_behavior",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.config:
        # Load configuration and run comprehensive analysis
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        results = run_comprehensive_brain_behavior_analysis(config, args.output_dir)
        print(f"Comprehensive brain-behavior analysis completed. Results saved to: {args.output_dir}")
    
    elif args.dataset and args.ig_dir and args.behavioral_data:
        # Analyze specific dataset
        config = {
            'data_paths': {
                args.dataset.lower().replace('-', '_'): {
                    'behavioral_data': args.behavioral_data
                }
            },
            'feature_attribution': {
                'ig_dir': args.ig_dir
            }
        }
        
        analyzer = ComprehensiveBrainBehaviorAnalyzer(config)
        
        # Define behavioral columns based on dataset
        behavioral_columns_map = {
            'NKI-RS_TD': ['hyperactivity', 'inattention'],
            'ADHD-200_ADHD': ['Hyper/Impulsive', 'Inattentive'],
            'CMI-HBN_ADHD': ['hyperactivity', 'inattention'],
            'ADHD-200_TD': ['age', 'sex', 'iq'],
            'CMI-HBN_TD': ['age', 'sex', 'iq', 'adhd_symptoms', 'anxiety_symptoms', 'depression_symptoms'],
            'ABIDE_ASD': ['ADOS_Total', 'ADOS_Comm', 'ADOS_Social', 'ADOS_Repetitive', 'SRS_Total'],
            'Stanford_ASD': ['SRS_Total', 'age', 'sex', 'iq'],
            'HCP-Dev': ['age', 'sex']
        }
        
        behavioral_columns = behavioral_columns_map.get(args.dataset, ['age', 'sex'])
        site_specific = args.dataset in ['ADHD-200_ADHD', 'ADHD-200_TD']
        
        results = analyzer.analyze_dataset(
            dataset_name=args.dataset,
            ig_dir=args.ig_dir,
            behavioral_data_path=args.behavioral_data,
            behavioral_columns=behavioral_columns,
            site_specific=site_specific
        )
        
        # Save results
        import json
        with open(os.path.join(args.output_dir, f'{args.dataset}_results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Analysis completed for {args.dataset}. Results saved to: {args.output_dir}")
    
    else:
        print("Error: Either --config or --dataset with --ig_dir and --behavioral_data must be provided")
        sys.exit(1)


if __name__ == "__main__":
    main()
