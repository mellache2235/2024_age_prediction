#!/usr/bin/env python3
"""
Brain age prediction analysis with model training, bias correction, and external testing.

This script handles the complete brain age prediction pipeline including:
1. Training ConvNet models on HCP-Dev data
2. Bias correction using TD cohorts
3. Testing on external datasets
4. Performance evaluation and visualization
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr
import logging
import random

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / 'utils'))

from data_utils import load_finetune_dataset, load_finetune_dataset_w_ids, reshape_data
from model_utils import (
    ConvNetLightning, 
    train_lightning_model, 
    load_lightning_model_from_checkpoint,
    evaluate_model_performance, 
    comprehensive_brain_age_analysis,
    AgeScaler
)


class BrainAgePredictor:
    """
    Brain age prediction pipeline with bias correction.
    """
    
    def __init__(self, model_config: Dict, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize brain age predictor.
        
        Args:
            model_config (Dict): Model configuration parameters
            device (str): Device to use for training ('cuda' or 'cpu')
        """
        self.model_config = model_config
        self.device = device
        self.models = {}
        self.bias_params = {}
        
        # Log GPU availability
        if torch.cuda.is_available():
            logging.info(f"GPU available: {torch.cuda.get_device_name(0)}")
            logging.info(f"CUDA version: {torch.version.cuda}")
        else:
            logging.info("GPU not available, using CPU")
        
    def train_models(self, data_dir: str, num_folds: int = 5) -> Dict:
        """
        Train brain age prediction models using PyTorch Lightning with age scaling.
        
        This implements:
        1. 5-fold cross-validation for model evaluation
        2. Age scaling fitted on training data only
        3. Model checkpointing at best validation loss
        4. PyTorch Lightning training with early stopping
        
        Args:
            data_dir (str): Directory containing .bin files for training
            num_folds (int): Number of cross-validation folds
            
        Returns:
            Dict: Training results and model paths
        """
        logging.info("Training brain age prediction models using PyTorch Lightning with age scaling...")
        
        results = {
            'fold_results': [],
            'model_paths': [],
            'age_scalers': [],
            'training_metrics': {}
        }
        
        for fold in range(num_folds):
            logging.info(f"Training fold {fold + 1}/{num_folds}...")
            
            # Load fold data
            fold_path = os.path.join(data_dir, f"fold_{fold}.bin")
            if not os.path.exists(fold_path):
                logging.warning(f"Fold {fold} data not found at {fold_path}")
                continue
                
            try:
                # Load data
                X_train, X_test, Y_train, Y_test = load_finetune_dataset(fold_path)
                X_train = reshape_data(X_train)
                X_test = reshape_data(X_test)
                
                # Create age scaler and fit on training data only
                age_scaler = AgeScaler()
                age_scaler.fit(Y_train)
                
                # Scale ages
                Y_train_scaled = age_scaler.transform(Y_train)
                Y_test_scaled = age_scaler.transform(Y_test)
                
                # Create data loaders
                train_dataset = TensorDataset(
                    torch.FloatTensor(X_train), 
                    torch.FloatTensor(Y_train_scaled)
                )
                val_dataset = TensorDataset(
                    torch.FloatTensor(X_test), 
                    torch.FloatTensor(Y_test_scaled)
                )
                
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
                
                # Initialize PyTorch Lightning model
                model = ConvNetLightning(
                    input_channels=X_train.shape[1],
                    dropout_rate=self.model_config.get('dropout_rate', 0.6),
                    learning_rate=self.model_config.get('learning_rate', 0.001)
                )
                
                # Train model
                output_dir = os.path.join(data_dir, f"fold_{fold}_checkpoints")
                best_checkpoint = train_lightning_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    output_dir=output_dir,
                    max_epochs=self.model_config.get('num_epochs', 100),
                    patience=self.model_config.get('patience', 10)
                )
                
                # Save age scaler
                scaler_path = os.path.join(data_dir, f"age_scaler_fold_{fold}.pkl")
                age_scaler.save(scaler_path)
                
                # Store results
                results['fold_results'].append({
                    'fold': fold,
                    'model_path': best_checkpoint,
                    'scaler_path': scaler_path,
                    'train_ages': Y_train.tolist(),
                    'val_ages': Y_test.tolist()
                })
                results['model_paths'].append(best_checkpoint)
                results['age_scalers'].append(scaler_path)
                
                logging.info(f"Fold {fold} completed - Best checkpoint: {best_checkpoint}")
                
            except Exception as e:
                logging.error(f"Error training fold {fold}: {e}")
                continue
        
        logging.info("Model training completed!")
        return results
    
    def load_existing_models(self, model_dir: str, num_folds: int = 5) -> Dict:
        """
        Load existing trained models and age scalers.
        
        Args:
            model_dir (str): Directory containing trained models and scalers
            num_folds (int): Number of cross-validation folds
            
        Returns:
            Dict: Model paths and scaler paths
        """
        logging.info("Loading existing trained models and age scalers...")
        
        results = {
            'fold_results': [],
            'model_paths': [],
            'age_scalers': [],
            'training_metrics': {}
        }
        
        for fold in range(num_folds):
            # Look for model checkpoints
            checkpoint_dir = os.path.join(model_dir, f"fold_{fold}_checkpoints")
            scaler_path = os.path.join(model_dir, f"age_scaler_fold_{fold}.pkl")
            
            # Find best model checkpoint
            if os.path.exists(checkpoint_dir):
                import glob
                checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "best_model*.ckpt"))
                if checkpoint_files:
                    model_path = checkpoint_files[0]  # Use first match
                    
                    if os.path.exists(scaler_path):
                        results['fold_results'].append({
                            'fold': fold,
                            'model_path': model_path,
                            'scaler_path': scaler_path
                        })
                        results['model_paths'].append(model_path)
                        results['age_scalers'].append(scaler_path)
                        logging.info(f"Loaded fold {fold}: {model_path}")
                    else:
                        logging.warning(f"Age scaler not found for fold {fold}: {scaler_path}")
                else:
                    logging.warning(f"No model checkpoint found for fold {fold} in {checkpoint_dir}")
            else:
                logging.warning(f"Checkpoint directory not found for fold {fold}: {checkpoint_dir}")
        
        logging.info(f"Loaded {len(results['model_paths'])} existing models")
        return results
    
    def _get_default_hyperparameters(self) -> Dict:
        """Get default hyperparameters (no optimization)."""
        return {
            'learning_rate': 0.001,
            'dropout_rate': 0.6,
            'weight_decay': 1e-4,
            'batch_size': 32,
            'num_epochs': 100
        }
    
    def _compute_ensemble_metrics(self, fold_results: List[Dict]) -> Dict:
        """Compute ensemble metrics across folds."""
        all_maes = [fold['train_metrics'].get('final_mae', np.nan) for fold in fold_results]
        all_r2s = [fold['train_metrics'].get('final_r2', np.nan) for fold in fold_results]
        
        return {
            'mean_mae': np.nanmean(all_maes),
            'std_mae': np.nanstd(all_maes),
            'mean_r2': np.nanmean(all_r2s),
            'std_r2': np.nanstd(all_r2s),
            'n_folds': len(fold_results)
        }
    
    def fit_bias_correction(self, td_data: Dict[str, np.ndarray], 
                           dataset_type: str = "external_td") -> Dict:
        """
        Fit bias correction parameters using TD cohort data.
        
        Args:
            td_data (Dict): Dictionary with 'ages' and 'predictions' arrays
            dataset_type (str): Type of dataset ("hcp_dev", "external_td", "adhd_asd")
            
        Returns:
            Dict: Bias correction parameters
        """
        logging.info(f"Fitting bias correction parameters for {dataset_type}...")
        
        ages = td_data['ages']
        predictions = td_data['predictions']
        
        # Compute Brain Age Gap (BAG)
        bag = predictions - ages
        
        # Fit linear regression: BAG = alpha + beta * age
        reg = LinearRegression().fit(ages.reshape(-1, 1), bag.reshape(-1, 1))
        
        bias_params = {
            'alpha': float(reg.intercept_[0]),
            'beta': float(reg.coef_[0][0]),
            'n_subjects': len(ages),
            'r2': float(reg.score(ages.reshape(-1, 1), bag.reshape(-1, 1))),
            'dataset_type': dataset_type
        }
        
        self.bias_params = bias_params
        logging.info(f"Bias correction fitted for {dataset_type} - R²: {bias_params['r2']:.3f}")
        
        return bias_params
    
    def apply_bias_correction(self, ages: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """
        Apply bias correction to predictions.
        
        Args:
            ages (np.ndarray): True ages
            predictions (np.ndarray): Raw predictions
            
        Returns:
            np.ndarray: Bias-corrected predictions
        """
        if not self.bias_params:
            logging.warning("No bias correction parameters found. Returning raw predictions.")
            return predictions
        
        # Compute predicted BAG
        predicted_bag = self.bias_params['alpha'] + self.bias_params['beta'] * ages
        
        # Apply correction
        corrected_predictions = predictions - predicted_bag
        
        return corrected_predictions
    
    def test_on_external_data(self, 
                            data_path: str, 
                            dataset_name: str,
                            apply_bias_correction: bool = True) -> Dict:
        """
        Test trained models on external dataset.
        
        Args:
            data_path (str): Path to external dataset (.bin file)
            dataset_name (str): Name of the dataset
            apply_bias_correction (bool): Whether to apply bias correction
            
        Returns:
            Dict: Test results
        """
        logging.info(f"Testing on external dataset: {dataset_name}")
        
        try:
            # Load external data with IDs
            X_train, X_test, id_train, Y_train, Y_test, id_test = load_finetune_dataset_w_ids(data_path)
            
            # Combine train and test data for external testing
            X_test = np.concatenate([X_train, X_test], axis=0)
            Y_test = np.concatenate([Y_train, Y_test], axis=0)
            id_test = id_train + id_test
            
            X_test = reshape_data(X_test)
            
            # Make predictions using ensemble of models
            all_predictions = []
            
            for i, model_path in enumerate(self.models.get('model_paths', [])):
                if not os.path.exists(model_path):
                    continue
                
                # Load corresponding age scaler
                scaler_path = self.models.get('age_scalers', [])[i]
                if not os.path.exists(scaler_path):
                    logging.warning(f"Age scaler not found: {scaler_path}")
                    continue
                
                age_scaler = AgeScaler.load(scaler_path)
                    
                # Load PyTorch Lightning model from checkpoint
                model = load_lightning_model_from_checkpoint(
                    checkpoint_path=model_path,
                    input_channels=X_test.shape[1],
                    dropout_rate=self.model_config.get('dropout_rate', 0.6),
                    learning_rate=self.model_config.get('learning_rate', 0.001)
                )
                
                # Move model to device
                if torch.cuda.is_available():
                    model = model.cuda()
                model.eval()
                
                # Make predictions
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_test)
                    if torch.cuda.is_available():
                        X_tensor = X_tensor.cuda()
                    predictions_scaled = model(X_tensor).cpu().numpy().flatten()
                    
                    # Inverse transform predictions back to original age scale
                    predictions = age_scaler.inverse_transform(predictions_scaled)
                    all_predictions.append(predictions)
            
            if not all_predictions:
                raise ValueError("No valid models found for prediction")
            
            # Ensemble predictions
            ensemble_predictions = np.mean(all_predictions, axis=0)
            
            # Apply bias correction if requested
            if apply_bias_correction and self.bias_params:
                ensemble_predictions = self.apply_bias_correction(Y_test, ensemble_predictions)
            
            # Compute metrics
            metrics = evaluate_model_performance(Y_test, ensemble_predictions)
            
            # Create results
            results = {
                'dataset_name': dataset_name,
                'n_subjects': len(Y_test),
                'raw_predictions': ensemble_predictions,
                'true_ages': Y_test,
                'metrics': metrics,
                'bias_correction_applied': apply_bias_correction
            }
            
            logging.info(f"{dataset_name} testing completed - MAE: {metrics['mae']:.3f}, R²: {metrics['r2']:.3f}")
            
            return results
            
        except Exception as e:
            logging.error(f"Error testing on {dataset_name}: {e}")
            return {}
    


def run_brain_age_prediction_analysis(config: Dict, output_dir: str) -> Dict:
    """
    Run complete brain age prediction analysis.
    
    Args:
        config (Dict): Configuration dictionary
        output_dir (str): Output directory
        
    Returns:
        Dict: Analysis results
    """
    logging.info("Starting brain age prediction analysis...")
    
    # Initialize predictor
    model_config = config.get('model', {})
    predictor = BrainAgePredictor(model_config)
    
    results = {}
    
    # Step 1: Train models on HCP-Dev
    hcp_dev_dir = config.get('data_paths', {}).get('hcp_dev', {}).get('training_data_dir')
    if hcp_dev_dir:
        logging.info("Training models on HCP-Dev data...")
        training_results = predictor.train_models(hcp_dev_dir)
        results['training'] = training_results
        predictor.models = training_results
    
    # Step 2: Fit bias correction using different strategies
    # For HCP-Dev: Use all ages from 5-fold CV
    if 'training' in results and 'fold_results' in results['training']:
        logging.info("Fitting bias correction using HCP-Dev 5-fold results...")
        hcp_ages = []
        hcp_predictions = []
        
        for fold_result in results['training']['fold_results']:
            hcp_ages.extend(fold_result['true_ages'])
            hcp_predictions.extend(fold_result['predictions'])
        
        hcp_bias_params = predictor.fit_bias_correction({
            'ages': np.array(hcp_ages),
            'predictions': np.array(hcp_predictions)
        }, dataset_type="hcp_dev")
        results['hcp_bias_correction'] = hcp_bias_params
    
    # For external TD: Use entire TD dataset
    td_data_paths = config.get('data_paths', {}).get('td_cohorts', {})
    if td_data_paths:
        logging.info("Fitting bias correction using external TD cohorts...")
        
        # Combine TD data for bias correction
        all_ages = []
        all_predictions = []
        
        for cohort_name, data_path in td_data_paths.items():
            if os.path.exists(data_path):
                # Test on TD cohort to get predictions
                td_results = predictor.test_on_external_data(
                    data_path, cohort_name, apply_bias_correction=False
                )
                if td_results:
                    all_ages.extend(td_results['true_ages'])
                    all_predictions.extend(td_results['raw_predictions'])
        
        if all_ages and all_predictions:
            td_bias_params = predictor.fit_bias_correction({
                'ages': np.array(all_ages),
                'predictions': np.array(all_predictions)
            }, dataset_type="external_td")
            results['td_bias_correction'] = td_bias_params
    
    # Step 3: Test on external datasets
    external_datasets = config.get('data_paths', {}).get('external_datasets', {})
    test_results = {}
    
    for dataset_name, data_path in external_datasets.items():
        if os.path.exists(data_path):
            logging.info(f"Testing on {dataset_name}...")
            test_result = predictor.test_on_external_data(
                data_path, dataset_name, apply_bias_correction=True
            )
            if test_result:
                test_results[dataset_name] = test_result
    
    results['external_testing'] = test_results
    
    # Step 4: Comprehensive brain age analysis
    if test_results:
        logging.info("Performing comprehensive brain age analysis...")
        comprehensive_analysis = comprehensive_brain_age_analysis(test_results)
        results['comprehensive_analysis'] = comprehensive_analysis
        
        # Print summary of brain age gap analysis
        logging.info("=== BRAIN AGE GAP ANALYSIS SUMMARY ===")
        for dataset_name, metrics in comprehensive_analysis['individual_metrics'].items():
            bag_stats = metrics['bag_stats']
            logging.info(f"{dataset_name}: Mean BAG = {bag_stats['mean']:.3f} ± {bag_stats['std']:.3f} years (n={bag_stats['n']})")
        
        # Print group comparisons
        if comprehensive_analysis['group_comparisons']:
            logging.info("\n=== GROUP COMPARISONS ===")
            for comparison_name, comparison in comprehensive_analysis['group_comparisons'].items():
                t_test = comparison['t_test']
                logging.info(f"{comparison_name}:")
                logging.info(f"  {comparison['group1_name']}: Mean BAG = {comparison['group1_stats']['mean']:.3f} ± {comparison['group1_stats']['std']:.3f} (n={comparison['group1_stats']['n']})")
                logging.info(f"  {comparison['group2_name']}: Mean BAG = {comparison['group2_stats']['mean']:.3f} ± {comparison['group2_stats']['std']:.3f} (n={comparison['group2_stats']['n']})")
                logging.info(f"  Difference: {comparison['difference_in_means']:.3f} years")
                logging.info(f"  T-test: t={t_test['statistic']:.3f}, p={t_test['p_value']:.4f} ({'significant' if t_test['significant'] else 'not significant'})")
                logging.info(f"  Effect size (Cohen's d): {comparison['effect_size']['cohens_d']:.3f} ({comparison['effect_size']['interpretation']})")
        
        # Print overall summary
        if comprehensive_analysis['summary']:
            summary = comprehensive_analysis['summary']
            logging.info(f"\n=== OVERALL SUMMARY ===")
            logging.info(f"Overall Mean BAG: {summary['overall_mean_bag']:.3f} ± {summary['overall_std_bag']:.3f} years")
            logging.info(f"Overall Median BAG: {summary['overall_median_bag']:.3f} years")
            logging.info(f"Total N: {summary['total_n']}")
    
    # Step 5: Save comprehensive analysis results
    if 'comprehensive_analysis' in results:
        import json
        analysis_file = os.path.join(output_dir, 'comprehensive_brain_age_analysis.json')
        with open(analysis_file, 'w') as f:
            json.dump(results['comprehensive_analysis'], f, indent=2, default=str)
        logging.info(f"Comprehensive analysis results saved to: {analysis_file}")
    
    # Step 6: Note - Visualizations are now created using separate plotting scripts
    
    logging.info("Brain age prediction analysis completed")
    return results


def run_brain_age_prediction_analysis_with_existing_models(config: Dict, model_dir: str, output_dir: str) -> Dict:
    """
    Run brain age prediction analysis using existing trained models.
    
    Args:
        config (Dict): Configuration dictionary
        model_dir (str): Directory containing existing trained models
        output_dir (str): Output directory for results
        
    Returns:
        Dict: Analysis results
    """
    logging.info("Running brain age prediction analysis with existing models...")
    
    # Initialize predictor
    model_config = config.get('model_config', {})
    predictor = BrainAgePredictor(model_config)
    
    # Load existing models
    num_folds = config.get('num_folds', 5)
    training_results = predictor.load_existing_models(model_dir, num_folds)
    
    if not training_results['model_paths']:
        raise ValueError("No existing models found in the specified directory")
    
    # Store models in predictor for testing
    predictor.models = training_results
    
    # Create results structure
    results = {
        'analysis_type': 'existing_models',
        'model_directory': model_dir,
        'training_info': training_results,
        'external_testing': {},
        'comprehensive_analysis': {}
    }
    
    # Step 1: Fit bias correction using external TD cohorts
    td_data_paths = {}
    for dataset_name, data_path in config.get('data_paths', {}).get('external_datasets', {}).items():
        if 'td' in dataset_name.lower() and os.path.exists(data_path):
            td_data_paths[dataset_name] = data_path
    
    if td_data_paths:
        logging.info("Fitting bias correction using external TD cohorts...")
        
        # Combine TD data for bias correction
        all_ages = []
        all_predictions = []
        
        for cohort_name, data_path in td_data_paths.items():
            if os.path.exists(data_path):
                # Test on TD cohort to get predictions
                td_results = predictor.test_on_external_data(
                    data_path, cohort_name, apply_bias_correction=False
                )
                if td_results:
                    all_ages.extend(td_results['true_ages'])
                    all_predictions.extend(td_results['raw_predictions'])
        
        if all_ages and all_predictions:
            td_bias_params = predictor.fit_bias_correction({
                'ages': np.array(all_ages),
                'predictions': np.array(all_predictions)
            }, dataset_type="external_td")
            results['td_bias_correction'] = td_bias_params
    
    # Step 2: Test on external datasets
    external_datasets = config.get('data_paths', {}).get('external_datasets', {})
    test_results = {}
    
    for dataset_name, data_path in external_datasets.items():
        if os.path.exists(data_path):
            logging.info(f"Testing on {dataset_name}...")
            test_result = predictor.test_on_external_data(
                data_path, dataset_name, apply_bias_correction=True
            )
            if test_result:
                test_results[dataset_name] = test_result
                
                # Save individual dataset results
                dataset_output_dir = os.path.join(output_dir, 'individual_datasets')
                os.makedirs(dataset_output_dir, exist_ok=True)
                
                import json
                dataset_file = os.path.join(dataset_output_dir, f'{dataset_name}_results.json')
                with open(dataset_file, 'w') as f:
                    json.dump(test_result, f, indent=2, default=str)
                
                logging.info(f"Saved {dataset_name} results to: {dataset_file}")
    
    results['external_testing'] = test_results
    
    # Step 3: Comprehensive brain age analysis
    if test_results:
        logging.info("Performing comprehensive brain age analysis...")
        comprehensive_analysis = comprehensive_brain_age_analysis(test_results)
        results['comprehensive_analysis'] = comprehensive_analysis
        
        # Print summary of brain age gap analysis
        logging.info("=== BRAIN AGE GAP ANALYSIS SUMMARY ===")
        for dataset_name, metrics in comprehensive_analysis['individual_metrics'].items():
            bag_stats = metrics['bag_stats']
            logging.info(f"{dataset_name}: Mean BAG = {bag_stats['mean']:.3f} ± {bag_stats['std']:.3f} years (n={bag_stats['n']})")
        
        if 'group_comparisons' in comprehensive_analysis:
            logging.info("\n=== GROUP COMPARISONS ===")
            for comparison, stats in comprehensive_analysis['group_comparisons'].items():
                logging.info(f"{comparison}: t={stats['t_statistic']:.3f}, p={stats['p_value']:.4f}, Cohen's d={stats['cohens_d']:.3f}")
        
        # Save comprehensive analysis
        analysis_file = os.path.join(output_dir, 'comprehensive_brain_age_analysis.json')
        with open(analysis_file, 'w') as f:
            json.dump(comprehensive_analysis, f, indent=2, default=str)
        
        logging.info(f"Comprehensive analysis saved to: {analysis_file}")
    
    return results


def main():
    """Main function for brain age prediction analysis."""
    parser = argparse.ArgumentParser(
        description="Brain age prediction analysis with bias correction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete brain age prediction analysis
  python brain_age_prediction.py \\
    --config config.yaml \\
    --output_dir results/brain_age_prediction

  # Train models only
  python brain_age_prediction.py \\
    --hcp_dev_dir /path/to/hcp_dev_data \\
    --output_dir results/training
        """
    )
    
    parser.add_argument("--config", type=str,
                       help="Path to configuration file")
    parser.add_argument("--hcp_dev_dir", type=str,
                       help="Directory containing HCP-Dev training data")
    parser.add_argument("--output_dir", type=str, default="results/brain_age_prediction",
                       help="Output directory for results")
    parser.add_argument("--num_folds", type=int, default=5,
                       help="Number of cross-validation folds")
    parser.add_argument("--use_existing_models", action="store_true",
                       help="Use existing trained models instead of training new ones")
    parser.add_argument("--model_dir", type=str,
                       help="Directory containing existing trained models (required if --use_existing_models)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.config:
        # Load configuration and run complete analysis
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        if args.use_existing_models:
            if not args.model_dir:
                print("Error: --model_dir is required when using --use_existing_models")
                sys.exit(1)
            results = run_brain_age_prediction_analysis_with_existing_models(config, args.model_dir, args.output_dir)
        else:
            results = run_brain_age_prediction_analysis(config, args.output_dir)
        
        # Save results with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(args.output_dir, f'brain_age_prediction_results_{timestamp}.json')
        
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Brain age prediction analysis completed. Results saved to: {results_file}")
    
    elif args.hcp_dev_dir:
        # Train models only
        model_config = {
            'num_epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'dropout_rate': 0.6
        }
        
        predictor = BrainAgePredictor(model_config)
        results = predictor.train_models(args.hcp_dev_dir, args.num_folds)
        
        # Save results
        import json
        with open(os.path.join(args.output_dir, 'training_results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Model training completed. Results saved to: {args.output_dir}")
    
    else:
        print("Error: Either --config or --hcp_dev_dir must be provided")
        sys.exit(1)


if __name__ == "__main__":
    main()
