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

from data_utils import load_finetune_dataset, reshape_data
from model_utils import ConvNet, train_regressor_w_embedding, evaluate_model_performance
from plotting_utils import plot_age_prediction, save_figure, setup_fonts


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
        
    def train_models(self, data_dir: str, num_folds: int = 5) -> Dict:
        """
        Train brain age prediction models using nested cross-validation with weights and biases hyperparameter optimization.
        
        This implements:
        1. 5-fold outer CV for model evaluation
        2. Inner CV for hyperparameter optimization using Weights & Biases
        3. Hyperparameter search over learning rate, dropout, weight decay, etc.
        
        Args:
            data_dir (str): Directory containing .bin files for training
            num_folds (int): Number of cross-validation folds
            
        Returns:
            Dict: Training results and model paths
        """
        logging.info("Training brain age prediction models using nested CV with W&B hyperparameter optimization...")
        
        # Initialize Weights & Biases for hyperparameter optimization
        try:
            import wandb
            wandb.init(project="brain-age-prediction", config=self.model_config)
            use_wandb = True
        except ImportError:
            logging.warning("Weights & Biases not available, using default hyperparameters")
            use_wandb = False
        
        results = {
            'fold_results': [],
            'model_paths': [],
            'training_metrics': {},
            'hyperparameter_optimization': {}
        }
        
        for fold in range(num_folds):
            logging.info(f"Training fold {fold + 1}/{num_folds} with nested CV...")
            
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
                
                # Hyperparameter optimization using inner CV
                best_params = self._optimize_hyperparameters(X_train, Y_train, use_wandb, fold)
                
                # Initialize model with best hyperparameters
                model = ConvNet(
                    input_channels=X_train.shape[1],
                    dropout_rate=best_params.get('dropout_rate', 0.6)
                ).to(self.device)
                
                # Train model with optimized hyperparameters
                train_metrics = train_regressor_w_embedding(
                    model=model,
                    X_train=X_train,
                    Y_train=Y_train,
                    X_val=X_test,
                    Y_val=Y_test,
                    epochs=best_params.get('num_epochs', 100),
                    batch_size=best_params.get('batch_size', 32),
                    learning_rate=best_params.get('learning_rate', 0.001),
                    device=self.device
                )
                
                # Save model
                model_path = os.path.join(data_dir, f"best_model_fold_{fold}.pth")
                torch.save(model.state_dict(), model_path)
                
                # Store results
                results['fold_results'].append({
                    'fold': fold,
                    'train_metrics': train_metrics,
                    'model_path': model_path,
                    'best_hyperparameters': best_params
                })
                results['model_paths'].append(model_path)
                results['hyperparameter_optimization'][f'fold_{fold}'] = best_params
                
                logging.info(f"Fold {fold} completed - Final MAE: {train_metrics.get('final_mae', 'N/A')}")
                logging.info(f"Best hyperparameters for fold {fold}: {best_params}")
                
            except Exception as e:
                logging.error(f"Error training fold {fold}: {e}")
                continue
        
        # Compute ensemble metrics
        if results['fold_results']:
            results['training_metrics'] = self._compute_ensemble_metrics(results['fold_results'])
        
        if use_wandb:
            wandb.finish()
        
        logging.info("Model training with nested CV completed")
        return results
    
    def _optimize_hyperparameters(self, X_train: np.ndarray, Y_train: np.ndarray, use_wandb: bool, fold: int) -> Dict:
        """
        Optimize hyperparameters using inner cross-validation.
        
        Hyperparameters to optimize:
        - learning_rate: [1e-4, 1e-3, 1e-2]
        - dropout_rate: [0.3, 0.5, 0.7]
        - weight_decay: [1e-5, 1e-4, 1e-3]
        - batch_size: [32, 64, 128]
        - num_epochs: [50, 100, 150]
        """
        logging.info(f"Starting hyperparameter optimization for fold {fold}...")
        
        # Define hyperparameter search space
        param_grid = {
            'learning_rate': [1e-4, 1e-3, 1e-2],
            'dropout_rate': [0.3, 0.5, 0.7],
            'weight_decay': [1e-5, 1e-4, 1e-3],
            'batch_size': [32, 64, 128],
            'num_epochs': [50, 100, 150]
        }
        
        best_score = float('inf')
        best_params = {}
        
        # Simple grid search (can be replaced with more sophisticated methods like Optuna)
        for lr in param_grid['learning_rate']:
            for dropout in param_grid['dropout_rate']:
                for wd in param_grid['weight_decay']:
                    for batch_size in param_grid['batch_size']:
                        for epochs in param_grid['num_epochs']:
                            params = {
                                'learning_rate': lr,
                                'dropout_rate': dropout,
                                'weight_decay': wd,
                                'batch_size': batch_size,
                                'num_epochs': epochs
                            }
                            
                            # Inner CV for this hyperparameter combination
                            score = self._evaluate_hyperparameters(X_train, Y_train, params)
                            
                            if use_wandb:
                                import wandb
                                wandb.log({
                                    'fold': fold,
                                    'learning_rate': lr,
                                    'dropout_rate': dropout,
                                    'weight_decay': wd,
                                    'batch_size': batch_size,
                                    'num_epochs': epochs,
                                    'cv_score': score
                                })
                            
                            if score < best_score:
                                best_score = score
                                best_params = params.copy()
        
        logging.info(f"Best hyperparameters for fold {fold}: {best_params} (CV score: {best_score:.4f})")
        return best_params
    
    def _evaluate_hyperparameters(self, X_train: np.ndarray, Y_train: np.ndarray, params: dict) -> float:
        """Evaluate hyperparameters using inner cross-validation."""
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=3, shuffle=True, random_state=42)  # Inner CV
        scores = []
        
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            Y_tr, Y_val = Y_train[train_idx], Y_train[val_idx]
            
            # Create model with current hyperparameters
            model = ConvNet(
                input_channels=X_tr.shape[1], 
                dropout_rate=params['dropout_rate']
            ).to(self.device)
            
            # Train briefly for hyperparameter evaluation
            train_metrics = train_regressor_w_embedding(
                model=model,
                X_train=X_tr,
                Y_train=Y_tr,
                X_val=X_val,
                Y_val=Y_val,
                epochs=min(10, params['num_epochs']),  # Quick evaluation
                batch_size=params['batch_size'],
                learning_rate=params['learning_rate'],
                device=self.device
            )
            
            # Use MAE as the evaluation metric
            mae = train_metrics.get('final_mae', float('inf'))
            scores.append(mae)
        
        return np.mean(scores)
    
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
    
    def fit_bias_correction(self, td_data: Dict[str, np.ndarray]) -> Dict:
        """
        Fit bias correction parameters using TD cohort data.
        
        Args:
            td_data (Dict): Dictionary with 'ages' and 'predictions' arrays
            
        Returns:
            Dict: Bias correction parameters
        """
        logging.info("Fitting bias correction parameters...")
        
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
            'r2': float(reg.score(ages.reshape(-1, 1), bag.reshape(-1, 1)))
        }
        
        self.bias_params = bias_params
        logging.info(f"Bias correction fitted - R²: {bias_params['r2']:.3f}")
        
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
            # Load external data
            X_test, _, Y_test, _ = load_finetune_dataset(data_path)
            X_test = reshape_data(X_test)
            
            # Make predictions using ensemble of models
            all_predictions = []
            
            for model_path in self.models.get('model_paths', []):
                if not os.path.exists(model_path):
                    continue
                    
                # Load model
                model = ConvNet(input_channels=X_test.shape[1]).to(self.device)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.eval()
                
                # Make predictions
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_test).to(self.device)
                    predictions = model(X_tensor).cpu().numpy().flatten()
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
    
    def create_visualizations(self, test_results: Dict, output_dir: str) -> None:
        """
        Create brain age prediction visualizations.
        
        Args:
            test_results (Dict): Test results from external datasets
            output_dir (str): Output directory for figures
        """
        setup_fonts()
        os.makedirs(output_dir, exist_ok=True)
        
        for dataset_name, results in test_results.items():
            if not results or 'raw_predictions' not in results:
                continue
                
            # Create scatter plot
            fig = plot_age_prediction(
                actual_ages=results['true_ages'],
                predicted_ages=results['raw_predictions'],
                title=f"{dataset_name} Brain Age Prediction",
                save_path=os.path.join(output_dir, f"{dataset_name}_brain_age_prediction.png")
            )
            
            logging.info(f"Visualization created for {dataset_name}")


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
    
    # Step 2: Fit bias correction using TD cohorts
    td_data_paths = config.get('data_paths', {}).get('td_cohorts', {})
    if td_data_paths:
        logging.info("Fitting bias correction using TD cohorts...")
        
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
            bias_params = predictor.fit_bias_correction({
                'ages': np.array(all_ages),
                'predictions': np.array(all_predictions)
            })
            results['bias_correction'] = bias_params
    
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
    
    # Step 4: Create visualizations
    if test_results:
        predictor.create_visualizations(test_results, output_dir)
    
    logging.info("Brain age prediction analysis completed")
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
        
        results = run_brain_age_prediction_analysis(config, args.output_dir)
        
        # Save results
        import json
        with open(os.path.join(args.output_dir, 'brain_age_prediction_results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Brain age prediction analysis completed. Results saved to: {args.output_dir}")
    
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
