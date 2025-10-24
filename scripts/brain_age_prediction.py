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
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, TensorDataset
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

from data_utils import load_finetune_dataset, load_finetune_dataset_w_ids, reshape_data, standardize_timeseries_data
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
            # Look for PyTorch Lightning checkpoints first
            checkpoint_dir = os.path.join(model_dir, f"fold_{fold}_checkpoints")
            scaler_path = os.path.join(model_dir, f"age_scaler_fold_{fold}.pkl")
            model_path = None
            
            # Try PyTorch Lightning checkpoints first
            if os.path.exists(checkpoint_dir):
                import glob
                checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "best_model*.ckpt"))
                if checkpoint_files:
                    model_path = checkpoint_files[0]  # Use first match
                    logging.info(f"Found PyTorch Lightning checkpoint for fold {fold}: {model_path}")
            
            # If no Lightning checkpoint, try legacy PyTorch models
            if model_path is None:
                # Look for legacy .pt files in the main directory
                import glob
                legacy_files = glob.glob(os.path.join(model_dir, f"*fold_{fold}*.pt"))
                if not legacy_files:
                    # Try alternative naming patterns
                    legacy_files = glob.glob(os.path.join(model_dir, f"*fold_{fold}*"))
                    legacy_files = [f for f in legacy_files if f.endswith('.pt')]
                
                # If still no files, try using the single legacy model for all folds
                if not legacy_files:
                    # Use the legacy model path from config for all folds
                    import yaml
                    try:
                        with open('config.yaml', 'r') as f:
                            config = yaml.safe_load(f)
                        legacy_model_path = config.get('existing_models', {}).get('legacy_model_path')
                        if legacy_model_path and os.path.exists(legacy_model_path):
                            model_path = legacy_model_path
                            logging.info(f"Using single legacy model for all folds: {model_path}")
                        else:
                            logging.warning(f"No model found for fold {fold} in {model_dir}")
                            continue
                    except Exception as e:
                        logging.warning(f"Could not load config: {e}")
                        continue
                else:
                    model_path = legacy_files[0]  # Use first match
                    logging.info(f"Found legacy PyTorch model for fold {fold}: {model_path}")
            
            # Check for age scaler (may not exist for legacy models)
            if os.path.exists(scaler_path):
                results['fold_results'].append({
                    'fold': fold,
                    'model_path': model_path,
                    'scaler_path': scaler_path,
                    'model_type': 'lightning' if model_path.endswith('.ckpt') else 'legacy'
                })
                results['model_paths'].append(model_path)
                results['age_scalers'].append(scaler_path)
                logging.info(f"Loaded fold {fold}: {model_path}")
            else:
                # For legacy models, create age scaler from training data
                logging.warning(f"Age scaler not found for fold {fold}: {scaler_path}")
                
                # Try to create scaler from training data
                age_scaler = self._create_scaler_from_training_data(fold, model_dir)
                
                if age_scaler is not None:
                    # Save the created scaler
                    age_scaler.save(scaler_path)
                    logging.info(f"Created and saved age scaler for fold {fold}: {scaler_path}")
                    
                    results['fold_results'].append({
                        'fold': fold,
                        'model_path': model_path,
                        'scaler_path': scaler_path,
                        'model_type': 'lightning' if model_path.endswith('.ckpt') else 'legacy'
                    })
                    results['model_paths'].append(model_path)
                    results['age_scalers'].append(scaler_path)
                    logging.info(f"Loaded fold {fold} (created scaler): {model_path}")
                else:
                    # Still add the model but without scaler
                    results['fold_results'].append({
                        'fold': fold,
                        'model_path': model_path,
                        'scaler_path': None,
                        'model_type': 'lightning' if model_path.endswith('.ckpt') else 'legacy'
                    })
                    results['model_paths'].append(model_path)
                    results['age_scalers'].append(None)
                    logging.info(f"Loaded fold {fold} (no scaler): {model_path}")
        
        logging.info(f"Loaded {len(results['model_paths'])} existing models")
        return results
    
    def retrain_models(self, config: Dict, model_dir: str, num_folds: int = 5) -> Dict:
        """
        Retrain models with consistent architecture.
        
        Args:
            config (Dict): Configuration dictionary
            model_dir (str): Directory to save retrained models
            num_folds (int): Number of cross-validation folds
            
        Returns:
            Dict: Training results
        """
        logging.info("Retraining models with consistent architecture...")
        
        # Get HCP-Dev data directory
        hcp_dev_data_dir = config.get('data_paths', {}).get('hcp_dev', {}).get('training_data_dir')
        if not hcp_dev_data_dir:
            hcp_dev_data_dir = "/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/hcp_dev_age_five_fold"
        
        results = {
            'fold_results': [],
            'model_paths': [],
            'age_scalers': [],
            'training_metrics': {}
        }
        
        for fold in range(num_folds):
            logging.info(f"Training fold {fold}...")
            
            # Load fold data
            fold_data_path = os.path.join(hcp_dev_data_dir, f"fold_{fold}.bin")
            if not os.path.exists(fold_data_path):
                logging.warning(f"Fold data not found: {fold_data_path}")
                continue
            
            X_train, X_test, Y_train, Y_test = load_finetune_dataset(fold_data_path)
            X_train = reshape_data(X_train)
            X_test = reshape_data(X_test)
            
            # Create and fit age scaler
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
            
            # Use custom training function that matches your previous approach exactly
            model, train_loss_list, val_loss_list, outputs_store, targets_store, outputs_store_train, targets_store_train = self._train_model_custom(
                X_train, Y_train_scaled, X_test, Y_test_scaled, model_dir, fold
            )
            
            # Save age scaler
            scaler_path = os.path.join(model_dir, f"age_scaler_fold_{fold}.pkl")
            age_scaler.save(scaler_path)
            
            # Get the best model path (saved during training)
            best_model_path = os.path.join(model_dir, f"retrained_fold_{fold}_model.pt")
            
            # Use predictions from custom training (already in scaled form)
            predictions_scaled = outputs_store  # From custom training
            predictions = age_scaler.inverse_transform(predictions_scaled)
            
            # Evaluate model
            metrics = evaluate_model_performance(Y_test, predictions)
            
            results['fold_results'].append({
                'fold': fold,
                'model_path': best_model_path,
                'scaler_path': scaler_path,
                'model_type': 'lightning',
                'train_metrics': metrics
            })
            results['model_paths'].append(best_model_path)
            results['age_scalers'].append(scaler_path)
            
            logging.info(f"Fold {fold} completed - MAE: {metrics['mae']:.3f}, R²: {metrics['r2']:.3f}")
        
        logging.info(f"Retrained {len(results['model_paths'])} models")
        return results
    
    def _train_model_custom(self, X_train, Y_train, X_test, Y_test, model_dir, fold):
        """
        Custom training function that exactly matches the previous PyTorch approach.
        """
        from model_utils import ConvNet
        
        # Create data loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(Y_test))
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Initialize model exactly like your previous approach
        model = ConvNet(dropout_rate=0.6)
        
        # Use CUDA if available
        USE_CUDA = torch.cuda.is_available()
        if USE_CUDA:
            model = model.cuda()
        
        # Setup training exactly like your previous approach
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0006, weight_decay=0.0001)
        
        total_step = len(train_loader)
        train_loss_list = []
        val_loss_list = []
        val_loss_temp = 100000000.0
        num_epochs = 100
        
        # Model save path
        fname_model = os.path.join(model_dir, f"retrained_fold_{fold}_model.pt")
        
        # Training loop exactly like your previous approach
        for epoch in range(num_epochs):
            # Put the model into the training mode
            model.train()
            for i, (data_ts, labels) in enumerate(train_loader):
                if USE_CUDA:
                    data_ts = data_ts.cuda()
                    labels = labels.cuda()
                
                # Run the forward pass
                outputs = model(data_ts)
                loss = torch.sqrt(criterion(outputs, labels))  # RMSE loss like your approach
                
                # Track the Training Loss
                train_loss_list.append(loss.item())
                train_loss = loss.item()
                
                # Backprop and perform Adam optimisation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Validation Loss and Accuracy (every batch like your approach)
                model.eval()
                with torch.no_grad():
                    total_valid_loss = 0.0
                    cnt = 0
                    for images, labels in val_loader:
                        if USE_CUDA:
                            images = images.cuda()
                            labels = labels.cuda()
                        outputs = model(images)
                        loss = criterion(outputs, labels).item()
                        total_valid_loss += loss
                        cnt += 1
                    total_valid_loss = total_valid_loss / cnt
                    val_loss_list.append(total_valid_loss)
                    
                    # Save model on best validation loss (like your approach)
                    if val_loss_temp > total_valid_loss:
                        val_loss_temp = total_valid_loss
                        logging.info('**Saving Model on Drive**')
                        torch.save(model.state_dict(), fname_model)
                        logging.info('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                                   .format(epoch + 1, num_epochs, i + 1, total_step, train_loss, total_valid_loss))
                
                if (i + 1) % 10 == 0:
                    logging.info('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                               .format(epoch + 1, num_epochs, i + 1, total_step, train_loss, total_valid_loss))
        
        # Load the best model for evaluation
        if USE_CUDA:
            model.load_state_dict(torch.load(fname_model))
            model.cuda()
        else:
            model.load_state_dict(torch.load(fname_model, map_location=torch.device('cpu')))
        
        model.eval()
        
        # Apply on the Test Data (like your approach)
        targets_store = []
        outputs_store = []
        with torch.no_grad():
            for images, labels in val_loader:
                if USE_CUDA:
                    images = images.cuda()
                    labels = labels.cuda()
                outputs = model(images)
                outputs_store.append(outputs.cpu().detach().numpy())
                targets_store.append(labels.cpu().detach().numpy())
        
        outputs_store = np.concatenate(outputs_store)
        targets_store = np.concatenate(targets_store)
        
        # Apply on the Train Data (like your approach)
        targets_store_train = []
        outputs_store_train = []
        with torch.no_grad():
            for images, labels in train_loader:
                if USE_CUDA:
                    images = images.cuda()
                    labels = labels.cuda()
                outputs = model(images)
                outputs_store_train.append(outputs.cpu().detach().numpy())
                targets_store_train.append(labels.cpu().detach().numpy())
        
        targets_store_train = np.concatenate(targets_store_train)
        outputs_store_train = np.concatenate(outputs_store_train)
        
        return model, train_loss_list, val_loss_list, outputs_store, targets_store, outputs_store_train, targets_store_train
    
    def _create_scaler_from_training_data(self, fold: int, model_dir: str) -> Optional[AgeScaler]:
        """
        Create age scaler from training data for a specific fold.
        
        Args:
            fold (int): Fold number
            model_dir (str): Model directory path
            
        Returns:
            Optional[AgeScaler]: Created age scaler or None if failed
        """
        try:
            # Load config to get HCP-Dev data directory
            import yaml
            config_path = os.path.join(os.path.dirname(model_dir), '..', 'config.yaml')
            if not os.path.exists(config_path):
                config_path = 'config.yaml'  # Fallback to current directory
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Get HCP-Dev training data directory from config
            hcp_dev_data_dir = config.get('data_paths', {}).get('hcp_dev', {}).get('training_data_dir')
            if not hcp_dev_data_dir:
                # Fallback to hardcoded path
                hcp_dev_data_dir = "/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/hcp_dev_age_five_fold"
            
            fold_data_path = os.path.join(hcp_dev_data_dir, f"fold_{fold}.bin")
            
            if not os.path.exists(fold_data_path):
                logging.warning(f"Training data not found for fold {fold}: {fold_data_path}")
                return None
            
            # Load training data
            X_train, X_test, Y_train, Y_test = load_finetune_dataset(fold_data_path)
            
            # Create and fit age scaler on training ages
            age_scaler = AgeScaler()
            age_scaler.fit(Y_train)
            
            logging.info(f"Created age scaler for fold {fold} using {len(Y_train)} training samples from {fold_data_path}")
            return age_scaler
            
        except Exception as e:
            logging.error(f"Failed to create age scaler for fold {fold}: {e}")
            return None
    
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
        Apply bias correction to predictions using the exact approach from previous implementation.
        
        Args:
            ages (np.ndarray): True ages
            predictions (np.ndarray): Raw predictions
            
        Returns:
            np.ndarray: Bias-corrected predictions
        """
        if not self.bias_params:
            logging.warning("No bias correction parameters found. Returning raw predictions.")
            return predictions
        
        # Use the exact approach: Offset = coef * ages + intercept
        # Then: predicted_ages = ensemble_predictions - Offset
        offset = self.bias_params['beta'] * ages + self.bias_params['alpha']
        corrected_predictions = predictions - offset
        
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
            # Try to load external data with IDs first
            try:
                X_train, X_test, id_train, Y_train, Y_test, id_test = load_finetune_dataset_w_ids(data_path)
                # Combine train and test data for external testing
                X_test = np.concatenate([X_train, X_test], axis=0)
                Y_test = np.concatenate([Y_train, Y_test], axis=0)
                id_test = np.concatenate([id_train, id_test], axis=0)
            except KeyError:
                # Fallback to loading without IDs if 'id_train' key is missing
                logging.warning(f"ID keys not found in {data_path}, loading without IDs")
                X_train, X_test, Y_train, Y_test = load_finetune_dataset(data_path)
                # Combine train and test data for external testing
                X_test = np.concatenate([X_train, X_test], axis=0)
                Y_test = np.concatenate([Y_train, Y_test], axis=0)
                # Create dummy IDs
                id_test = np.arange(len(Y_test))
            
            X_test = reshape_data(X_test)
            
            # Apply standardization for Stanford ASD data
            if 'stanford' in dataset_name.lower():
                logging.info(f"Applying standardization for {dataset_name}")
                X_test = standardize_timeseries_data(X_test)
                logging.info(f"Standardization applied to {dataset_name}")
            
            # Make predictions using ensemble of models
            all_predictions = []
            
            for i, model_path in enumerate(self.models.get('model_paths', [])):
                if not os.path.exists(model_path):
                    continue
                
                # Get model type and scaler path
                fold_result = self.models.get('fold_results', [])[i]
                model_type = fold_result.get('model_type', 'legacy')
                scaler_path = self.models.get('age_scalers', [])[i]
                
                # Load age scaler if available
                age_scaler = None
                if scaler_path and os.path.exists(scaler_path):
                    age_scaler = AgeScaler.load(scaler_path)
                elif model_type == 'legacy':
                    logging.info(f"Using legacy model without age scaler for fold {i}")
                else:
                    logging.warning(f"Age scaler not found: {scaler_path}")
                    continue
                
                # Load model based on type
                if model_type == 'lightning' or model_path.endswith('.ckpt'):
                    # Load PyTorch Lightning model from checkpoint
                    model = load_lightning_model_from_checkpoint(
                        checkpoint_path=model_path,
                        input_channels=X_test.shape[1],
                        dropout_rate=self.model_config.get('dropout_rate', 0.6),
                        learning_rate=self.model_config.get('learning_rate', 0.001)
                    )
                else:
                    # Load legacy PyTorch model
                    from model_utils import ConvNet
                    model = ConvNet(dropout_rate=0.6)
                    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
                
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
                    
                    # Handle age scaling
                    if age_scaler is not None:
                        # Inverse transform predictions back to original age scale
                        predictions = age_scaler.inverse_transform(predictions_scaled)
                    else:
                        # For legacy models without scalers, assume predictions are already in original scale
                        predictions = predictions_scaled
                        logging.info(f"Using raw predictions for legacy model (no age scaling)")
                    
                    all_predictions.append(predictions)
            
            if not all_predictions:
                raise ValueError("No valid models found for prediction")
            
            # Ensemble predictions
            ensemble_predictions = np.mean(all_predictions, axis=0)
            
            # Compute metrics BEFORE bias correction
            metrics_before = evaluate_model_performance(Y_test, ensemble_predictions)
            
            # Apply bias correction if requested
            corrected_predictions = ensemble_predictions
            if apply_bias_correction and self.bias_params:
                corrected_predictions = self.apply_bias_correction(Y_test, ensemble_predictions)
                # Compute metrics AFTER bias correction
                metrics_after = evaluate_model_performance(Y_test, corrected_predictions)
            else:
                metrics_after = metrics_before
            
            # Create results
            results = {
                'dataset_name': dataset_name,
                'n_subjects': len(Y_test),
                'raw_predictions': ensemble_predictions,
                'corrected_predictions': corrected_predictions,
                'true_ages': Y_test,
                'metrics_before_correction': metrics_before,
                'metrics_after_correction': metrics_after,
                'bias_correction_applied': apply_bias_correction
            }
            
            logging.info(f"{dataset_name} testing completed:")
            logging.info(f"  BEFORE correction - MAE: {metrics_before['mae']:.3f}, R²: {metrics_before['r2']:.3f}")
            if apply_bias_correction and self.bias_params:
                logging.info(f"  AFTER correction  - MAE: {metrics_after['mae']:.3f}, R²: {metrics_after['r2']:.3f}")
            
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
                if td_results and 'true_ages' in td_results and 'raw_predictions' in td_results:
                    all_ages.extend(td_results['true_ages'])
                    all_predictions.extend(td_results['raw_predictions'])
                    logging.info(f"Successfully processed {cohort_name}: {len(td_results['true_ages'])} subjects")
                else:
                    logging.warning(f"No valid results from {cohort_name}")
        
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


def run_brain_age_prediction_analysis_with_retraining(config: Dict, model_dir: str, output_dir: str) -> Dict:
    """
    Run brain age prediction analysis with model retraining.
    
    Args:
        config (Dict): Configuration dictionary
        model_dir (str): Directory to save retrained models
        output_dir (str): Output directory for results
        
    Returns:
        Dict: Analysis results
    """
    logging.info("Running brain age prediction analysis with model retraining...")
    
    # Initialize predictor
    model_config = config.get('model_config', {})
    predictor = BrainAgePredictor(model_config)
    
    # Retrain models
    num_folds = config.get('num_folds', 5)
    training_results = predictor.retrain_models(config, model_dir, num_folds)
    
    if not training_results['model_paths']:
        raise ValueError("No models were successfully retrained")
    
    # Store models in predictor for testing
    predictor.models = training_results
    
    # Create results structure
    results = {
        'analysis_type': 'retrained_models',
        'model_directory': model_dir,
        'training_info': training_results,
        'external_testing': {},
        'comprehensive_analysis': {}
    }
    
    # Continue with the same testing pipeline as existing models
    return _run_external_testing_pipeline(predictor, config, results, output_dir)


def _run_external_testing_pipeline(predictor, config: Dict, results: Dict, output_dir: str) -> Dict:
    """
    Run the external testing pipeline (common for both existing and retrained models).
    
    Args:
        predictor: BrainAgePredictor instance
        config (Dict): Configuration dictionary
        results (Dict): Results dictionary to update
        output_dir (str): Output directory for results
        
    Returns:
        Dict: Updated results
    """
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
                try:
                    # Test on TD cohort to get predictions
                    td_results = predictor.test_on_external_data(
                        data_path, cohort_name, apply_bias_correction=False
                    )
                    if td_results and 'true_ages' in td_results and 'raw_predictions' in td_results:
                        all_ages.extend(td_results['true_ages'])
                        all_predictions.extend(td_results['raw_predictions'])
                        logging.info(f"Successfully processed {cohort_name}: {len(td_results['true_ages'])} subjects")
                    else:
                        logging.warning(f"No valid results from {cohort_name}")
                except Exception as e:
                    logging.error(f"Error processing {cohort_name}: {str(e)}")
                    continue
        
        if all_ages and all_predictions:
            logging.info(f"Fitting bias correction with {len(all_ages)} subjects from TD cohorts")
            td_bias_params = predictor.fit_bias_correction({
                'ages': np.array(all_ages),
                'predictions': np.array(all_predictions)
            }, dataset_type="external_td")
            results['td_bias_correction'] = td_bias_params
        else:
            logging.warning("No TD data available for bias correction fitting")
    
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
                
    results['external_testing'] = test_results
    
    # Step 3: Comprehensive analysis
    if test_results:
        comprehensive_results = comprehensive_brain_age_analysis(test_results)
        results['comprehensive_analysis'] = comprehensive_results
    
    # Step 4: Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"brain_age_prediction_results_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logging.info(f"Brain age prediction analysis completed. Results saved to: {results_file}")
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
    
    # Use common testing pipeline
    return _run_external_testing_pipeline(predictor, config, results, output_dir)
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
                if td_results and 'true_ages' in td_results and 'raw_predictions' in td_results:
                    all_ages.extend(td_results['true_ages'])
                    all_predictions.extend(td_results['raw_predictions'])
                    logging.info(f"Successfully processed {cohort_name}: {len(td_results['true_ages'])} subjects")
                else:
                    logging.warning(f"No valid results from {cohort_name}")
        
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
    parser.add_argument("--retrain_models", action="store_true",
                       help="Retrain models with consistent architecture (32 channels)")
    parser.add_argument("--model_dir", type=str,
                       help="Directory containing existing trained models (required if --use_existing_models or --retrain_models)")
    
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
        
        if args.retrain_models:
            if not args.model_dir:
                print("Error: --model_dir is required when using --retrain_models")
                sys.exit(1)
            results = run_brain_age_prediction_analysis_with_retraining(config, args.model_dir, args.output_dir)
        elif args.use_existing_models:
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
