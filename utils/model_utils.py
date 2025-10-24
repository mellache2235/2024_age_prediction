"""
Model utilities for age prediction analysis.

This module provides neural network models, training functions, and model evaluation utilities.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import math
from typing import Dict, List, Tuple, Optional, Any
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats
import os
import pickle


class AgeScaler:
    """Age scaling utility for brain age prediction models."""
    
    def __init__(self):
        """Initialize age scaler."""
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, ages: np.ndarray) -> None:
        """
        Fit scaler on training ages.
        
        Args:
            ages (np.ndarray): Training ages to fit scaler on
        """
        self.scaler.fit(ages.reshape(-1, 1))
        self.is_fitted = True
        
    def transform(self, ages: np.ndarray) -> np.ndarray:
        """
        Transform ages using fitted scaler.
        
        Args:
            ages (np.ndarray): Ages to transform
            
        Returns:
            np.ndarray: Scaled ages
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
        return self.scaler.transform(ages.reshape(-1, 1)).flatten()
        
    def inverse_transform(self, scaled_ages: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled ages back to original scale.
        
        Args:
            scaled_ages (np.ndarray): Scaled ages to inverse transform
            
        Returns:
            np.ndarray: Unscaled ages
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse_transform")
        return self.scaler.inverse_transform(scaled_ages.reshape(-1, 1)).flatten()
        
    def save(self, filepath: str) -> None:
        """
        Save scaler to file.
        
        Args:
            filepath (str): Path to save scaler
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
            
    @classmethod
    def load(cls, filepath: str) -> 'AgeScaler':
        """
        Load scaler from file.
        
        Args:
            filepath (str): Path to load scaler from
            
        Returns:
            AgeScaler: Loaded scaler instance
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class RMSELoss(nn.Module):
    """Root Mean Square Error loss function."""
    
    def __init__(self, eps: float = 1e-6):
        """
        Initialize RMSE loss.
        
        Args:
            eps (float): Small epsilon value for numerical stability
        """
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute RMSE loss.
        
        Args:
            yhat (torch.Tensor): Predicted values
            y (torch.Tensor): True values
            
        Returns:
            torch.Tensor: RMSE loss
        """
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


class ConvNetLightning(pl.LightningModule):
    """PyTorch Lightning version of ConvNet for brain age prediction."""
    
    def __init__(self, input_channels=246, dropout_rate=0.6, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        
        # Use the exact same architecture as ConvNet
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, stride=1, bias=False),
            nn.BatchNorm1d(32),
            nn.PReLU(32),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=7, stride=1, bias=False),
            nn.BatchNorm1d(32),
            nn.PReLU(32),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.drop_out = nn.Dropout(p=dropout_rate)
        self.regressor = nn.Linear(32, 1)
        
        self.loss_fn = RMSELoss()
        self.learning_rate = learning_rate
        
    def forward(self, x):
        # x shape: (batch_size, channels, sequence_length)
        out = self.layer1(x)
        out = self.drop_out(out)
        out = self.layer2(out)
        out = self.drop_out(out)
        out = out.mean(axis=2)  # Global average pooling
        out = self.regressor(out)
        return out.squeeze()
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class Conv1dSame(nn.Module):
    """1D Convolution with same padding."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, dilation: int = 1):
        """
        Initialize Conv1dSame layer.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Size of the convolution kernel
            stride (int): Stride of the convolution
            dilation (int): Dilation of the convolution
        """
        super().__init__()
        self.cut_last_element = (kernel_size % 2 == 0 and stride == 1 and dilation % 2 == 1)
        self.padding = math.ceil((1 - stride + dilation * (kernel_size-1))/2)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             padding=self.padding, stride=stride, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        if self.cut_last_element:
            return self.conv(x)[:, :, :-1]
        else:
            return self.conv(x)


class ConvNet(nn.Module):
    """Convolutional Neural Network for age prediction."""
    
    def __init__(self, dropout_rate: float = 0.6):
        """
        Initialize ConvNet model.
        
        Args:
            dropout_rate (float): Dropout rate for regularization
        """
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(246, 32, kernel_size=5, stride=1, bias=False),
            nn.BatchNorm1d(32),
            nn.PReLU(32),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=7, stride=1, bias=False),
            nn.BatchNorm1d(32),
            nn.PReLU(32),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.drop_out = nn.Dropout(p=dropout_rate)
        self.regressor = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Predicted age values
        """
        out = self.layer1(x)
        out = self.drop_out(out)
        out = self.layer2(out)
        out = self.drop_out(out)
        out = out.mean(axis=2)
        out = self.regressor(out)
        return out


class ConvNet_v2(nn.Module):
    """Improved version of ConvNet with different architecture."""
    
    def __init__(self):
        """Initialize ConvNet_v2 model."""
        super(ConvNet_v2, self).__init__()
        self.layer1 = nn.Sequential(
            Conv1dSame(246, 128, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3, stride=1, padding=int(3/2)))
        self.layer2 = nn.Sequential(
            Conv1dSame(128, 128, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=5, stride=1, padding=int(5/2)))

        self.drop_out = nn.Dropout(p=0.40581892575490663)
        self.regressor = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Predicted age values
        """
        out = self.layer1(x)
        out = self.drop_out(out)
        out = self.layer2(out)
        out = self.drop_out(out)
        out = out.mean(axis=2)
        out = self.regressor(out)
        return out


class ConvNet_resting_data_mask(nn.Module):
    """CNN for resting state data with masking capabilities."""
    
    def __init__(self, drop_out_rate: float = 0.5):
        """
        Initialize ConvNet_resting_data_mask model.
        
        Args:
            drop_out_rate (float): Dropout rate for regularization
        """
        super(ConvNet_resting_data_mask, self).__init__()
        self.drop_out_rate = drop_out_rate
        
        # CNN Block 1
        self.layer1 = Conv1dSame(246, 256, kernel_size=3, stride=1)
        self.layer2 = nn.ReLU()
        self.layer3 = Conv1dSame(256, 256, kernel_size=3, stride=1)
        self.layer4 = nn.ReLU()
        self.layer5 = nn.AvgPool1d(kernel_size=3, stride=1, padding=int(3 / 2))
        
        # CNN Block 2
        self.layer6 = Conv1dSame(256, 512, kernel_size=10, stride=1)
        self.layer7 = nn.ReLU()
        self.layer8 = Conv1dSame(512, 512, kernel_size=10, stride=1)
        self.layer9 = nn.ReLU()
        self.layer10 = nn.AvgPool1d(kernel_size=7, stride=1, padding=int(7 / 2))
        
        # CNN Block 3: Bring back to Channel Size
        self.layer11 = Conv1dSame(512, 246, kernel_size=12, stride=1)
        self.layer12 = nn.ReLU()
        self.layer13 = Conv1dSame(246, 246, kernel_size=12, stride=1)
        self.layer14 = nn.ReLU()
        self.layer15 = nn.AvgPool1d(kernel_size=7, stride=1, padding=int(7 / 2))
        
        # Dropout
        self.drop_out = nn.Dropout(p=self.drop_out_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        return out


def get_data_loaders_for_regression(data: Dict[str, np.ndarray], 
                                  hyper_parameters: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create data loaders for regression training.

    Args:
        data (Dict[str, np.ndarray]): Dictionary containing train/val/test data
        hyper_parameters (Dict[str, Any]): Hyperparameters including batch_size

    Returns:
        Tuple[DataLoader, DataLoader, Optional[DataLoader]]: Train, validation, and test loaders
    """
    batch_size = hyper_parameters['batch_size']

    # Train Data
    input_tensor = torch.from_numpy(data['train_features']).type(torch.FloatTensor)
    label_tensor = torch.from_numpy(data['train_labels']).type(torch.FloatTensor)
    dataset_train = TensorDataset(input_tensor, label_tensor)

    # Validation Data
    input_tensor_valid = torch.from_numpy(data['valid_features']).type(torch.FloatTensor)
    label_tensor_valid = torch.from_numpy(data['valid_labels']).type(torch.FloatTensor)
    dataset_valid = TensorDataset(input_tensor_valid, label_tensor_valid)

    # Test Data
    test_loader = None
    if data.get('test_features') is not None:
        input_tensor_test = torch.from_numpy(data['test_features']).type(torch.FloatTensor)
        label_tensor_test = torch.from_numpy(data['test_labels']).type(torch.FloatTensor)
        dataset_test = TensorDataset(input_tensor_test, label_tensor_test)
        test_loader = DataLoader(dataset=dataset_test, batch_size=data['test_features'].shape[0], shuffle=False)

    # Create loaders
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=dataset_valid, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader


def train_regressor_w_embedding(train_loader: DataLoader, valid_loader: DataLoader, 
                              test_loader: Optional[DataLoader], hyper_parameters: Dict[str, Any],
                              fname_model: str, use_cuda: bool = False) -> Tuple[Any, List[float], List[float], np.ndarray, np.ndarray]:
    """
    Train regression model with embedding.

    Args:
        train_loader (DataLoader): Training data loader
        valid_loader (DataLoader): Validation data loader
        test_loader (DataLoader, optional): Test data loader
        hyper_parameters (Dict[str, Any]): Hyperparameters
        fname_model (str): Path to save the model
        use_cuda (bool): Whether to use CUDA

    Returns:
        Tuple containing model, train_loss_list, val_loss_list, outputs_store, targets_store
    """
    model = ConvNet()
    
    if use_cuda:
        model.cuda()
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=hyper_parameters['learning_rate'], 
                                weight_decay=0.0001)
    total_step = len(train_loader)

    # Training
    train_loss_list = []
    val_loss_list = []
    val_loss_temp = 100000000.0
    num_epochs = hyper_parameters['num_epochs']
    
    for epoch in range(num_epochs):
        model.train()
        for i, (data_ts, labels) in enumerate(train_loader):
            if use_cuda:
                data_ts = data_ts.cuda()
                labels = labels.cuda()
            
            # Forward pass
            outputs = model(data_ts)
            loss = torch.sqrt(criterion(outputs, labels))
            train_loss_list.append(loss.item())
            train_loss = loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                total_valid_loss = 0.0
                cnt = 0
                for images, labels in valid_loader:
                    if use_cuda:
                        images = images.cuda()
                        labels = labels.cuda()
                    outputs = model(images)
                    loss = criterion(outputs, labels).item()
                    total_valid_loss += loss
                    cnt += 1
                total_valid_loss = total_valid_loss / cnt
                val_loss_list.append(total_valid_loss)

            if val_loss_temp > total_valid_loss:
                val_loss_temp = total_valid_loss
                print('**Saving Model on Drive**')
                torch.save(model.state_dict(), fname_model)
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f} '
                      .format(epoch + 1, num_epochs, i + 1, total_step, train_loss, total_valid_loss))
            
            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f} '
                      .format(epoch + 1, num_epochs, i + 1, total_step, train_loss, total_valid_loss))

    # Load best model and evaluate
    model = ConvNet()
    if use_cuda:
        model.load_state_dict(torch.load(fname_model))
        model.cuda()
    else:
        model.load_state_dict(torch.load(fname_model, map_location=torch.device('cpu')))
    
    model.eval()
    targets_store = []
    outputs_store = []
    
    with torch.no_grad():
        for images, labels in valid_loader:
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()
            outputs = model(images)
            outputs_store.append(outputs.cpu().detach().numpy())
            targets_store.append(labels.cpu().detach().numpy())
        
        outputs_store = np.concatenate(outputs_store)
        targets_store = np.concatenate(targets_store)

    return model, train_loss_list, val_loss_list, outputs_store, targets_store


def test_model(x_valid: np.ndarray, y_valid: np.ndarray, 
              hyper_parameters: Dict[str, Any], fname_model: str) -> Tuple[float, Tuple[float, float]]:
    """
    Test model performance on validation data.

    Args:
        x_valid (np.ndarray): Validation features
        y_valid (np.ndarray): Validation labels
        hyper_parameters (Dict[str, Any]): Hyperparameters
        fname_model (str): Path to the saved model

    Returns:
        Tuple[float, Tuple[float, float]]: Correlation coefficient and p-value
    """
    input_tensor_valid = torch.from_numpy(x_valid).type(torch.FloatTensor)
    label_tensor_valid = torch.from_numpy(y_valid).type(torch.FloatTensor)
    dataset_valid = TensorDataset(input_tensor_valid, label_tensor_valid)
    valid_loader = DataLoader(dataset=dataset_valid, batch_size=x_valid.shape[0], shuffle=False)
    
    model = ConvNet()
    model.load_state_dict(torch.load(fname_model, map_location=torch.device('cpu')))
    
    model.eval()
    with torch.no_grad():
        for images, labels in valid_loader:
            outputs = model(images)
            outputs = torch.squeeze(outputs)
            labels = torch.squeeze(labels)
            
            test_corr_coeff = np.corrcoef(labels.cpu(), outputs.cpu())[0, 1]
            pval = stats.pearsonr(labels.cpu(), outputs.cpu())
            
            print('Test Accuracy of the model: {} %'.format(abs(test_corr_coeff)))
            print('P-value:', stats.pearsonr(labels.cpu(), outputs.cpu()))
    
    return test_corr_coeff, pval


def evaluate_model_performance(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate model performance with multiple metrics including brain age gap.

    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values

    Returns:
        Dict[str, float]: Dictionary containing performance metrics
    """
    # Calculate basic metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = r2_score(y_true, y_pred)
    corr, p_value = stats.pearsonr(y_true, y_pred)
    
    # Calculate brain age gap (BAG) metrics
    bag = y_pred - y_true  # Brain Age Gap = Predicted - True
    mean_bag = np.mean(bag)
    std_bag = np.std(bag)
    median_bag = np.median(bag)
    
    # Calculate additional metrics
    mean_absolute_bag = np.mean(np.abs(bag))
    max_bag = np.max(bag)
    min_bag = np.min(bag)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'correlation': corr,
        'p_value': p_value,
        'mean_bag': mean_bag,
        'std_bag': std_bag,
        'median_bag': median_bag,
        'mean_absolute_bag': mean_absolute_bag,
        'max_bag': max_bag,
        'min_bag': min_bag
    }


def compare_brain_age_gaps(group1_bag: np.ndarray, group2_bag: np.ndarray, 
                          group1_name: str = "Group 1", group2_name: str = "Group 2") -> Dict[str, Any]:
    """
    Compare brain age gaps between two groups using statistical tests.
    
    Args:
        group1_bag (np.ndarray): Brain age gaps for group 1
        group2_bag (np.ndarray): Brain age gaps for group 2
        group1_name (str): Name of group 1
        group2_name (str): Name of group 2
        
    Returns:
        Dict[str, Any]: Statistical comparison results
    """
    # Basic statistics
    group1_stats = {
        'mean': np.mean(group1_bag),
        'std': np.std(group1_bag),
        'median': np.median(group1_bag),
        'n': len(group1_bag)
    }
    
    group2_stats = {
        'mean': np.mean(group2_bag),
        'std': np.std(group2_bag),
        'median': np.median(group2_bag),
        'n': len(group2_bag)
    }
    
    # Statistical tests
    # T-test for difference in means
    t_stat, t_p = stats.ttest_ind(group1_bag, group2_bag)
    
    # Mann-Whitney U test (non-parametric)
    u_stat, u_p = stats.mannwhitneyu(group1_bag, group2_bag, alternative='two-sided')
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(group1_bag) - 1) * np.var(group1_bag, ddof=1) + 
                         (len(group2_bag) - 1) * np.var(group2_bag, ddof=1)) / 
                        (len(group1_bag) + len(group2_bag) - 2))
    cohens_d = (group1_stats['mean'] - group2_stats['mean']) / pooled_std
    
    return {
        'group1_name': group1_name,
        'group2_name': group2_name,
        'group1_stats': group1_stats,
        'group2_stats': group2_stats,
        'difference_in_means': group1_stats['mean'] - group2_stats['mean'],
        't_test': {
            'statistic': t_stat,
            'p_value': t_p,
            'significant': t_p < 0.05
        },
        'mann_whitney_u': {
            'statistic': u_stat,
            'p_value': u_p,
            'significant': u_p < 0.05
        },
        'effect_size': {
            'cohens_d': cohens_d,
            'interpretation': 'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'
        }
    }


def comprehensive_brain_age_analysis(results_dict: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Perform comprehensive brain age analysis across all datasets.
    
    Args:
        results_dict (Dict[str, Dict]): Dictionary with dataset names as keys and 
                                       results containing 'true_ages' and 'predictions' as values
        
    Returns:
        Dict[str, Any]: Comprehensive analysis results
    """
    analysis_results = {
        'individual_metrics': {},
        'group_comparisons': {},
        'summary': {}
    }
    
    # Calculate metrics for each dataset
    for dataset_name, results in results_dict.items():
        if 'true_ages' in results and 'predictions' in results:
            y_true = np.array(results['true_ages'])
            y_pred = np.array(results['predictions'])
            
            # Calculate brain age gaps
            bag = y_pred - y_true
            
            # Store individual metrics
            analysis_results['individual_metrics'][dataset_name] = {
                'performance': evaluate_model_performance(y_true, y_pred),
                'bag_stats': {
                    'mean': np.mean(bag),
                    'std': np.std(bag),
                    'median': np.median(bag),
                    'n': len(bag)
                }
            }
    
    # Perform group comparisons
    datasets = list(results_dict.keys())
    
    # Define group types
    adhd_datasets = [d for d in datasets if 'adhd' in d.lower() and 'td' not in d.lower()]
    adhd_td_datasets = [d for d in datasets if 'adhd' in d.lower() and 'td' in d.lower()]
    asd_datasets = [d for d in datasets if 'asd' in d.lower() and 'td' not in d.lower()]
    asd_td_datasets = [d for d in datasets if 'asd' in d.lower() and 'td' in d.lower()]
    external_td_datasets = [d for d in datasets if 'td' in d.lower() and 'external' in d.lower()]
    
    # ADHD vs ADHD-TD comparison
    if adhd_datasets and adhd_td_datasets:
        adhd_bags = []
        adhd_td_bags = []
        
        for dataset in adhd_datasets:
            if dataset in analysis_results['individual_metrics']:
                bag = np.array(results_dict[dataset]['predictions']) - np.array(results_dict[dataset]['true_ages'])
                adhd_bags.extend(bag)
        
        for dataset in adhd_td_datasets:
            if dataset in analysis_results['individual_metrics']:
                bag = np.array(results_dict[dataset]['predictions']) - np.array(results_dict[dataset]['true_ages'])
                adhd_td_bags.extend(bag)
        
        if adhd_bags and adhd_td_bags:
            analysis_results['group_comparisons']['adhd_vs_adhd_td'] = compare_brain_age_gaps(
                np.array(adhd_bags), np.array(adhd_td_bags), "ADHD", "ADHD-TD"
            )
    
    # ASD vs ASD-TD comparison
    if asd_datasets and asd_td_datasets:
        asd_bags = []
        asd_td_bags = []
        
        for dataset in asd_datasets:
            if dataset in analysis_results['individual_metrics']:
                bag = np.array(results_dict[dataset]['predictions']) - np.array(results_dict[dataset]['true_ages'])
                asd_bags.extend(bag)
        
        for dataset in asd_td_datasets:
            if dataset in analysis_results['individual_metrics']:
                bag = np.array(results_dict[dataset]['predictions']) - np.array(results_dict[dataset]['true_ages'])
                asd_td_bags.extend(bag)
        
        if asd_bags and asd_td_bags:
            analysis_results['group_comparisons']['asd_vs_asd_td'] = compare_brain_age_gaps(
                np.array(asd_bags), np.array(asd_td_bags), "ASD", "ASD-TD"
            )
    
    # TD vs External TD comparison
    if external_td_datasets:
        td_bags = []
        external_td_bags = []
        
        # Get TD datasets (non-external)
        td_datasets = [d for d in datasets if 'td' in d.lower() and 'external' not in d.lower()]
        
        for dataset in td_datasets:
            if dataset in analysis_results['individual_metrics']:
                bag = np.array(results_dict[dataset]['predictions']) - np.array(results_dict[dataset]['true_ages'])
                td_bags.extend(bag)
        
        for dataset in external_td_datasets:
            if dataset in analysis_results['individual_metrics']:
                bag = np.array(results_dict[dataset]['predictions']) - np.array(results_dict[dataset]['true_ages'])
                external_td_bags.extend(bag)
        
        if td_bags and external_td_bags:
            analysis_results['group_comparisons']['td_vs_external_td'] = compare_brain_age_gaps(
                np.array(td_bags), np.array(external_td_bags), "TD", "External-TD"
            )
    
    # Summary statistics
    all_bags = []
    for dataset_name, results in results_dict.items():
        if 'true_ages' in results and 'predictions' in results:
            bag = np.array(results['predictions']) - np.array(results['true_ages'])
            all_bags.extend(bag)
    
    if all_bags:
        analysis_results['summary'] = {
            'overall_mean_bag': np.mean(all_bags),
            'overall_std_bag': np.std(all_bags),
            'overall_median_bag': np.median(all_bags),
            'total_n': len(all_bags)
        }
    
    return analysis_results


def train_lightning_model(model: ConvNetLightning, 
                         train_loader: DataLoader, 
                         val_loader: DataLoader,
                         output_dir: str,
                         max_epochs: int = 100,
                         patience: int = 10) -> str:
    """
    Train PyTorch Lightning model with best validation loss checkpointing.
    
    Args:
        model (ConvNetLightning): PyTorch Lightning model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        output_dir (str): Output directory for checkpoints
        max_epochs (int): Maximum number of epochs
        patience (int): Early stopping patience
        
    Returns:
        str: Path to best model checkpoint
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='best_model-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
        verbose=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=patience,
        verbose=True
    )
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        default_root_dir=output_dir,
        enable_progress_bar=True,
        enable_model_summary=True,
        accelerator='auto',
        devices='auto'
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Return path to best checkpoint
    return checkpoint_callback.best_model_path


def load_lightning_model_from_checkpoint(checkpoint_path: str, 
                                        input_channels: int = 246,
                                        dropout_rate: float = 0.6,
                                        learning_rate: float = 0.001) -> ConvNetLightning:
    """
    Load PyTorch Lightning model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to model checkpoint
        input_channels (int): Number of input channels
        dropout_rate (float): Dropout rate
        learning_rate (float): Learning rate
        
    Returns:
        ConvNetLightning: Loaded model
    """
    return ConvNetLightning.load_from_checkpoint(
        checkpoint_path,
        input_channels=input_channels,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )
