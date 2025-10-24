"""
Model utilities for age prediction analysis.

This module provides neural network models, training functions, and model evaluation utilities.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import math
from typing import Dict, List, Tuple, Optional, Any
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats


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
    Evaluate model performance with multiple metrics.

    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values

    Returns:
        Dict[str, float]: Dictionary containing performance metrics
    """
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = r2_score(y_true, y_pred)
    corr, p_value = stats.pearsonr(y_true, y_pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'correlation': corr,
        'p_value': p_value
    }
