"""
Data loading and preprocessing utilities for age prediction analysis.

This module provides functions for loading, preprocessing, and managing
neuroimaging and behavioral data across different datasets.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def load_finetune_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load numpy dictionary for finetune data from given path.

    Args:
        path (str): Path to the pickle file containing the data dictionary

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            X_train, X_test, Y_train, Y_test arrays
    """
    with open(path, "rb") as fp:
        data_dict = pickle.load(fp)
    
    return (data_dict["X_train"], data_dict["X_test"], 
            data_dict["Y_train"], data_dict["Y_test"])


def load_finetune_dataset_w_sites(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                      np.ndarray, np.ndarray, np.ndarray]:
    """
    Load numpy dictionary for finetune data with site information.

    Args:
        path (str): Path to the pickle file containing the data dictionary

    Returns:
        Tuple containing X_train, X_test, site_train, Y_train, Y_test, site_test
    """
    with open(path, "rb") as fp:
        data_dict = pickle.load(fp)
    
    return (data_dict["X_train"], data_dict["X_test"], data_dict["site_train"],
            data_dict["Y_train"], data_dict["Y_test"], data_dict["site_test"])


def load_finetune_dataset_w_ids(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                    np.ndarray, np.ndarray, np.ndarray]:
    """
    Load numpy dictionary for finetune data with subject IDs.

    Args:
        path (str): Path to the pickle file containing the data dictionary

    Returns:
        Tuple containing X_train, X_test, id_train, Y_train, Y_test, id_test
    """
    with open(path, "rb") as fp:
        data_dict = pickle.load(fp)
    
    return (data_dict["X_train"], data_dict["X_test"], data_dict["id_train"],
            data_dict["Y_train"], data_dict["Y_test"], data_dict["id_test"])


def remove_nans(data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove samples with NaN labels from data and labels arrays.

    Args:
        data (np.ndarray): Input data array
        labels (np.ndarray): Labels array

    Returns:
        Tuple[np.ndarray, np.ndarray]: Cleaned data and labels arrays
    """
    ix_nan = np.isnan(labels)
    labels_clean = labels[~ix_nan]
    data_clean = data[~ix_nan, :, :]
    return data_clean, labels_clean


def reshape_data(data: np.ndarray) -> np.ndarray:
    """
    Reshape data from (subjects, timepoints, channels) to (subjects, channels, timepoints).

    Args:
        data (np.ndarray): Input data with shape (n_subjects, n_timepoints, n_channels)

    Returns:
        np.ndarray: Reshaped data with shape (n_subjects, n_channels, n_timepoints)
    """
    no_subjs, no_ts, no_channels = data.shape
    data_reshape = np.empty((no_subjs, no_channels, no_ts))
    
    for subj in range(no_subjs):
        x_subj = data[subj, :, :]
        x_subj = np.transpose(x_subj)
        data_reshape[subj, :, :] = x_subj
    
    return data_reshape


def add_zeros(X: np.ndarray, k: int) -> np.ndarray:
    """
    Add zeros between timepoints in the data.

    Args:
        X (np.ndarray): Input data with shape (N, T, C)
        k (int): Number of zeros to add between timepoints

    Returns:
        np.ndarray: Extended data with zeros inserted
    """
    N, T, C = X.shape
    X_extend = np.zeros((N, k * T, C))
    X_extend[:, 0:k*T:k, :] = X
    return X_extend


def detect_roi_columns(df: pd.DataFrame, 
                      non_roi_candidates: Optional[set] = None) -> list:
    """
    Detect ROI columns in a DataFrame by excluding non-ROI columns.

    Args:
        df (pd.DataFrame): Input DataFrame
        non_roi_candidates (set, optional): Set of column names to exclude

    Returns:
        list: List of ROI column names
    """
    if non_roi_candidates is None:
        non_roi_candidates = {
            "subject_id", "participant_id", "subid", "sub_id", "id",
            "Unnamed: 0", "Unnamed: 0.1", "Unnamed: 0.2"
        }
    
    cols = [c for c in df.columns if c not in non_roi_candidates]
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return numeric_cols


def load_roi_labels(roi_labels_path: str) -> list:
    """
    Load ROI labels from a text file.

    Args:
        roi_labels_path (str): Path to the ROI labels text file

    Returns:
        list: List of ROI labels
    """
    with open(roi_labels_path, 'r') as f:
        roi_labels = f.readlines()
        roi_labels = [x.strip() for x in roi_labels]
    return roi_labels


def validate_data_consistency(data: np.ndarray, labels: np.ndarray, 
                            expected_timepoints: Optional[int] = None) -> bool:
    """
    Validate data consistency and remove problematic samples.

    Args:
        data (np.ndarray): Input data
        labels (np.ndarray): Labels array
        expected_timepoints (int, optional): Expected number of timepoints

    Returns:
        bool: True if data is consistent
    """
    if len(data) != len(labels):
        print(f"Warning: Data length ({len(data)}) != labels length ({len(labels)})")
        return False
    
    if expected_timepoints is not None:
        inconsistent_indices = []
        for i, ts_data in enumerate(data):
            if len(ts_data) != expected_timepoints:
                inconsistent_indices.append(i)
        
        if inconsistent_indices:
            print(f"Warning: {len(inconsistent_indices)} samples have inconsistent timepoints")
            return False
    
    # Check for NaN values
    nan_indices = []
    for i, ts_data in enumerate(data):
        if np.sum(np.isnan(ts_data)) > 0:
            nan_indices.append(i)
    
    if nan_indices:
        print(f"Warning: {len(nan_indices)} samples contain NaN values")
        return False
    
    return True


def create_data_splits(data: np.ndarray, labels: np.ndarray, 
                      train_ratio: float = 0.7, val_ratio: float = 0.15,
                      random_state: int = 42) -> Dict[str, np.ndarray]:
    """
    Create train/validation/test splits from data.

    Args:
        data (np.ndarray): Input data
        labels (np.ndarray): Labels
        train_ratio (float): Proportion for training set
        val_ratio (float): Proportion for validation set
        random_state (int): Random seed for reproducibility

    Returns:
        Dict[str, np.ndarray]: Dictionary containing train/val/test splits
    """
    np.random.seed(random_state)
    n_samples = len(data)
    indices = np.random.permutation(n_samples)
    
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    return {
        'train_data': data[train_indices],
        'train_labels': labels[train_indices],
        'val_data': data[val_indices],
        'val_labels': labels[val_indices],
        'test_data': data[test_indices],
        'test_labels': labels[test_indices]
    }
