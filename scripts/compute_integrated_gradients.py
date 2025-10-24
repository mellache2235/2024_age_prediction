#!/usr/bin/env python3
"""
Compute Integrated Gradients for all cohorts using trained HCP-Dev model.

This script uses the get_and_analyze_features function to compute IG scores
and generate count data for network plots.

Usage:
    python scripts/compute_integrated_gradients.py --dataset nki_rs_td --fold 0
    python scripts/compute_integrated_gradients.py --dataset adhd200_adhd --fold 0
    python scripts/compute_integrated_gradients.py --dataset abide_asd --fold 0
"""

import os
import sys
import yaml
import logging
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / 'utils'))

from model_utils import load_lightning_model_from_checkpoint, AgeScaler
from data_utils import load_finetune_dataset, load_finetune_dataset_w_ids, reshape_data

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants from your function
FEATURE_SCALE_FACTOR = 1.0  # Adjust as needed
PERCENTILE = 50  # Use top 50% of features as requested
USE_CUDA = torch.cuda.is_available()

def get_and_analyze_features(data_all, labels_all, subjects, model_path, roi_labels_path=None):
    """
    Compute Integrated Gradients and analyze features using your function.
    
    Args:
        data_all: Input data (subjects x features x timepoints)
        labels_all: Age labels
        subjects: Subject IDs
        model_path: Path to trained model
        roi_labels_path: Path to ROI labels file
        
    Returns:
        pd.DataFrame: Features DataFrame with IG scores
    """
    attr_data = np.zeros((data_all.shape[0], data_all.shape[1], data_all.shape[2]))
    cuda_available = USE_CUDA and torch.cuda.is_available()
    
    print(f"Loading model: {model_path}")
    
    # Check if it's a PyTorch Lightning checkpoint or legacy PyTorch model
    if model_path.endswith('.ckpt'):
        # Load PyTorch Lightning model from checkpoint
        model = load_lightning_model_from_checkpoint(
            checkpoint_path=model_path,
            input_channels=data_all.shape[1],
            dropout_rate=0.6,
            learning_rate=0.001
        )
    else:
        # Load legacy PyTorch model
        from model_utils import ConvNet
        model = ConvNet(input_channels=data_all.shape[1], dropout_rate=0.6)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if cuda_available else 'cpu')))
    
    if cuda_available:
        model.cuda()
    model.eval()

    ig_tensor_data = torch.from_numpy(data_all).type(torch.FloatTensor)
    ig_tensor_labels = torch.from_numpy(labels_all).type(torch.FloatTensor)

    if cuda_available:
        ig_tensor_data = ig_tensor_data.cuda()
        ig_tensor_labels = ig_tensor_labels.cuda()

    from captum.attr import IntegratedGradients
    ig = IntegratedGradients(model)
    ig_tensor_data.requires_grad_()

    # Compute IG in batches of 10
    for idx, i in enumerate(range(0, len(ig_tensor_data), 10)):
        if i < len(ig_tensor_data) - 10:
            attr, delta = ig.attribute(ig_tensor_data[i:i + 10, :, :], target=0,
                                       return_convergence_delta=True)
        else:
            attr, delta = ig.attribute(ig_tensor_data[i:len(ig_tensor_data), :, :], target=0,
                                       return_convergence_delta=True)
        attr_data[i:i + 10, :, :] = attr[0].detach().cpu().numpy()
        del attr, delta

    # Take median across time dimension
    attr_data_tsavg = np.median(attr_data, axis=2) * FEATURE_SCALE_FACTOR
    attr_data_grpavg = np.mean(attr_data_tsavg, axis=0)   # Mean across all subjects

    # Sort ROIs by importance
    attr_data_sorted = np.sort(np.abs(attr_data_grpavg))
    attr_data_sortedix = np.argsort(np.abs(attr_data_grpavg))
    attr_data_percentileix = np.argwhere(
        np.abs(attr_data_sorted) >= np.percentile(np.abs(attr_data_sorted), PERCENTILE))
    features_idcs = attr_data_sortedix[attr_data_percentileix]

    # Load ROI labels
    if roi_labels_path and os.path.exists(roi_labels_path):
        with open(roi_labels_path) as f:
            roi_labels = f.readlines()
            roi_labels = [x.strip() for x in roi_labels]
    else:
        roi_labels = [f'ROI_{i}' for i in range(data_all.shape[1])]

    roi_labels_sorted = np.array(roi_labels)[attr_data_sortedix]
    print(f'Age prediction: {PERCENTILE}% percentile Channel/ROI by descending order of importance')
    print(*roi_labels_sorted[attr_data_percentileix[::-1]], sep='\n')

    # Create features DataFrame
    if PERCENTILE == 0:
        features_df = pd.DataFrame(attr_data_tsavg, columns=roi_labels)
    else:
        features_df = pd.DataFrame(np.squeeze(attr_data_tsavg[:, features_idcs]),
                                   columns=roi_labels_sorted[attr_data_percentileix[::-1]])

    features_df['subject_id'] = np.asarray(subjects)
    
    return features_df

def generate_count_data_from_ig_scores(features_df, percentile=50):
    """
    Generate count data from IG scores DataFrame.
    
    Args:
        features_df: DataFrame with IG scores
        percentile: Percentile threshold for top features
        
    Returns:
        pd.DataFrame: Count data with attribution and region columns
    """
    # Get ROI columns (exclude subject_id)
    roi_columns = [col for col in features_df.columns if col != 'subject_id']
    ig_scores = features_df[roi_columns].values
    
    n_subjects, n_rois = ig_scores.shape
    logging.info(f"Processing {n_subjects} subjects and {n_rois} ROIs")
    
    # Calculate attribution (count of times each ROI is in top percentile)
    attributions = []
    
    for roi_idx in range(n_rois):
        roi_scores = np.abs(ig_scores[:, roi_idx])  # Use absolute values
        threshold = np.percentile(roi_scores, percentile)
        count = np.sum(roi_scores >= threshold)
        attribution = count / n_subjects  # Normalize by number of subjects
        attributions.append(attribution)
    
    # Create DataFrame in the format expected by the R script
    df_plot = pd.DataFrame({
        'attribution': attributions,
        'region': roi_columns
    })
    
    return df_plot

def main():
    """Compute Integrated Gradients for specified dataset."""
    parser = argparse.ArgumentParser(
        description="Compute Integrated Gradients for brain age prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Datasets:
  nki_rs_td      - NKI-RS TD cohort
  adhd200_adhd   - ADHD-200 ADHD cohort
  cmihbn_adhd    - CMI-HBN ADHD cohort
  adhd200_td     - ADHD-200 TD cohort
  cmihbn_td      - CMI-HBN TD cohort
  abide_asd      - ABIDE ASD cohort
  stanford_asd   - Stanford ASD cohort

Examples:
  # Compute IG for NKI-RS TD cohort
  python compute_integrated_gradients.py --dataset nki_rs_td --fold 0
  
  # Compute IG with custom percentile threshold
  python compute_integrated_gradients.py --dataset adhd200_adhd --fold 0 --percentile 80
        """
    )
    
    parser.add_argument("--dataset", type=str, required=True,
                       choices=['nki_rs_td', 'adhd200_adhd', 'cmihbn_adhd', 'adhd200_td', 'cmihbn_td', 'abide_asd', 'stanford_asd'],
                       help="Dataset to compute IG for (see available options below)")
    parser.add_argument("--fold", type=int, default=0, 
                       help="HCP-Dev model fold to use (0-4, default: 0)")
    parser.add_argument("--model_dir", type=str, default="results/brain_age_models",
                       help="Directory containing trained models")
    parser.add_argument("--roi_labels", type=str, 
                       default="/oak/stanford/groups/menon/projects/cdla/2021_hcp_earlypsychosis/scripts/restfmri/classify/CNN1dPyTorch/brainnetome_roi_labels.txt",
                       help="Path to ROI labels file")
    parser.add_argument("--percentile", type=float, default=50,
                       help="Percentile threshold for top features (default: 50, range: 0-100)")
    
    args = parser.parse_args()
    
    # Load configuration
    with open('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    output_dir = f"results/integrated_gradients/{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load trained model from config (use legacy PyTorch model for now)
    model_path = config['existing_models']['legacy_model_path']
    if not os.path.exists(model_path):
        logging.error(f"Legacy model not found: {model_path}")
        return
    
    # Get ROI labels path from config
    roi_labels_path = config['network_analysis']['roi_labels_path']
    
    # Load external dataset
    external_data_path = config['data_paths']['external_datasets'][args.dataset]
    # Load external data with IDs
    X_train, X_test, id_train, Y_train, Y_test, id_test = load_finetune_dataset_w_ids(external_data_path)
    
    # Combine train and test data
    X_combined = np.concatenate([X_train, X_test], axis=0)
    Y_combined = np.concatenate([Y_train, Y_test], axis=0)
    id_combined = id_train + id_test
    
    # Reshape data
    X_combined = reshape_data(X_combined)
    
    # Use actual subject IDs from .bin file
    subjects = id_combined
    
    # Compute Integrated Gradients using your function
    logging.info(f"Computing Integrated Gradients for {args.dataset} using HCP-Dev model...")
    features_df = get_and_analyze_features(
        X_combined, Y_combined, subjects, model_path, roi_labels_path
    )
    
    # Save IG scores CSV
    ig_csv_file = os.path.join(output_dir, f"{args.dataset}_features_IG_convnet_regressor_trained_on_hcp_dev_fold_{args.fold}.csv")
    features_df.to_csv(ig_csv_file, index=False)
    
    # Generate count data
    count_data = generate_count_data_from_ig_scores(features_df, args.percentile)
    
    # Save count data CSV
    count_csv_file = os.path.join(output_dir, f"{args.dataset}_count_data_top{args.percentile}.csv")
    count_data.to_csv(count_csv_file, index=False)
    
    logging.info(f"Integrated Gradients computed for {args.dataset}!")
    logging.info(f"IG scores shape: {features_df.shape}")
    logging.info(f"IG scores saved to: {ig_csv_file}")
    logging.info(f"Count data saved to: {count_csv_file}")

if __name__ == "__main__":
    main()
