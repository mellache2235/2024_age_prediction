#!/usr/bin/env python3
"""
NKI Brain-Behavior Analysis Using Integrated Gradient Features

This script:
1. Loads pre-trained HCP-Dev age prediction models
2. Computes Integrated Gradients for NKI subjects
3. Performs brain-behavior correlation analyses using IG features

Cleaned version focused solely on IG computation and brain-behavior analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import scipy.stats as spss
import sys
import torch
import torch.nn as nn
import warnings
from captum.attr import IntegratedGradients
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

try:
    from statsmodels.stats.multitest import multipletests
except ImportError:
    multipletests = None

warnings.filterwarnings("ignore", category=FutureWarning)

# Constants
CUDA_SEED = 2344
NP_RANDOM_SEED = 652
K_FOLDS = 5
PERCENTILE = 0  # 0 = use all features
FEATURE_SCALE_FACTOR = 10000
MAX_AGE = 21  # Maximum age for IG feature analysis
N_NEURONS_LAYER1 = 32
N_NEURONS_LAYER2 = 32
KERNEL_LAYER1 = 5
KERNEL_LAYER2 = 7
DROPOUT_RATE = 0.6
USE_CUDA = True

# File paths
IG_FEATURES_CSV = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/feature_attribution/nki_cog_dev_wIDS_features_IG_convnet_regressor_single_model_fold_0.csv'
CAARS_FILE = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/FLUX/assessment_data/8100_CAARS-S-S_20191009.csv'
ACTUAL_AGES_NPZ = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/nki_updated/actual_nki_ages_oct25.npz'
PREDICTED_AGES_NPZ = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/nki_updated/predicted_nki_ages_oct25.npz'
OUTPUT_DIR = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/nki_rs_td_ig_analysis'


class ConvNet(nn.Module):
    """HCP-Dev age prediction model architecture."""
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(246, N_NEURONS_LAYER1, kernel_size=KERNEL_LAYER1, stride=1, bias=False),
            nn.BatchNorm1d(N_NEURONS_LAYER1),
            nn.PReLU(N_NEURONS_LAYER1),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(N_NEURONS_LAYER1, N_NEURONS_LAYER2, kernel_size=KERNEL_LAYER2, stride=1, bias=False),
            nn.BatchNorm1d(N_NEURONS_LAYER2),
            nn.PReLU(N_NEURONS_LAYER2),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.drop_out = nn.Dropout(p=DROPOUT_RATE)
        self.regressor = nn.Linear(N_NEURONS_LAYER2, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.drop_out(out)
        out = self.layer2(out)
        out = self.drop_out(out)
        out = out.mean(axis=2)
        out = self.regressor(out)
        return out


def get_and_analyze_features(data_all, labels_all, subjects, model_paths, output_csv):
    """
    Compute Integrated Gradients using pre-trained HCP-Dev models.
    
    Args:
        data_all: NKI timeseries data (N_subjects, 246 ROIs, timepoints)
        labels_all: Chronological ages
        subjects: Subject IDs
        model_paths: List of paths to HCP-Dev model files (one per fold)
        output_csv: Path to save IG features CSV
        
    Returns:
        DataFrame with IG features (columns = ROI labels, rows = subjects)
    """
    attr_data = np.zeros((K_FOLDS, data_all.shape[0], data_all.shape[1], data_all.shape[2]))
    cuda_available = USE_CUDA and torch.cuda.is_available()
    
    for fold_id in range(K_FOLDS):
        print(f"Computing IG with fold {fold_id} model: {model_paths[fold_id]}")
        
        model = ConvNet()
        if cuda_available:
            model.cuda()
        model.load_state_dict(torch.load(model_paths[fold_id]))
        model.eval()

        ig_tensor_data = torch.from_numpy(data_all).type(torch.FloatTensor)
        if cuda_available:
            ig_tensor_data = ig_tensor_data.cuda()
        ig_tensor_data.requires_grad_()

        ig = IntegratedGradients(model)
        
        # Process in batches of 10
        for i in range(0, len(ig_tensor_data), 10):
            batch_end = min(i + 10, len(ig_tensor_data))
            attr, delta = ig.attribute(
                ig_tensor_data[i:batch_end, :, :],
                target=0,
                return_convergence_delta=True
            )
            attr_data[fold_id, i:batch_end, :, :] = attr.detach().cpu().numpy()
            del attr, delta

    # Median across folds
    attr_data_median = np.median(attr_data, axis=0)
    
    # Filter by age
    age_mask = labels_all < MAX_AGE
    attr_data_filtered = attr_data_median[age_mask, :, :]
    
    # Subjects array should match labels_all length
    if len(subjects) != len(labels_all):
        print(f"  ⚠︎ Subject count mismatch: {len(subjects)} IDs vs {len(labels_all)} ages")
        # Use first N subjects
        subjects = subjects[:len(labels_all)] if len(subjects) > len(labels_all) else np.pad(subjects, (0, len(labels_all) - len(subjects)), constant_values='unknown')
    
    subjects_filtered = subjects[age_mask]
    
    # Median along time dimension
    attr_data_tsavg = np.median(attr_data_filtered, axis=2)
    
    # Load ROI labels
    with open('/oak/stanford/groups/menon/projects/cdla/2021_hcp_earlypsychosis/scripts/restfmri/classify/CNN1dPyTorch/brainnetome_roi_labels.txt') as f:
        roi_labels = [x.strip() for x in f.readlines()]
    
    # Create DataFrame
    if PERCENTILE == 0:
        features_df = pd.DataFrame(attr_data_tsavg, columns=roi_labels)
    else:
        # Top percentile features
        attr_data_grpavg = np.mean(attr_data_tsavg, axis=0)
        attr_data_sortedix = np.argsort(np.abs(attr_data_grpavg))
        attr_data_percentileix = np.argwhere(
            np.abs(np.sort(np.abs(attr_data_grpavg))) >= np.percentile(np.abs(attr_data_grpavg), PERCENTILE)
        )
        features_idcs = attr_data_sortedix[attr_data_percentileix]
        roi_labels_sorted = np.array(roi_labels)[attr_data_sortedix]
        features_df = pd.DataFrame(
            np.squeeze(attr_data_tsavg[:, features_idcs]),
            columns=roi_labels_sorted[attr_data_percentileix[::-1]]
        )
    
    features_df['subject_id'] = subjects_filtered
    features_df.to_csv(output_csv, index=False)
    print(f"✓ Saved IG features to {output_csv}")

    return features_df


def perform_bag_behavior_correlation(features_df, actual_ages, predicted_ages, subjids_all, visitids_all, behavior_file_name, behavior_data_dir, output_dir):
    """
    Correlate Brain Age Gap with behavioral scores (with FDR correction).
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute BAG
    bag = predicted_ages - actual_ages
    
    # Load behavioral data
    behv_datao = pd.read_csv(os.path.join(behavior_data_dir, behavior_file_name), skiprows=1)
    
    behv_datao_sel = pd.DataFrame()
    bag_sel = []
    
    for idix, id_ in enumerate(subjids_all):
        row = behv_datao[(behv_datao['ID'] == id_) & (behv_datao['VISIT'] == visitids_all[idix])]
        if not row.empty:
            behv_datao_sel = behv_datao_sel.append(row.iloc[0])
            bag_sel.append(bag[idix])
    
    if behv_datao_sel.empty:
        return
    
    behv_df = behv_datao_sel.select_dtypes('float64')
    behv_df = behv_df.reset_index(drop=True)
    bag_arr = np.array(bag_sel)
    
    print(f"\n{'='*60}")
    print(f"BAG-Behavior Correlations ({behavior_file_name}): N={len(bag_arr)}")
    print(f"{'='*60}")
    
    # Collect all p-values for FDR correction
    correlations = []
    for behav_col in behv_df.columns:
        valid_mask = ~np.isnan(bag_arr) & ~np.isnan(behv_df[behav_col])
        if valid_mask.sum() < 10:
            continue
        
        corr, pvalue = spss.spearmanr(bag_arr[valid_mask], behv_df[behav_col][valid_mask])
        correlations.append({
            'column': behav_col,
            'rho': corr,
            'p': pvalue,
            'n': valid_mask.sum(),
            'bag': bag_arr[valid_mask],
            'behav': behv_df[behav_col][valid_mask]
        })
    
    if not correlations:
        return
    
    # Apply FDR correction
    pvalues = [c['p'] for c in correlations]
    if multipletests:
        _, p_fdr, _, _ = multipletests(pvalues, method='fdr_bh')
    else:
        # Manual BH correction
        p_fdr = np.array(pvalues) * len(pvalues) / (np.arange(len(pvalues)) + 1)
        p_fdr = np.minimum.accumulate(p_fdr[::-1])[::-1]
        p_fdr = np.clip(p_fdr, 0, 1)
    
    # Report with FDR
    for idx, corr_data in enumerate(correlations):
        print(f"  {corr_data['column']}: ρ={corr_data['rho']:.3f}, p={corr_data['p']:.4f}, p_FDR={p_fdr[idx]:.4f}")
        
        if p_fdr[idx] < 0.05:
            import seaborn as sns
            fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=300)
            sns.set_style("white")
            sns.regplot(
                x=corr_data['bag'], y=corr_data['behav'], ci=None,
                scatter_kws={'color': 'navy', 'alpha': 0.6, 's': 40, 'edgecolor': 'w', 'linewidth': 0.5},
                line_kws={'color': 'red', 'linewidth': 2},
                ax=ax
            )
            
            p_text = r"$\mathit{P}_{FDR} < 0.001$" if p_fdr[idx] < 0.001 else rf"$\mathit{{P}}_{{FDR}} = {p_fdr[idx]:.3f}$"
            ax.text(0.95, 0.05, f"$\mathit{{R}}$ = {corr_data['rho']:.3f}\n{p_text}",
                   transform=ax.transAxes, ha='right', va='bottom', fontsize=12)
            
            ax.spines[['right', 'top']].set_visible(False)
            ax.set_xlabel("Brain Age Gap", fontsize=15, labelpad=10)
            ax.set_ylabel(f"{corr_data['column']}", fontsize=15, labelpad=10)
            ax.set_title("NKI-RS TD BAG-Behavior", fontsize=14, pad=10)
            plt.tight_layout(pad=1.2)
            
            safe_name = corr_data['column'].replace('/', '_').replace(' ', '_')
            png_path = os.path.join(output_dir, f'nki_BAG_{safe_name}_scatter.png')
            plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"    ✓ Saved: {png_path}")


def perform_brain_behavior_analyses(subjids_all, visitids_all, features_df, behavior_file_name, behavior_data_dir, output_dir):
    """
    Perform brain-behavior correlation using IG features.
    
    Args:
        subjids_all: Subject IDs
        visitids_all: Visit IDs
        features_df: DataFrame with IG features
        behavior_file_name: Behavioral data CSV filename
        behavior_data_dir: Directory containing behavioral CSVs
    """
    behv_datao = pd.read_csv(os.path.join(behavior_data_dir, behavior_file_name), skiprows=1)

    behv_datao_sel = pd.DataFrame()
    features_sel_ix = []
    
    # Match subjects and visits
    for idix, id_ in enumerate(subjids_all):
        row = behv_datao[(behv_datao['ID'] == id_) & (behv_datao['VISIT'] == visitids_all[idix])]
        if not row.empty:
            behv_datao_sel = behv_datao_sel.append(row.iloc[0])
            features_sel_ix.append(idix)

    if behv_datao_sel.empty:
        return
    
    features_df = features_df.iloc[features_sel_ix].copy()
    # Drop subject_id column (and any other non-ROI columns)
    if 'subject_id' in features_df.columns:
        features_df = features_df.drop('subject_id', axis=1)
    
    ids = None
    if len(behv_datao_sel.index) != 0:
        ids = np.asarray(behv_datao_sel.iloc[:, 0]).astype(str)
    
    behv_df = behv_datao_sel.select_dtypes('float64')
    behv_df = behv_df.reset_index(drop=True)
    features_df = features_df.reset_index(drop=True)

    # Remove NaN rows
    na_indices_df1 = features_df[features_df.isna().any(axis=1)].index
    na_indices_df2 = behv_df[behv_df.isna().any(axis=1)].index
    na_indices = na_indices_df1.union(na_indices_df2)
    features_df = features_df.drop(na_indices)
    behv_df = behv_df.drop(na_indices)
    
    if ids is not None:
        new_ids = np.delete(ids, na_indices)
    
    if behv_df.empty:
        return

    # Diagnostic: print shapes
    print(f"\n[DEBUG] Features shape: {features_df.shape}")
    print(f"[DEBUG] Behavior shape: {behv_df.shape}")
    print(f"[DEBUG] Features sample (first 3 values): {features_df.iloc[0, :3].values if not features_df.empty else 'empty'}")
    print(f"[DEBUG] Behavior sample (first 3 values): {behv_df.iloc[0, :3].values if not behv_df.empty else 'empty'}")
    
    if features_df.shape[0] < 30:
        print(f"  ⚠︎ Insufficient subjects ({features_df.shape[0]}) for reliable analysis; skipping.")
        return
    
    # Apply PCA dimensionality reduction (limit components to avoid overfitting)
    sc = StandardScaler()
    features_scaled = sc.fit_transform(features_df)
    
    # Use conservative component count: min(10, N/3)
    n_components = min(10, features_scaled.shape[0] // 3, features_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=0)
    features_pca = pca.fit_transform(features_scaled)
    
    print(f"[DEBUG] PCA shape: {features_pca.shape}, Explained var: {pca.explained_variance_ratio_.sum():.2%}")

    # Multivariate regression for each behavioral measure
    print('*' * 30)
    print(f'Performing Multivariate Brain-Behavior Analyses: {behavior_file_name}')
    
    multregression_results = {}
    for column in behv_df.columns:
        model = LinearRegression().fit(features_pca, behv_df[column])
        
        predictions = model.predict(features_pca)
        score = mean_squared_error(behv_df[column], predictions)
        corr, pvalue = spss.spearmanr(behv_df[column], predictions)
        
        print(f"  {column}: MSE={score:.4f}, Spearman r={corr:.3f}, p={pvalue:.4f}")
        
        if pvalue < 0.05 and new_ids is not None:
            plot_corr = pd.DataFrame({
                'ID': new_ids,
                'Behavior': behv_df[column],
                'Behavior_Predicted': predictions
            })
            safe_column = column.replace('/', '_').replace(' ', '_')
            plot_corr.to_csv(f'Features_HCP_Dev_model_{safe_column}_wIDS.csv', index=False)
            multregression_results[f'features-{column}'] = {'correlation': corr, 'p-value': pvalue}

    if multregression_results:
        multregression_results_df = pd.DataFrame(multregression_results).T
        print(multregression_results_df)


def get_data(path):
    """Load data from binary pickle file."""
    with open(path, "rb") as fp:
        data_dict = pickle.load(fp)
    return data_dict["X_train"], data_dict["X_test"], data_dict["Y_train"], data_dict["Y_test"]


def reshapeData(data):
    """Reshape from (subjects, timepoints, channels) to (subjects, channels, timepoints)."""
    no_subjs, no_ts, no_channels = data.shape
    data_reshape = np.empty((no_subjs, no_channels, no_ts))
    for subj in range(no_subjs):
        x_subj = data[subj, :, :]
        x_subj = np.transpose(x_subj)
        data_reshape[subj, :, :] = x_subj
    return data_reshape


if __name__ == '__main__':
    print("="*80)
    print("NKI-RS TD Brain-Behavior Analysis (IG Features)")
    print("="*80)
    
    # Load pre-computed IG features
    print(f"\nLoading IG features from: {IG_FEATURES_CSV}")
    features_df = pd.read_csv(IG_FEATURES_CSV)
    
    # Extract subject IDs and visitids from features CSV
    if 'subject_id' not in features_df.columns:
        print("✗ No subject_id column in IG CSV")
        sys.exit(1)
    
    subjids_all = features_df['subject_id'].astype(str).values
    print(f"  Loaded {len(subjids_all)} subjects with IG features")
    
    # Construct visitids by matching with demo file
    if os.name == 'nt':
        OAK = 'Z:/'
    else:
        OAK = '/oak/stanford/groups/menon/'
    
    behavior_data_dir = OAK + 'projects/mellache/2021_foundation_model/scripts/FLUX/assessment_data/'
    demo_datao = pd.read_csv(os.path.join(behavior_data_dir, '8100_Demos_20191009.csv'), skiprows=1)
    
    visitids_all = []
    for subj_id in subjids_all:
        visit_rows = demo_datao[demo_datao['ID'] == subj_id]
        if not visit_rows.empty:
            visitids_all.append(str(visit_rows.iloc[0]['VISIT']))
        else:
            visitids_all.append('unknown')
    
    print(f"  Matched {len([v for v in visitids_all if v != 'unknown'])} visit IDs")
    
    # Load brain ages
    print(f"\nLoading brain ages...")
    actual_ages_data = np.load(ACTUAL_AGES_NPZ)
    predicted_ages_data = np.load(PREDICTED_AGES_NPZ)
    
    actual_ages = actual_ages_data['actual']
    predicted_ages = predicted_ages_data['predicted']
    
    print(f"  Actual ages: {len(actual_ages)}")
    print(f"  Predicted ages: {len(predicted_ages)}")
    
    # Load CAARS
    print(f"\nLoading CAARS from: {CAARS_FILE}")
    caars_df = pd.read_csv(CAARS_FILE, skiprows=1)
    
    # Perform analyses
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print('\n' + '='*80)
    print('BRAIN-BEHAVIOR & BAG CORRELATION ANALYSES (CAARS)')
    print('='*80)
    
    # IG features → behavior
    perform_brain_behavior_analyses(
        subjids_all,
        visitids_all,
        features_df,
        '8100_CAARS-S-S_20191009.csv',
        behavior_data_dir,
        OUTPUT_DIR
    )
    
    # BAG → behavior
    perform_bag_behavior_correlation(
        features_df,
        actual_ages,
        predicted_ages,
        subjids_all,
        visitids_all,
        '8100_CAARS-S-S_20191009.csv',
        behavior_data_dir,
        OUTPUT_DIR
    )
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

# Remove all the old code below (data loading, IG computation, etc.)
'''
    # OLD CODE - keeping for reference but not executed
    datao = np.load(
        OAK + 'deriveddata/public/nkirs/restfmri/timeseries/group_level/brainnetome/normz/nkirs_site-nkirs_run-rest_645_brainnetome_mean_regMov-6param_wmcsf_dt1_bpf008-09_normz_246ROIs.pklz',
        allow_pickle=True
    )

    # Remove subjects with incomplete data
    indices_to_remove = [
        ind for ind, i in enumerate(datao["data"].values)
        if (len(i) != 900 or np.sum(np.isnan(i)) > 0)
    ]
    data_pklz = np.asarray([
        np.asarray(i) for ind, i in enumerate(datao["data"].values)
        if ind not in indices_to_remove
    ])
    datao_pklz = datao.drop(indices_to_remove)

    demo_datao = pd.read_csv(os.path.join(behavior_data_dir, DEMO_FILE), skiprows=1)

    # Load validation data across folds to get all NKI subjects
    data_all = []
    labels_all = []
    subjids_all = []
    visitids_all = []
    
    for fold_id in range(K_FOLDS):
        path = data_dir + f'fold_{fold_id}.bin'
        X_train, X_valid, Y_train, Y_valid = get_data(path)

        # Match subject IDs from pklz data (do this for every fold to maintain order)
        fold_subjids = []
        fold_visitids = []
        for row_id in range(Y_valid.shape[0]):
            datao_pklz_sel = datao_pklz[datao_pklz['age'] == Y_valid[row_id]]
            found = False
            for sel_id in range(datao_pklz_sel.shape[0]):
                if np.sum(datao_pklz_sel.iloc[sel_id].data - X_valid[row_id]) == 0:
                    fold_subjids.append(datao_pklz_sel.iloc[sel_id].subject_id)
                    visitid = demo_datao[
                        (demo_datao['ID'] == datao_pklz_sel.iloc[sel_id].subject_id) &
                        (demo_datao['AGE'] == Y_valid[row_id])
                    ]['VISIT'].values
                    fold_visitids.append(visitid[0] if len(visitid) > 0 else 'unknown')
                    found = True
                    break
            if not found:
                fold_subjids.append(f'unknown_{fold_id}_{row_id}')
                fold_visitids.append('unknown')
        
        subjids_all.extend(fold_subjids)
        visitids_all.extend(fold_visitids)

        X_valid = reshapeData(X_valid)
        
        if fold_id == 0:
            data_all = X_valid
            labels_all = Y_valid
        else:
            data_all = np.concatenate((data_all, X_valid))
            labels_all = np.concatenate((labels_all, Y_valid))

    # Load cached subject IDs from original location (read-only)
    original_ids_path = OAK + 'projects/ksupekar/2024_scratch/mellache/results/models/stdnn_age_allsubjs/'
    subjids_cache = os.path.join(original_ids_path, 'subjectids.txt')
    visitids_cache = os.path.join(original_ids_path, 'visitids.txt')
    
    if os.path.exists(subjids_cache) and os.path.exists(visitids_cache):
        print(f"  Loading cached subject IDs from: {original_ids_path}")
        subjids_all = np.loadtxt(subjids_cache, dtype=str)
        visitids_all = np.loadtxt(visitids_cache, dtype=str)
        print(f"  Loaded {len(subjids_all)} subjects from cache")
    else:
        # Use the in-memory lists if cache doesn't exist
        subjids_all = np.asarray(subjids_all).astype('str')
        visitids_all = np.asarray(visitids_all).astype('str')
        print(f"  Using {len(subjids_all)} subjects from fold matching")

    # Pre-trained HCP-Dev model paths
    model_paths = [
        f"/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/train_regression_models/dev/best_outer_fold_{fold_id}_hcp_dev_model_2_6_24.pt"
        for fold_id in range(K_FOLDS)
    ]
    
    # Compute IG features
    feature_csv = os.path.join(model_dir, f'nki_brain_features_IG_percentile_{PERCENTILE}.csv')
    
    if not os.path.exists(feature_csv):
        features_df = get_and_analyze_features(
            data_all,
            labels_all.flatten().astype('float64'),
            subjids_all,
            model_paths,
            feature_csv
        )
    else:
        print(f"Loading existing features from {feature_csv}")
        features_df = pd.read_csv(feature_csv)
    
    # Use only the subject IDs from the features DataFrame (age <21 subset)
    if 'subject_id' in features_df.columns:
        subjids_all = features_df['subject_id'].values
        # Try to load matching visitids for this subset from CAARS
        visitids_all = ['unknown'] * len(subjids_all)  # Default
        
        # If demo file has visit info, try to match
        for idx, subj_id in enumerate(subjids_all):
            age = labels_all[idx] if idx < len(labels_all) else None
            if age is not None:
                visit_rows = demo_datao[(demo_datao['ID'] == subj_id) & (demo_datao['AGE'] == age)]
                if not visit_rows.empty:
                    visitids_all[idx] = str(visit_rows.iloc[0]['VISIT'])
        
        print(f"  Using {len(subjids_all)} subjects from features CSV (age <{MAX_AGE})")
    
    # Load brain ages
    print(f"\nLoading brain ages...")
    actual_ages_data = np.load(ACTUAL_AGES_NPZ)
    predicted_ages_data = np.load(PREDICTED_AGES_NPZ)
    
    actual_ages = actual_ages_data['actual']
    predicted_ages = predicted_ages_data['predicted']
    
    # Load CAARS behavioral data
    print(f"\nLoading CAARS from: {CAARS_FILE}")
    caars_df = pd.read_csv(CAARS_FILE, skiprows=1)
    
    # Perform brain-behavior analyses on CAARS only
    print('\n' + '='*80)
    print('BRAIN-BEHAVIOR & BAG CORRELATION ANALYSES (CAARS)')
    print('='*80)
    
    # IG features → behavior
    perform_brain_behavior_analyses(
        subjids_all,
        visitids_all,
        features_df,
        '8100_CAARS-S-S_20191009.csv',
        os.path.dirname(CAARS_FILE),
        result_dir
    )
    
    # BAG → behavior with FDR correction
    perform_bag_behavior_correlation(
        features_df,
        actual_ages,
        predicted_ages,
        subjids_all,
        visitids_all,
        '8100_CAARS-S-S_20191009.csv',
        os.path.dirname(CAARS_FILE),
        result_dir
    )

