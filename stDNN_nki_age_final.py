# use pyks
# ml cuda/12.4
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import pandas as pd
import pickle
import random
import scipy.stats as spss
import seaborn as sns
import sys
import time
import torch
import torch.nn as nn
#import umap
import wandb
import warnings
from captum.attr import IntegratedGradients
from collections import Counter
from nilearn import image, plotting
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from torch.utils.data import DataLoader, TensorDataset
import pdb

warnings.filterwarnings("ignore", category=FutureWarning)

# Constants
CUDA_SEED = 2344
NP_RANDOM_SEED = 652
PYTHON_RANDOM_SEED = 819
CLIP_VALUE = 0
N_ITERS = 100
K_FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 0.0006614540317896509
WEIGHT_DECAY = 0.0001
N_NEURONS_LAYER1 = 32
N_NEURONS_LAYER2 = 32
KERNEL_LAYER1 = 5
KERNEL_LAYER2 = 7
DROPOUT_RATE = 0.6
#PERCENTILE = 95
PERCENTILE = 0
FEATURE_SCALE_FACTOR = 10000
MAX_AGE = 21 #### Indicates which ages we will look at brain features computed by IG for

USE_CUDA = True
TRAIN_MODEL = True
TEST_MODEL = False
BASE_FNAME = 'regression_from_scratch_age_fold_'
ATLAS_NIFTI = '/oak/stanford/groups/menon/projects/sryali/2019_DNN/scripts/features/BN_Atlas_246_2mm.nii'
DEMO_FILE = '8100_Demos_20191009.csv'
BEHAVIOR_FILE = '8100_MoCA_20191009.csv'
FEATURE_FILE_PREFIX = 'hcp_dev_kfold_all_group_age_lessthan21_site_percentile_' + str(PERCENTILE)
OUTPUT_FILE_PREFIX = 'nki_age_prediction_output'


class ConvNet(nn.Module):
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


def train():
    cuda_available = USE_CUDA and torch.cuda.is_available()
    with wandb.init() as run:
        config = run.config
        rmse = np.zeros(K_FOLDS)
        pearsonr = np.zeros(K_FOLDS)
        mae = np.zeros(K_FOLDS)

        fname_model = {}
        for fold_id in np.arange(K_FOLDS):
            fname_model[fold_id] = result_dir + 'models/XXXFineTuneAll' + BASE_FNAME + str(fold_id) + '.pt'
            model = ConvNet(config)

            if cuda_available:
                model.cuda()

            criterion = nn.MSELoss()
            optimizer = torch.optim.AdamW(model.parameters(),
                                          lr=config.learning_rate,
                                          weight_decay=config.weight_decay)

            loss_list, acc_list = [], []
            best_val_loss = 1000000
            best_val_pearsonr = -9999
            best_val_mae = 1000000
            early_stop_counter = 0
            clip_value = config.clip_value

            for epoch in range(config.epochs):
                model.train()
                for data_ts, labels in train_loader[fold_id]:
                    # labels = labels.long()
                    if cuda_available:
                        data_ts, labels = data_ts.cuda(), labels.cuda()

                    with torch.cuda.amp.autocast():
                        outputs = model(data_ts)
                        loss = torch.sqrt(criterion(outputs, labels))
                        loss_list.append(loss.item())
                        # print(loss.item())
                        optimizer.zero_grad()
                        loss.backward()
                        if clip_value != 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
                        optimizer.step()

                        wandb.log({"Train Loss": loss.item(), "Epoch": epoch})

                    val_loss, val_pearsonr, val_mae = test(model, val_loader[fold_id], cuda_available)
                    wandb.log({"Val Loss": val_loss, "Best Val Loss": best_val_loss, "Epoch": epoch})
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_val_pearsonr = val_pearsonr
                        best_val_mae = val_mae
                        # torch.save(model.state_dict(), fname_model[fold_id])
                        torch.save(model.state_dict(),
                                   os.path.join(wandb.run.dir, 'XXFineTuneAll' + BASE_FNAME + str(fold_id) + '.pt'))
                        # wandb.save(fname_model[fold_id], policy="now")
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1

                    if val_loss == 0.0 or early_stop_counter == 25:
                        break
            rmse[fold_id] = best_val_loss
            pearsonr[fold_id] = best_val_pearsonr
            mae[fold_id] = best_val_mae
            print(f'Best Validation RMSE: {best_val_loss:.2f}')
            print(f'Best Validation r: {best_val_pearsonr:.2f}')
            print(f'Best Validation mae: {best_val_mae:.2f}')

        mean_rmse = np.mean(rmse)
        mean_pearsonr = np.mean(pearsonr)
        mean_mae = np.mean(mae)
        wandb.log({"Mean MSE": mean_rmse, "Mean r": mean_pearsonr, "Mean MAE": mean_mae})


def test(model, data_loader, cuda_available):
    if not isinstance(model, ConvNet):
        fname_model = model
        model = ConvNet()
        if cuda_available:
            model.cuda()
        # model.load_state_dict(torch.load(fname_model, map_location=torch.device('cpu')))
        model.load_state_dict(torch.load(fname_model))

    model.eval()

    criterion = nn.MSELoss()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for data_ts, labels in data_loader:
                # labels = labels.long()
                if cuda_available:
                    data_ts, labels = data_ts.cuda(), labels.cuda()

                outputs = model(data_ts)
                loss = torch.sqrt(criterion(outputs, labels))
                test_pearsonr = np.corrcoef(outputs.cpu().numpy().flatten(), labels.cpu().numpy().flatten())[0, 1]
                test_mae = np.mean(np.abs(outputs.cpu().numpy().flatten() - labels.cpu().numpy().flatten()))

        test_loss = loss.item()

    return test_loss, test_pearsonr, test_mae, outputs

##### labels_all contains original,unscaled ages. val_loader contains scaled ages using StandardScaler(), mae calculated based on scaled ages !!!
def test_all_folds(labels_all, scalar_all, subjids_all, meanfd_all):
    cuda_available = USE_CUDA and torch.cuda.is_available()
    val_rmse = np.zeros(K_FOLDS)
    val_pearsonr = np.zeros(K_FOLDS)
    val_mae = np.zeros(K_FOLDS)
    val_predicted = []

    for fold_id in np.arange(K_FOLDS):
        fname_model = f"/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/train_regression_models/dev/best_outer_fold_{fold_id}_hcp_dev_model_2_6_24.pt"
        model = ConvNet()
        if cuda_available:
            model.cuda()
        # model.load_state_dict(torch.load(fname_model, map_location=torch.device('cpu')))
        model.load_state_dict(torch.load(fname_model))
        val_rmse[fold_id], val_pearsonr[fold_id], val_mae[fold_id], val_predicted_tmp = test(model, val_loader[fold_id],
                                                                                             cuda_available)
        if cuda_available:
            val_predicted_tmp = val_predicted_tmp.cpu()
        val_predicted.append(np.squeeze(scaler_all[fold_id].inverse_transform(val_predicted_tmp)))

    print(f'Mean Validation RMSE: {np.mean(val_rmse):.2f}')
    print(f'Mean Validation r: {np.mean(val_pearsonr):.2f}')
    print(f'Mean Validation mae: {np.mean(val_mae):.2f}')

    df = pd.DataFrame({
        'subject_id': np.squeeze(np.array(subjids_all)),
        'age_observed': labels_all,
        'age_predicted': np.squeeze(np.concatenate(val_predicted))
        # 'mean_fd': np.squeeze(np.array(meanfd_all))
    })
    # Write the DataFrame to a CSV file
    df.to_csv('nki_age_prediction_output.csv', index=False)

    return np.squeeze(np.concatenate(val_predicted)), labels_all


def get_and_analyze_features(data_all, labels_all,subjects):
    attr_data = np.zeros((K_FOLDS, data_all.shape[0], data_all.shape[1], data_all.shape[2]))
    cuda_available = USE_CUDA and torch.cuda.is_available()
    for fold_id in np.arange(K_FOLDS):
        fname_model = f"/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/train_regression_models/dev/best_outer_fold_{fold_id}_hcp_dev_model_2_6_24.pt"   
        print(fname_model)
        model = ConvNet()
        if cuda_available:
            model.cuda()
        # model.load_state_dict(torch.load(fname_model, map_location=torch.device('cpu')))
        model.load_state_dict(torch.load(fname_model))

        ig_tensor_data = torch.from_numpy(data_all).type(torch.FloatTensor)
        ig_tensor_labels = torch.from_numpy(labels_all).type(torch.FloatTensor)

        if cuda_available:
            ig_tensor_data = ig_tensor_data.cuda()
            ig_tensor_labels = ig_tensor_labels.cuda()

        ig = IntegratedGradients(model)
        ig_tensor_data.requires_grad_()

        for idx, i in enumerate(range(0, len(ig_tensor_data), 10)):
            if i < len(ig_tensor_data) - 10:
                attr, delta = ig.attribute(ig_tensor_data[i:i + 10, :, :], target=0,
                                           return_convergence_delta=True)
            else:
                attr, delta = ig.attribute(ig_tensor_data[i:len(ig_tensor_data), :, :], target=0,
                                           return_convergence_delta=True)
            attr_data[fold_id, i:i + 10, :, :] = attr[0].detach().cpu().numpy()
            del attr, delta

    attr_data_tmp = np.median(attr_data, 0)
    attr_data_median = np.squeeze(attr_data_tmp)
    # attr_data_median = np.squeeze(attr_data[BEST_FOLD_ID]) # best fold
    ageix = np.squeeze(np.argwhere(labels_all < MAX_AGE))  # only analyze certain age attributes
    attr_data_median = attr_data_median[ageix, :, :]
    ix = labels_all < MAX_AGE
    subjects_dev = subjects[ix]
    # find most discriminating ROIs between the two groups
    attr_data_tsavg = np.median(attr_data_median, axis=2) #### Median along time dimension
    attr_data_grpavg = np.mean(attr_data_tsavg, axis=0)   #### Mean across all subjects

    attr_data_sorted = np.sort(np.abs(attr_data_grpavg))  #### Sort ROIs (after we've taken mean of all subjects)
    attr_data_sortedix = np.argsort(np.abs(attr_data_grpavg))
    attr_data_percentileix = np.argwhere(
        np.abs(attr_data_sorted) >= np.percentile(np.abs(attr_data_sorted), PERCENTILE))  ### Percentile set to 95, this get top 5% features
    features_idcs = attr_data_sortedix[attr_data_percentileix]
    #features_idcs = attr_data_sortedix
    features = np.abs(attr_data_grpavg)

    with open(
            '/oak/stanford/groups/menon/projects/cdla/2021_hcp_earlypsychosis/scripts/restfmri/classify/CNN1dPyTorch/brainnetome_roi_labels.txt') as f:
        roi_labels = f.readlines()
        roi_labels = [x.strip() for x in roi_labels]

    roi_labels_sorted = np.array(roi_labels)[attr_data_sortedix]
    print('Age prediction: {}% percentile Channel/ROI by descending order of importance'.format(PERCENTILE))
    print(*roi_labels_sorted[attr_data_percentileix[::-1]], sep='\n')

    atlas_volume = image.load_img(ATLAS_NIFTI)
    roi_nifti = image.math_img('img-img', img=atlas_volume)
    img_data = atlas_volume.get_fdata()

    for idx in features_idcs:
        roi_idx = np.where(img_data == idx + 1, features[idx] * FEATURE_SCALE_FACTOR, 0)
        roi_img = image.new_img_like(roi_nifti, roi_idx)
        roi_nifti = image.math_img('img1+img2', img1=roi_nifti, img2=roi_img)

    roi_nifti.to_filename(FEATURE_FILE_PREFIX + '.nii.gz')
    #display = plotting.plot_stat_map(roi_nifti,display_mode='ortho',cut_coords=(0,0,0),colorbar=True,cmap='inferno',output_file='nki_brain_region_importance_IG_v3.png') 
    #for ax in display.axes.values():
    #    ax.axis('off')
    #np.save(os.path.join(model_dir, FEATURE_FILE_PREFIX + '.npy'), attr_data_tsavg[:, features_idcs])
    if PERCENTILE == 0:
        features_df = pd.DataFrame(attr_data_tsavg, columns=roi_labels)
    else:
        features_df = pd.DataFrame(np.squeeze(attr_data_tsavg[:, features_idcs]),
                                   columns=roi_labels_sorted[attr_data_percentileix[::-1]])
    
    features_df['subject_id'] = np.asarray(subjects_dev)
    features_df.to_csv('nki_brain_features_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS.csv')

    return features_df


def perform_resilience_behavior_analyses(subjids_all, visitids_all, observed_age, predicted_age, behavior_file_name):
    subjids_all = np.asarray(subjids_all)[labels_all < MAX_AGE]
    visitids_all = np.asarray(visitids_all)[labels_all < MAX_AGE]

    br = predicted_age[labels_all < MAX_AGE] - observed_age[labels_all < MAX_AGE]
    # br = observed_age[labels_all < MAX_AGE]
    br_label = [0 if x <= 0 else 1 for x in br]
    resilience_df = pd.DataFrame({'BR': br, 'br_label': br_label})

    behv_datao = pd.read_csv(os.path.join(behavior_data_dir, behavior_file_name), skiprows=1)
    
    behv_datao_sel_df = pd.DataFrame()
    resilience_sel_ix = []
    for idix, id_ in enumerate(subjids_all):
        row = behv_datao[(behv_datao['ID'] == id_) & (behv_datao['VISIT'] == visitids_all[idix])]
        if not row.empty:
            behv_datao_sel_df = behv_datao_sel_df.append(row.iloc[0])
            resilience_sel_ix.append(idix)

    if behv_datao_sel_df.empty:
        return

    resilience_df = resilience_df.iloc[resilience_sel_ix]

    behv_df = behv_datao_sel_df.select_dtypes('float64')
    behv_df = behv_df.reset_index(drop=True)
    resilience_df = resilience_df.reset_index(drop=True)

    # Perform ttest-analyses
    results = {}
    for behv_col in behv_df.columns:
        if len(behv_df[behv_col].dropna().unique()) == 2:  # check if binary variable
            # stat, p_value, dof, expected = spss.chi2_contingency(pd.crosstab(resilience_df['br_label'], behv_df[behv_col]))
            stat, p_value = spss.fisher_exact(pd.crosstab(resilience_df['br_label'], behv_df[behv_col]))
        else:
            ### T-test for BAG > 0 vs BAG < 0. Brain looks older than it should be versus brain looks younger than it should be
            ### ***** T-test is done using Brain Age Gap and not chronological age !!!!!
            stat, p_value = spss.ttest_ind(behv_df[resilience_df['br_label'] == 0][behv_col],
                                           behv_df[resilience_df['br_label'] == 1][behv_col], equal_var=False,
                                           nan_policy='omit')  # Assuming unequal variance
        if p_value < 0.05:
            results[behv_col] = {'statistic': stat, 'p_value': p_value}

    results_df = pd.DataFrame(results).T
    print(results_df)

    # Univariate analyses: Compute Spearman correlation and p-value for each pair of columns
    print('*' * 30)
    # print('Performing Univariate Resilience Behavior Analyses')
    correlation_results = {}
    for behv_col in behv_df.columns:
        # corr, p_value = spss.spearmanr(resilience_df['BR'], behv_df[behv_col], nan_policy='omit')

        validix = resilience_df['BR'].notna() & behv_df[behv_col].notna()
        if not validix.empty and validix.sum() > 2:
            corr, p_value = spss.pearsonr(resilience_df['BR'][validix], behv_df[behv_col][validix])
            # corr, p_value = spss.spearmanr(resilience_df['BR'][validix], behv_df[behv_col][validix])
            if p_value < 0.05:
                plot_corr = pd.DataFrame({'BR':resilience_df['BR'][validix],'Behavior':behv_df[behv_col][validix]})
                plot_corr.to_csv(f'HCP_DEV_MODEL_BR_{behv_col}.csv',index=False)
                correlation_results[f'BR-{behv_col}'] = {'correlation': corr, 'p-value': p_value}
        #     print(behv_col)
        #     print(corr)
        #     print(p_value)

    correlation_results_df = pd.DataFrame(correlation_results).T
    print(correlation_results_df)


def perform_brain_resilience_analyses(subjids_all, visitids_all, features_df, observed_age, predicted_age):
    subjids_all = np.asarray(subjids_all)[labels_all < MAX_AGE]
    visitids_all = np.asarray(visitids_all)[labels_all < MAX_AGE]
    features_df = features_df.drop(features_df.columns[0], axis=1)  # ffirst column contains index so drop
    br = predicted_age[labels_all < MAX_AGE] - observed_age[labels_all < MAX_AGE]
    br_label = [0 if x <= 0 else 1 for x in br]
    brain_resilience_df = pd.DataFrame(br, columns=['BR'])

    # Perform ttest-analyses
    results = {}
    features_df['br_label'] = br_label
    for feature_col in features_df.columns:
        stat, p_value = spss.ttest_ind(features_df[features_df['br_label'] == 0][feature_col],
                                       features_df[features_df['br_label'] == 1][feature_col],
                                       equal_var=False)  # Assuming unequal variance
        results[feature_col] = {'statistic': stat, 'p_value': p_value}

    results_df = pd.DataFrame(results).T
    print(results_df)
    features_df = features_df.drop('br_label', axis=1)

    # Perform umap-analyses
    # reducer = umap.UMAP(random_state=42)
    # embedding = reducer.fit_transform(features_df.to_numpy())
    # plt.figure(figsize=(12, 10))
    # # plt.scatter(embedding[:, 0], embedding[:, 1], c=br, cmap='RdBu', vmin=-8, vmax=8, s=50)
    # plt.scatter(embedding[:, 0], embedding[:, 1], c=br_label, cmap='Spectral', s=50)
    # plt.gca().set_aspect('equal', 'datalim')
    # # plt.colorbar(boundaries=np.arange(4) - 0.5).set_ticks(np.arange(3))
    # plt.title('UMAP projection of the Age predictive features', fontsize=24)
    # plt.savefig('umap_lessthan60.jpg')
    # plt.show()

    # Univariate analyses: Compute Spearman correlation and p-value for each pair of columns
    print('*' * 10)
    print('Performing Univariate Brain Resilience Analyses')
    correlation_results = {}
    for col1 in brain_resilience_df.columns: #### Brain Age Gap predicted using 1DCNN
        for col2 in features_df.columns:    #### Features computed from IG
            corr, p_value = spss.spearmanr(brain_resilience_df[col1], features_df[col2], nan_policy='omit')
            # validix = behv_df[col1].notna() & features_df[col2].notna()
            # corr, p_value = spss.pearsonr(behv_df[col1][validix], features_df[col2][validix])

            correlation_results[f'{col1}-{col2}'] = {'correlation': corr, 'p-value': p_value}
            if p_value < 0.05:
                print(col1)
                print(col2)
                print(corr)
                print(p_value)

    # print(correlation_results)
    print("\n".join(
        [f"Key: {k}, Correlation: {v['correlation']}, p-value: {v['p-value']}" for k, v in correlation_results.items()
         if "hippocampus" in k.lower()]))

    # Mutlivariate analyses: Perform linear regression
    print('*' * 10)
    print('Performing Multivariate Brain-Behavior Analyses')
    na_indices_df1 = features_df[features_df.isna().any(axis=1)].index
    na_indices_df2 = brain_resilience_df[brain_resilience_df.isna().any(axis=1)].index
    na_indices = na_indices_df1.union(na_indices_df2)
    features_df = features_df.drop(na_indices)
    brain_resilience_df = brain_resilience_df.drop(na_indices)

    # X_train, X_test, y_train, y_test = train_test_split(features_df, behv_df, test_size=0.3, random_state=111)
    X_train, X_test, y_train, y_test = features_df, features_df, brain_resilience_df, brain_resilience_df

    for column in brain_resilience_df.columns:
        model = LinearRegression().fit(X_train, y_train[column])
        # Compute model p using non-parametric
        # original_r2, original_p = spss.spearmanr(y_train[column], model.predict(X_train))
        # n_iterations = 1000
        # count_exceeds = 0
        # for _ in range(n_iterations):
        #     X_resampled, y_resampled = resample(X_train, y_train[column])
        #     model_resampled = LinearRegression().fit(X_resampled, y_resampled)
        #     r2_resampled,  p_resampled = spss.spearmanr(y_resampled, model_resampled.predict(X_resampled))
        #     if r2_resampled >= original_r2:
        #         count_exceeds += 1
        # p_value = count_exceeds / n_iterations

        # Compute p value of model coefficients
        # original_coef = model.coef_
        # n_iterations = 1000
        # count_exceeds = np.zeros(original_coef.shape)
        # for _ in range(n_iterations):
        #     X_resampled, y_resampled = resample(X_train, y_train[column])
        #     model_resampled = LinearRegression().fit(X_resampled, y_resampled)
        #     count_exceeds += (np.abs(model_resampled.coef_) >= np.abs(original_coef)).astype(int)
        # p_values = count_exceeds / n_iterations

        score = mean_squared_error(y_test[column], model.predict(X_test))
        corr, pvalue = spss.spearmanr(y_test[column], model.predict(X_test))
        print(f"MSE for {column}: {score}")
        print(f"Spearman r, p for {column}: {corr}, {pvalue}")
        # print(f"Coefficient p values for {column}: {p_values}")


def perform_brain_behavior_analyses(subjids_all, visitids_all, features_df, behavior_file_name):
    subjids_all = np.asarray(subjids_all)[labels_all < MAX_AGE]
    visitids_all = np.asarray(visitids_all)[labels_all < MAX_AGE]
    behv_datao = pd.read_csv(os.path.join(behavior_data_dir, behavior_file_name), skiprows=1)

    behv_datao_sel = pd.DataFrame()
    features_sel_ix = []
    for idix, id_ in enumerate(subjids_all):
        row = behv_datao[(behv_datao['ID'] == id_) & (behv_datao['VISIT'] == visitids_all[idix])]
        if not row.empty:
            behv_datao_sel = behv_datao_sel.append(row.iloc[0])
            features_sel_ix.append(idix)

    
    features_df = features_df.iloc[features_sel_ix].copy()
    features_df.drop(features_df.columns[0], axis=1, inplace=True)
    ###################### Below line gets rid of ID, we want to store ID for each predicted and observed behavior to find shared sample across various metrics
    ids = None
    if len(behv_datao_sel.index) != 0:
        ids = np.asarray(behv_datao_sel.iloc[:,0])
        ids = ids.astype(str)
    behv_df = behv_datao_sel.select_dtypes('float64')
    behv_df = behv_df.reset_index(drop=True)
    features_df = features_df.reset_index(drop=True)
    #print(features_df)
    #pdb.set_trace()
    # Univariate analyses: Compute Spearman correlation and p-value for each pair of columns
    # print('*'*30)
    # print('Performing Univariate Brain-Behavior Analyses')
    # correlation_results = {}
    # for col1 in behv_df.columns:
    #     for col2 in features_df.columns:
    #         corr, p_value = spss.spearmanr(behv_df[col1], features_df[col2], nan_policy='omit')
    #         #validix = behv_df[col1].notna() & features_df[col2].notna()
    #         #corr, p_value = spss.pearsonr(behv_df[col1][validix], features_df[col2][validix])
    #
    #
    #         if p_value < 0.05:
    #             correlation_results[f'{col1}-{col2}'] = {'correlation': corr, 'p-value': p_value}

    # correlation_results_df = pd.DataFrame(correlation_results).T
    # print(correlation_results_df)

    # print(correlation_results)
    # print("\n".join(
    #     [f"Key: {k}, Correlation: {v['correlation']}, p-value: {v['p-value']}" for k, v in correlation_results.items()
    #      if "hippocampus" in k.lower()]))

    # Mutlivariate analyses: Perform linear regression
    print('*' * 30)
    # print('Performing Multivariate Brain-Behavior Analyses')
    na_indices_df1 = features_df[features_df.isna().any(axis=1)].index
    na_indices_df2 = behv_df[behv_df.isna().any(axis=1)].index
    na_indices = na_indices_df1.union(na_indices_df2)
    features_df = features_df.drop(na_indices)
    behv_df = behv_df.drop(na_indices)
    if ids is not None:
        new_ids = np.delete(ids,na_indices)
        print(new_ids)
    if behv_df.empty:
        return

    roinames = ['a4ll', 'a39rv', 'a8vl', 'a7c', 'chipp']
    colnames = [col[0] for col in features_df.columns]
    colnames = [x.split(',',1)[0] for x in colnames]
    features_df.columns = colnames
    features_df.columns = features_df.columns.str.lower()
    features_df = features_df.loc[:,roinames]
    sc = StandardScaler()
    features_df = sc.fit_transform(features_df)

    pca = PCA(n_components=25,random_state=0)
    features_df = pca.fit_transform(features_df)
    # X_train, X_test, y_train, y_test = train_test_split(features_df, behv_df, test_size=0.3, random_state=111)
    X_train, X_test, y_train, y_test = features_df, features_df, behv_df, behv_df

    multregression_results = {}
    for column in behv_df.columns:
        model = LinearRegression().fit(X_train, y_train[column])
        # Compute model p using non-parametric
        # original_r2, original_p = spss.spearmanr(y_train[column], model.predict(X_train))
        # n_iterations = 1000
        # count_exceeds = 0
        # for _ in range(n_iterations):
        #     X_resampled, y_resampled = resample(X_train, y_train[column])
        #     model_resampled = LinearRegression().fit(X_resampled, y_resampled)
        #     r2_resampled,  p_resampled = spss.spearmanr(y_resampled, model_resampled.predict(X_resampled))
        #     if r2_resampled >= original_r2:
        #         count_exceeds += 1
        # p_value = count_exceeds / n_iterations

        # Compute p value of model coefficients
        # original_coef = model.coef_
        # n_iterations = 1000
        # count_exceeds = np.zeros(original_coef.shape)
        # for _ in range(n_iterations):
        #     X_resampled, y_resampled = resample(X_train, y_train[column])
        #     model_resampled = LinearRegression().fit(X_resampled, y_resampled)
        #     count_exceeds += (np.abs(model_resampled.coef_) >= np.abs(original_coef)).astype(int)
        # p_values = count_exceeds / n_iterations

        score = mean_squared_error(y_test[column], model.predict(X_test))
        corr, pvalue = spss.spearmanr(y_test[column], model.predict(X_test))
        #sig_participants = np.asarray(ids[pvalue < 0.05])
        # print(f"MSE for {column}: {score}")
        # print(f"Spearman r, p for {column}: {corr}, {pvalue}")
        # print(f"Coefficient p values for {column}: {p_values}")
        if pvalue < 0.05 and new_ids is not None:
            plot_corr = pd.DataFrame({'ID':new_ids,'Behavior':y_test[column],'Behavior_Features':model.predict(X_test)})
            plot_corr.to_csv(f'Features_HCP_Dev_model_{column}_wIDS.csv',index=False)
            multregression_results[f'features-{column}'] = {'correlation': corr, 'p-value': pvalue}

    multregression_results_df = pd.DataFrame(multregression_results).T
    print(multregression_results_df)


def get_data(path):
    fp = open(path, "rb")
    data_dict = pickle.load(fp)
    fp.close()
    return data_dict["X_train"], data_dict["X_test"], data_dict["Y_train"], data_dict["Y_test"]


def reshapeData(data):
    no_subjs, no_ts, no_channels = data.shape
    data_reshape = np.empty((no_subjs, no_channels, no_ts))
    for subj in np.arange(no_subjs):
        x_subj = data[subj, :, :]
        x_subj = np.transpose(x_subj)
        data_reshape[subj, :, :] = x_subj
    return data_reshape


if __name__ == '__main__':
    # Set random seeds for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed(CUDA_SEED)
    torch.manual_seed(CUDA_SEED)
    np.random.seed(NP_RANDOM_SEED)
    random.seed(PYTHON_RANDOM_SEED)

    # Set global variables
    global train_loader, val_loader, test_loader

    # I/O paths setup
    if os.name == 'nt':
        OAK = 'Z:/'
    else:
        OAK = '/oak/stanford/groups/menon/'
    #data_dir = '/scratch/users/ksupekar/nki_age_windowed/'
    data_dir = OAK + '/projects/mellache/2021_foundation_model/data/imaging/for_dnn/nki_age_cog_dev/'
    result_dir = OAK + '/projects/ksupekar/2024_scratch/mellache/results/'
    model_dir = OAK + '/projects/ksupekar/2024_scratch/mellache/results/models/stdnn_age_allsubjs/'
    behavior_data_dir = OAK + '/projects/mellache/2021_foundation_model/scripts/FLUX/assessment_data/'

    datao = np.load(
        OAK + '/deriveddata/public/nkirs/restfmri/timeseries/group_level/brainnetome/normz/nkirs_site-nkirs_run-rest_645_brainnetome_mean_regMov-6param_wmcsf_dt1_bpf008-09_normz_246ROIs.pklz',
        allow_pickle=True)

    indices_to_remove = [ind for ind, i in enumerate(datao["data"].values) if
                         (len(i) != 900 or np.sum(np.isnan(i)) > 0)]
    data_pklz = np.asarray(
        [np.asarray(i) for ind, i in enumerate(datao["data"].values) if ind not in indices_to_remove])
    datao_pklz = datao.drop(indices_to_remove)

    demo_datao = pd.read_csv(os.path.join(behavior_data_dir, DEMO_FILE), skiprows=1)

    data_all = []
    labels_all = []
    train_loader = []
    val_loader = []
    scaler_all = []
    subjids_all = []
    visitids_all = []
    meanfd_all = []
    for fold_id in np.arange(K_FOLDS):
        path = data_dir + 'fold_' + str(fold_id) + '.bin'
        path_dev = OAK + '/projects/mellache/2021_foundation_model/data/imaging/for_dnn/hcp_dev_age_five_fold/' + 'fold_' + str(fold_id) + '.bin'
        X_train, X_valid, Y_train, Y_valid = get_data(path)
        X_train_dev, X_valid_dev, Y_train_dev, Y_valid_dev = get_data(path_dev)

        print(Y_valid.shape)

        if not os.path.exists(os.path.join(model_dir, 'subjectids.txt')):
            for row_id in np.arange(Y_valid.shape[0]):
                datao_pklz_sel = datao_pklz[datao_pklz['age'] == Y_valid[row_id]]
                for sel_id in np.arange(datao_pklz_sel.shape[0]):
                    if np.sum(datao_pklz_sel.iloc[sel_id].data - X_valid[row_id]) == 0:
                        if not (datao_pklz_sel.iloc[sel_id].subject_id in subjids_all):
                            subjids_all.append(datao_pklz_sel.iloc[sel_id].subject_id)
                            meanfd_all.append(datao_pklz_sel.iloc[sel_id].mean_fd)
                            visitid = demo_datao[(demo_datao['ID'] == datao_pklz_sel.iloc[sel_id].subject_id) & (
                                        demo_datao['AGE'] == Y_valid[row_id])]['VISIT'].values
                            visitids_all.append(visitid)
                            break

        X_train = reshapeData(X_train)
        X_valid = reshapeData(X_valid)
        if fold_id == 0:
            data_all = X_valid
            labels_all = Y_valid
        else:
            data_all = np.concatenate((data_all, X_valid))
            labels_all = np.concatenate((labels_all, Y_valid))

        scaler = StandardScaler()
        scaler.fit(Y_train_dev.reshape(-1,1))
 
        Y_train = scaler.transform(Y_train.reshape(-1, 1))
        Y_valid = scaler.transform(Y_valid.reshape(-1, 1))
        #print(Y_valid)
        #pdb.set_trace()
        train_loader.append(
            DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train.astype('float64'))),
                       batch_size=BATCH_SIZE, shuffle=True))
        val_loader.append(
            DataLoader(TensorDataset(torch.FloatTensor(X_valid), torch.FloatTensor(Y_valid.astype('float64'))),
                       batch_size=X_valid.shape[0], shuffle=False))

        scaler_all.append(scaler)

    if not os.path.exists(os.path.join(model_dir, 'subjectids.txt')):
        np.savetxt(os.path.join(model_dir, 'subjectids.txt'), np.asarray(subjids_all).astype('str'), fmt='%s')
        np.savetxt(os.path.join(model_dir, 'visitids.txt'),
                   np.array([arr[0] if arr.size > 0 else None for arr in visitids_all]).astype('str'), fmt='%s')

    subjids_all = np.loadtxt(os.path.join(model_dir, 'subjectids.txt'), dtype=str)
    visitids_all = np.loadtxt(os.path.join(model_dir, 'visitids.txt'), dtype=str)
    
    #print(labels_all)
    #pdb.set_trace()
    # test the fine-tuned model
    #predicted_age, observed_age = test_all_folds(labels_all, scaler_all, subjids_all, meanfd_all)
    #df = pd.DataFrame({'Actual':observed_age,'Predicted':predicted_age})
    #df.to_csv('nki_age_prediction_ksupekar_model_results_v2.csv')
    #pdb.set_trace()
    #features_df = get_and_analyze_features(data_all, labels_all.flatten().astype('float64'),subjids_all)
    
    #if not os.path.exists(os.path.join(model_dir, OUTPUT_FILE_PREFIX + '.csv')):
    #    predicted_age, observed_age = test_all_folds(labels_all, scaler_all, subjids_all, meanfd_all)
    #else:
    #    output_df = pd.read_csv(os.path.join(model_dir, OUTPUT_FILE_PREFIX + '.csv'))
    #    predicted_age, observed_age = output_df['age_predicted'].values, output_df['age_observed'].values

    #get features from the fine-tuned model
    if not os.path.exists(os.path.join(model_dir, FEATURE_FILE_PREFIX + '.csv')):
        features_df = get_and_analyze_features(data_all, labels_all.flatten().astype('float64'),subjids_all)
    else:
        features_df = pd.read_csv(os.path.join(model_dir, FEATURE_FILE_PREFIX + '.csv'))
    
    # brain behavior
    # perform_brain_behavior_analyses(subjids_all, visitids_all, features_df, BEHAVIOR_FILE)
    # for behavior_file_name in sorted(os.listdir(behavior_data_dir)):
    #     if behavior_file_name.endswith('.csv'):
    #         print('*'*50)
    #         print(behavior_file_name)
    #         print('*'*50)
    #         perform_brain_behavior_analyses(subjids_all, visitids_all, features_df, behavior_file_name)

    # brain resilience analyses
    # perform_brain_resilience_analyses(subjids_all, visitids_all, features_df, observed_age, predicted_age)

    # resilience behavior analysis
    #perform_resilience_behavior_analyses(subjids_all, visitids_all, observed_age, predicted_age, '8100_Metabolites_set_1_20191009.csv')
    #sorted_file_list = sorted(os.listdir)
    for behavior_file_name in sorted(os.listdir(behavior_data_dir)):
        if behavior_file_name.endswith('.csv'):
            print('*'*50)
            print(behavior_file_name)
            print('*'*50)
            #perform_resilience_behavior_analyses(subjids_all, visitids_all, observed_age, predicted_age, behavior_file_name)
            perform_brain_behavior_analyses(subjids_all, visitids_all, features_df, behavior_file_name)
    # plt.figure
    # sns.regplot(x=predicted_age, y=observed_age, scatter_kws={"color": "red"}, line_kws={"color": "black"})
    # #sns.regplot(x=predicted_age[observed_age < MAX_AGE], y=observed_age[observed_age < MAX_AGE], scatter_kws={"color": "red"}, line_kws={"color": "black"})
    # plt.savefig('stdnn_nki_age_predicted_vs_age_observed_plot_v6.jpg')
    # plt.xlabel('Predicted Age')
    # plt.ylabel('Observed Age')

    # plt.figure(5)
    # #sns.regplot(x=observed_age, y=predicted_age - observed_age, scatter_kws={"color": "red"}, line_kws={"color": "black"})
    # my_color = (202/255, 30/255, 30/255)
    # sns.regplot(x=observed_age[observed_age < MAX_AGE], y=predicted_age[observed_age < MAX_AGE] - observed_age[observed_age < MAX_AGE], scatter_kws={"color": my_color}, line_kws={"color": "black"})
    # plt.ylabel('Resilience to aging')
    # plt.xlabel('Age')
    # plt.savefig('stdnn_nki_age_observed_vs_br_plot_v1.jpg')

    breakpoint()
