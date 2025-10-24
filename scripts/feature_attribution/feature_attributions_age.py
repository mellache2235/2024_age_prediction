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
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from torch.utils.data import DataLoader, TensorDataset
import pdb
sys.path.append('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/')
from utility_functions import *
from itertools import chain

warnings.filterwarnings("ignore", category=FutureWarning)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(246, 32, kernel_size=5, stride=1, bias=False),
            nn.BatchNorm1d(32),
            nn.PReLU(32),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(32,32, kernel_size=7, stride=1, bias=False),
            nn.BatchNorm1d(32),
            nn.PReLU(32),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.drop_out = nn.Dropout(p=0.6)
        self.regressor = nn.Linear(32, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.drop_out(out)
        out = self.layer2(out)
        out = self.drop_out(out)
        out = out.mean(axis=2)
        out = self.regressor(out)
        return out

def get_and_analyze_features(data_all, labels_all,subjects):
    attr_data = np.zeros((5, data_all.shape[0], data_all.shape[1], data_all.shape[2]))
    cuda_available = USE_CUDA and torch.cuda.is_available()
    for fold_id in np.arange(5):
        fname_model = model_dir + f'Convnet_cmihbn_TD_age_regression_fold_{fold_id}.pt'
        print(fname_model)
        model = ConvNet()
        if cuda_available:
            model.cuda()
        # model.load_state_dict(torch.load(fname_model, map_location=torch.device('cpu')))
        model.load_state_dict(torch.load(fname_model))
        #data_all = torch.from_numpy(data_all).type(torch.FloatTensor)
        #data_all = data_all.cuda()
        #predicted_ages = model(data_all)
        #predicted_ages = predicted_ages.cpu().detach().numpy()
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
    data_tensor = torch.from_numpy(data_all).type(torch.FloatTensor)
    data_tensor = data_tensor.cuda()
    predicted_ages = model(data_tensor)
    predicted_ages = predicted_ages.cpu().detach().numpy()
    attr_data_tmp = np.median(attr_data, 0)
    attr_data_median = np.squeeze(attr_data_tmp)
    # attr_data_median = np.squeeze(attr_data[BEST_FOLD_ID]) # best fold
    #ageix = np.squeeze(np.argwhere(labels_all < MAX_AGE))  # only analyze certain age attributes
    #attr_data_median = attr_data_median[ageix, :, :]
    #ix = labels_all < MAX_AGE
    #subjects_dev = subjects[ix]
    # find most discriminating ROIs between the two groups
    attr_data_tsavg = np.median(attr_data_median, axis=2) * FEATURE_SCALE_FACTOR #### Median along time dimension
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

    #roi_nifti.to_filename(FEATURE_FILE_PREFIX + '.nii.gz')
    #display = plotting.plot_stat_map(roi_nifti,display_mode='ortho',cut_coords=(0,0,0),colorbar=True,cmap='inferno',output_file='nki_brain_region_importance_IG_v3.png') 
    #for ax in display.axes.values():
    #    ax.axis('off')
    #np.save(os.path.join(model_dir, FEATURE_FILE_PREFIX + '.npy'), attr_data_tsavg[:, features_idcs])
    if PERCENTILE == 0:
        features_df = pd.DataFrame(attr_data_tsavg, columns=roi_labels)
    else:
        features_df = pd.DataFrame(np.squeeze(attr_data_tsavg[:, features_idcs]),
                                   columns=roi_labels_sorted[attr_data_percentileix[::-1]])

    features_df['subject_id'] = np.asarray(subjects)
    #features_df.to_csv('brain_features_IG_convnet_regressor_cmihbn_age_top_regions_wIDS.csv')

    return features_df, predicted_ages

if __name__ == '__main__':
    CUDA_SEED = 2344
    NP_RANDOM_SEED = 652
    PYTHON_RANDOM_SEED = 819
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed(CUDA_SEED)
    torch.manual_seed(CUDA_SEED)
    np.random.seed(NP_RANDOM_SEED)
    random.seed(PYTHON_RANDOM_SEED) 
    PERCENTILE = 0
    #PERCENTILE = 0
    FEATURE_SCALE_FACTOR = 10000
    USE_CUDA = True
    ATLAS_NIFTI = '/oak/stanford/groups/menon/projects/sryali/2019_DNN/scripts/features/BN_Atlas_246_2mm.nii'
    FEATURE_FILE_PREFIX = 'bn_features_kfold_all_group_cmihbn_age_TD_percentile_' + str(PERCENTILE)
    OUTPUT_FILE_PREFIX = 'adhd200_TD_age_prediction_output'
 
    data_dir = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/data/cmihbn_age_TD/'

    behavior_dir = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/prepare_data/cmihbn/behavior/'

    model_dir = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/models/cmihbn/'

    '''
    srs_scores = pd.read_csv(behavior_dir + 'Cmihbn-SocialResponsiveness_ParentReport_DATA_2024-11-04_1336.csv')
    srs_scores = srs_scores.dropna(subset=['srs_srs_awr_t'])
    srs_scores = srs_scores.reset_index()
    srs_scores['basic_demos_eid'] = srs_scores['basic_demos_eid'].astype('str')
    srs_scores = srs_scores.rename(columns={'basic_demos_eid':'subject_id'})
    '''

    adhd_scores = pd.read_csv(behavior_dir + 'Cmihbn-ConnersADHDRatingSca_DATA_2024-11-01_1206_8-21years.csv')
    adhd_scores = adhd_scores.dropna(subset=['c3sr_c3sr_hy_t'])
    adhd_scores = adhd_scores.reset_index()
    adhd_scores['basic_demos_eid'] = adhd_scores['basic_demos_eid'].astype('str')
    adhd_scores = adhd_scores.rename(columns={'basic_demos_eid':'subject_id'})
 
    ids_all = []
    data_all = []
    labels_all = []
    srs_all = []
    hys_all = []

    for fold in range(5):

        #path = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/data/cmihbn_age_TD/fold_%d.bin'%(fold)
        path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/adhd_cmihbn_age_ADHD_wIDS/fold_%d.bin'%(fold)

        X_train, X_valid, id_train, Y_train, Y_valid, id_test = load_finetune_dataset_w_ids(path)

        id_test = id_test.astype('str')
        ids_all.append(list(id_test))
       
        path_dev = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/hcp_dev_age_five_fold/fold_0.bin'

        X_train_dev, X_valid_dev, Y_train_dev, Y_valid_dev = load_finetune_dataset(path_dev)
 
        X_train = reshapeData(X_train)
        X_valid = reshapeData(X_valid)
       
        #scaler = StandardScaler()
        #scaler.fit(X_train) 
        '''
        mask = np.isin(srs_scores['subject_id'],id_test)
        
        srs_data_subset = srs_scores[mask]
        order = np.argsort([np.where(id_test == id_)[0][0] for id_ in srs_data_subset['subject_id']])
        srs_data_subset_order = srs_data_subset.iloc[order]
        scores = np.asarray(srs_data_subset_order['srs_srs_awr_t'])
        scores = scores.astype(float)
        srs_all.append(scores)
        '''
        mask = np.isin(adhd_scores['subject_id'],id_test)
        
        adhd_data_subset = adhd_scores[mask]
        order = np.argsort([np.where(id_test == id_)[0][0] for id_ in adhd_data_subset['subject_id']])
        adhd_data_subset_order = adhd_data_subset.iloc[order]
        scores = np.asarray(adhd_data_subset_order['c3sr_c3sr_hy_t'])
        scores = scores.astype(float)
        hys_all.append(scores)

        if fold == 0:
            data_all = X_valid
            labels_all = Y_valid
            X_dev_total = np.concatenate((X_train_dev,X_valid_dev))
            X_dev_total = reshapeData(X_dev_total)
            Y_dev_total = np.concatenate((Y_train_dev,Y_valid_dev))
            X_train_new, X_valid_new, Y_train_new, Y_valid_new = train_test_split(X_dev_total, Y_dev_total, test_size=0.1,random_state=42)
            sc = StandardScaler()
            sc.fit(Y_train_new.reshape(-1,1))
        else:
            data_all = np.concatenate((data_all,X_valid))
            labels_all = np.concatenate((labels_all,Y_valid))
         
    ids_all = list(chain(*ids_all))
    #srs_all = np.asarray(list(chain(*srs_all)))
    hys_all = np.asarray(list(chain(*hys_all)))

    features_df, predicted_ages = get_and_analyze_features(data_all, labels_all.flatten().astype('float64'),ids_all)
    #features_df = pd.read_csv('brain_features_IG_convnet_regressor_adhd200_age_top_regions_wIDS.csv')
    #print(features_df.shape)
    '''
    features_df = features_df[~pd.isna(srs_all)]
    srs_all = srs_all[~pd.isna(srs_all)]
 
    features_df['SRS'] = np.asarray(srs_all)
    '''
    actual_labels = labels_all.flatten().astype('float64')

    features_df = features_df[~pd.isna(hys_all)]
    hys_all = hys_all[~pd.isna(hys_all)]
    actual_subset = actual_labels[~pd.isna(hys_all)]
    predicted_subset = predicted_ages[~pd.isna(hys_all)].flatten()

    print(actual_subset.shape)
    print(predicted_subset.shape)
    print(hys_all.shape)
    BAG = predicted_subset - actual_subset
    frame = pd.DataFrame({'BAG':BAG,'HY':hys_all})
    print(frame)
    corr, pvalue = spss.spearmanr(frame['BAG'], frame['HY'])
    print(corr)
    print(pvalue)
     
    features_df['HY'] = np.asarray(hys_all)
    #print(features_df.shape)
    print(features_df) 

    #features_df.drop(features_df.columns[0], axis=1, inplace=True)
    features_df.drop(features_df.columns[246],axis=1, inplace=True)
    print(features_df)  
    #print(ids_all)
    #print(hys_all)
    X = features_df.iloc[:,0:246]
    Y = features_df.iloc[:,246]
    model = Lasso(alpha=0.01).fit(X,Y)
    #score = mean_squared_error(y_test[column], model.predict(X_test))
    corr, pvalue = spss.spearmanr(Y, model.predict(X))
    print(corr)
    print(pvalue)
    outputs = model.predict(X)
    labels = Y
    np.savez('predicted_cmihbn_ADHD_HY_all_regions_dev_model',output=outputs)
    np.savez('actual_cmihbn_ADHD_HY_all_regions_dev_model',output=outputs) 
    #print(len(ids_all))
    #print(len(hys_all))

    
