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
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from torch.utils.data import DataLoader, TensorDataset
import pdb
from utility_functions import *
from itertools import chain
from pprint import pprint
from termcolor import colored
import networkx as nx
from functools import reduce
from matplotlib.colors import to_rgb, hsv_to_rgb, rgb_to_hsv
from matplotlib.patches import Patch
from matplotlib import colormaps

warnings.filterwarnings("ignore", category=FutureWarning)

fname_model_dir = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/results/dnn/models/pretraining/original/'
hyper_parameters = {}
hyper_parameters['fname_masked_model'] = fname_model_dir + 'final_FM_hcp_nki_resting_data_normz_MFM_embeddings_3_block_kernel_size_increase_relu__percentMask_25_epochs_500_lr_0.001_drop_out_rate_0.0_test_data_train_test_split_zero_v2.pt'

PERCENTILE = 80
FEATURE_FILE_PREFIX = 'bn_features_abcd_anyuse_y0_all_data_percentile_' + str(PERCENTILE)

def plot_difference(file1,file2):
    difference_map = image.math_img("img1 - img2", img1=file1, img2=file2)
    display = plotting.plot_stat_map(difference_map,display_mode='ortho',colorbar=True,cmap='bwr',output_file='abcd_bisbas_RR_gender_difference_site16_IG.png',draw_cross=False, annotate=False)

def get_and_analyze_features(data_all, labels_all,subjects,yeo):
    attr_data = np.zeros((data_all.shape[0], data_all.shape[1], data_all.shape[2]))
    cuda_available = USE_CUDA and torch.cuda.is_available()
    fname_model = model_dir + f'predict_any_use_y0_classifier.pt'
    print(fname_model)
    model = CovnetClassifier_brainnectome_wEmbedder(hyper_parameters['fname_masked_model'],USE_CUDA)
    if cuda_available:
        model.cuda()
    # model.load_state_dict(torch.load(fname_model, map_location=torch.device('cpu')))
    model.load_state_dict(torch.load(fname_model))

    ig_tensor_data = torch.from_numpy(data_all).type(torch.FloatTensor)
    ig_tensor_labels = torch.from_numpy(labels_all).type(torch.LongTensor)

    if cuda_available:
        ig_tensor_data = ig_tensor_data.cuda()
        ig_tensor_labels = ig_tensor_labels.cuda()

    ig = IntegratedGradients(model)
    ig_tensor_data.requires_grad_()

    for idx, i in enumerate(range(0, len(ig_tensor_data), 10)):
        if i < len(ig_tensor_data) - 10:
            attr, delta = ig.attribute(ig_tensor_data[i:i + 10, :, :], target=ig_tensor_labels[i:i+10],
                                       return_convergence_delta=True)
        else:
            attr, delta = ig.attribute(ig_tensor_data[i:len(ig_tensor_data), :, :], target=ig_tensor_labels[i:len(ig_tensor_labels)],
                                           return_convergence_delta=True)
        attr_data[i:i + 10, :, :] = attr[0].detach().cpu().numpy()
        del attr, delta
    class_attr = {}
    for k in range(2):
        idx = labels_all == k
        class_attr[k] = np.abs(np.median(attr_data[idx],axis=2))
    class_0_attr = class_attr[0]
    class_1_attr = class_attr[1]
    cohens_d = (np.mean(class_0_attr,axis=0) - np.mean(class_1_attr,axis=0)) / (np.sqrt((np.std(class_0_attr,axis=0) ** 2 + np.std(class_1_attr,axis=0) ** 2) / 2))
    effect_size = np.abs(cohens_d)
    #attr_data_tmp = np.median(attr_data, 0)
    #attr_data_median = np.squeeze(attr_data)
    # attr_data_median = np.squeeze(attr_data[BEST_FOLD_ID]) # best fold
    #ageix = np.squeeze(np.argwhere(labels_all < MAX_AGE))  # only analyze certain age attributes
    #attr_data_median = attr_data_median[ageix, :, :]
    #ix = labels_all < MAX_AGE
    #subjects_dev = subjects[ix]
    # find most discriminating ROIs between the two groups
    #attr_data_tsavg = np.median(attr_data_median, axis=2) #### Median along time dimension
    #attr_data_grpavg = np.mean(attr_data_tsavg, axis=0)   #### Mean across all subjects

    attr_data_sorted = np.sort(effect_size)  #### Sort ROIs (after we've taken mean of all subjects)
    attr_data_sortedix = np.argsort(effect_size)
    attr_data_percentileix = np.argwhere(
        np.abs(attr_data_sorted) >= np.percentile(np.abs(attr_data_sorted), PERCENTILE))  ### Percentile set to 95, this get top 5% features
    features_idcs = attr_data_sortedix[attr_data_percentileix]
    #features_idcs = attr_data_sortedix
    features = np.abs(effect_size)

    with open(
            '/oak/stanford/groups/menon/projects/cdla/2021_hcp_earlypsychosis/scripts/restfmri/classify/CNN1dPyTorch/brainnetome_roi_labels.txt') as f:
        roi_labels = f.readlines()
        roi_labels = [x.strip() for x in roi_labels]

    roi_labels_sorted = np.array(roi_labels)[attr_data_sortedix]
    print('Age prediction: {}% percentile Channel/ROI by descending order of importance'.format(PERCENTILE))
    print(*roi_labels_sorted[attr_data_percentileix[::-1]], sep='\n')
    yeo_sorted = yeo.iloc[attr_data_sortedix,:]
    yeo_sorted = yeo_sorted.iloc[attr_data_percentileix[::-1].flatten(),:]
    features_sorted = features[attr_data_sortedix]
    features_sorted = features_sorted[attr_data_percentileix[::-1].flatten()]

    atlas_volume = image.load_img(ATLAS_NIFTI)
    roi_nifti = image.math_img('img-img', img=atlas_volume)
    img_data = atlas_volume.get_fdata()

    for idx in features_idcs:
        roi_idx = np.where(img_data == idx + 1, features[idx], 0)
        roi_img = image.new_img_like(roi_nifti, roi_idx)
        roi_nifti = image.math_img('img1+img2', img1=roi_nifti, img2=roi_img)

    roi_nifti.to_filename(FEATURE_FILE_PREFIX + '.nii.gz')
    #display = plotting.plot_stat_map(roi_nifti,display_mode='ortho',colorbar=True,cmap=colormaps['Reds'],output_file='abcd_bisbas_RR_all_data_site16_top20_IG.png',draw_cross=False, annotate=False) 
    #for ax in display.axes.values():
    #    ax.axis('off')
    #np.save(os.path.join(model_dir, FEATURE_FILE_PREFIX + '.npy'), attr_data_tsavg[:, features_idcs])
    '''
    if PERCENTILE == 0:
        features_df = pd.DataFrame(attr_data_tsavg, columns=roi_labels)
    else:
        features_df = pd.DataFrame(np.squeeze(attr_data_tsavg[:, features_idcs]),
                                   columns=roi_labels_sorted[attr_data_percentileix[::-1]])
    '''
    #features_df['subject_id'] = np.asarray(subjects)
    #features_df.to_csv('brain_features_IG_convnet_regressor_abcd_bisbas_RR_all_data_top_regions.csv')

    return features_sorted, yeo_sorted

if __name__ == '__main__':
    '''
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
    '''
    #PERCENTILE = 95
    #PERCENTILE = 0
    FEATURE_SCALE_FACTOR = 10000
    USE_CUDA = True
    ATLAS_NIFTI = '/oak/stanford/groups/menon/projects/sryali/2019_DNN/scripts/features/BN_Atlas_246_2mm.nii'
    #FEATURE_FILE_PREFIX = 'bn_features_kfold_all_group_adhd200_TD_percentile_' + str(PERCENTILE)
    OUTPUT_FILE_PREFIX = 'abcd_bisbas_RR_site16_prediction_output'
    
    data_dir = '/oak/stanford/groups/menon/projects/mellache/2024_ABCD_NIDA/data/'

    model_dir = '/oak/stanford/groups/menon/projects/mellache/2024_ABCD_NIDA/results/models/'

    df = pd.read_pickle(data_dir + 'abcd_NIDA_w_timeseries_more_binary_labels.pkl')
    
    #ix = df['gender'] == 'female'
    #df = df[ix]

    df['subject_id'] = df['subject_id'].astype('string')
    #hy_data['Hyper/Impulsive'] = hy_data['Hyper/Impulsive'].astype(float)
 
    ids_all = []
    data_all = []
    labels_all = []
    hys_all = []

    ids_all.append(list(df['subject_id']))
    
    df = df.dropna(subset=['Any_Use_Y0'])
    df = df.reset_index(drop=True)

    lst = []
    ts_data = df['data']
    for i in range(ts_data.shape[0]):
        data_subj = ts_data[i]
        if len(data_subj) != 383:
            lst.append(i)

    df = df.drop(lst)
    df = df.reset_index(drop=True)

    indices_to_remove = [ind for ind,i in enumerate(df['data'].values) if np.sum(np.isnan(i)) > 0]

    df = df.drop(indices_to_remove)
    df = df.reset_index(drop=True)

    data = np.asarray([np.asarray(i) for i in df['data'].values]) 
    data = reshapeData(data)

    y = df['Any_Use_Y0'].values

    data_all = data
    labels_all = y
 
    ids_all = list(chain(*ids_all))

    yeo = pd.read_csv("/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/feature_attribution/csv_files/subregion_func_network_Yeo_updated_yz.csv")
    features, yeo_sorted = get_and_analyze_features(data_all, labels_all.flatten(),ids_all,yeo)
    #male_map = image.load_img('bn_features_abcd_bisbas_RR_site16_males_only_percentile_95.nii.gz')
    #female_map = image.load_img('bn_features_abcd_bisbas_RR_site16_females_only_percentile_95.nii.gz')
    #plot_difference(male_map,female_map)
    #plot_attr(f'/oak/stanford/groups/menon/projects/mellache/2024_ABCD_NIDA/results/figures/attr_IG_abcd_bisbas_RR.png', group_avg, name, xlabel = 'Features/ROIs', ylabel = 'IG') 
    frame = {"Yeo" : yeo_sorted["Yeo_17network"], "Scores" : features}
    frame = pd.DataFrame(frame)
    frame = frame.groupby(['Yeo']).mean()
    net_names = np.asarray(["Yeo17_0","VisPeri","VisCent","SomMotA", "SomMotB", "DorsAttnA", "DorsAttnB", "SalVentAttnA", "SalVentAttnB", "LimbicB", "LimbicA", "FPA", "FPB", "FPC", "DefaultA", "DefaultB", "DefaultC", "TempPar", "AmyHip", "Striatum", "Thalamus"])
    net_names_sub = net_names[list(frame.index)]
    frame['Name'] = net_names_sub
    frame.to_csv("yeao_attrib_collapsed_mean_abcd_any_use_y0_all_subjects_top20.csv",header=False,index=False)

    pdb.set_trace()
    #features_df = pd.read_csv('brain_features_IG_convnet_regressor_adhd200_age_top_regions_wIDS.csv')
    #print(features_df.shape)

    features_df = features_df[~pd.isna(hys_all)]
    hys_all = hys_all[~pd.isna(hys_all)]
 
    features_df['HY'] = np.asarray(hys_all)
    #print(features_df.shape)
    print(features_df) 
   
    features_df.drop(features_df.columns[0], axis=1, inplace=True)
    features_df.drop(features_df.columns[13],axis=1, inplace=True)
    print(features_df)  
    #print(ids_all)
    #print(hys_all)
    X = features_df.iloc[:,0:13]
    Y = features_df.iloc[:,13]
    model = LinearRegression().fit(X,Y)
    #score = mean_squared_error(y_test[column], model.predict(X_test))
    corr, pvalue = spss.spearmanr(Y, model.predict(X))
    print(corr)
    print(pvalue)
 
    #print(len(ids_all))
    #print(len(hys_all))

    
