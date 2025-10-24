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
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from torch.utils.data import DataLoader, TensorDataset
import pdb
sys.path.append('/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/train_regression_models/')
from utility_functions import *
from itertools import chain
from scipy.interpolate import interp1d
from os import listdir
from os.path import isfile, join
import matplotlib.backends.backend_pdf as pdf
from matplotlib import font_manager

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
    attr_data = np.zeros((data_all.shape[0], data_all.shape[1], data_all.shape[2]))
    cuda_available = USE_CUDA and torch.cuda.is_available()
    fname_model = model_dir + f'best_outer_fold_0_hcp_dev_model_2_27_24.pt'
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
        attr_data[i:i + 10, :, :] = attr[0].detach().cpu().numpy()
        del attr, delta

    #attr_data_tmp = np.median(attr_data, 0)
    #attr_data_median = np.squeeze(attr_data_tmp)
    # attr_data_median = np.squeeze(attr_data[BEST_FOLD_ID]) # best fold
    #ageix = np.squeeze(np.argwhere(labels_all < MAX_AGE))  # only analyze certain age attributes
    #attr_data_median = attr_data_median[ageix, :, :]
    #ix = labels_all < MAX_AGE
    #subjects_dev = subjects[ix]
    # find most discriminating ROIs between the two groups
    attr_data_tsavg = np.median(attr_data, axis=2) * FEATURE_SCALE_FACTOR #### Median along time dimension
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
    #features_df.to_csv('brain_features_IG_convnet_regressor_adhd200_age_top_regions_wIDS.csv')

    return features_df

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
    
    font_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/clustering_analysis/arial.ttf'
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()

    #PERCENTILE = 80
    PERCENTILE = 0
    FEATURE_SCALE_FACTOR = 10000
    USE_CUDA = True
    ATLAS_NIFTI = '/oak/stanford/groups/menon/projects/sryali/2019_DNN/scripts/features/BN_Atlas_246_2mm.nii'
    FEATURE_FILE_PREFIX = 'bn_features_kfold_all_group_adhd200_TD_percentile_' + str(PERCENTILE)
    OUTPUT_FILE_PREFIX = 'adhd200_TD_age_prediction_output'
 
    data_dir_SRS = "/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/prepare_data/stanford_autism/"
    SRS_file = pd.read_csv(data_dir_SRS + 'SRS_data_20230925.csv',skiprows=[0])
    SRS_file = SRS_file.drop_duplicates(subset=['record_id'],keep='last')
    SRS_file['record_id'] = SRS_file['record_id'].astype('str')
    ids_2 = np.asarray(SRS_file['record_id'])
    srs_score = np.asarray(SRS_file['srs_total_score_standard'])
    ids_2 = ids_2.astype('str')
    total_frame = pd.DataFrame({'subject_id':ids_2,'SRS':srs_score})
    #print(appended_data_td_dev.columns)
    #hy_dir = '/oak/stanford/groups/menon/deriveddata/public/adhd200/restfmri/timeseries/group_level/brainnetome/normz/'

    #model_dir = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/results/dnn/models/finetuning_adhd200/regression/'
    #model_dir = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/models/dev/final/' 
    model_dir = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/train_regression_models/dev/'
    #hy_data = np.load(hy_dir + 'adhd200_run-rest_brainnetome_mean_regMov-6param_wmcsf_dt1_bpf008-09_normz_246ROIs_nn.pklz',allow_pickle=True)
    
    #hy_data['subject_id'] = hy_data['subject_id'].astype('string')
    #hy_data['Hyper/Impulsive'] = hy_data['Hyper/Impulsive'].astype(float)
 
    ids_all = []
    data_all = []
    labels_all = []
    social_all = []

    for fold in range(5):

        path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/stanford_autism_age_wIDS/fold_%d.bin'%(fold)

        X_train, X_valid, id_train, Y_train, Y_valid, id_test = load_finetune_dataset_wids(path)

        id_test = id_test.astype('str')

        path_dev = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/hcp_dev_age_five_fold/fold_0.bin'

        X_train_dev, X_valid_dev, Y_train_dev, Y_valid_dev = load_finetune_dataset(path_dev)

        #ids_all.append(list(id_test))
       
        ### Interpolation leads to worse performance
        #interp_data_train = interp1d(np.linspace(0, 1,X_train.shape[1]), X_train, axis=1)
        #data_extend_train = interp_data_train(np.linspace(0, 1,math.floor(X_train.shape[1] * 2 / 0.65)))

        #interp_data_valid = interp1d(np.linspace(0, 1,X_valid.shape[1]), X_valid, axis=1)
        #data_extend_valid = interp_data_valid(np.linspace(0, 1,math.floor(X_valid.shape[1] * 2 / 0.65)))

        X_train = reshapeData(X_train)
        X_valid = reshapeData(X_valid)

        ''' 
        mask = np.isin(total_frame['subject_id'],id_test)
         
        social_data_subset = total_frame[mask]
        mask = np.isin(id_test,social_data_subset['subject_id'])
        
        id_test = id_test[mask]
        X_valid = X_valid[mask]
        Y_valid = Y_valid[mask]
        ids_all.append(list(id_test))
         
        order = np.argsort([np.where(id_test == id_)[0][0] for id_ in social_data_subset['subject_id']])
        social_data_subset_order = social_data_subset.iloc[order]
        scores = np.asarray(social_data_subset_order['SRS'])
        scores = scores.astype(float)
        social_all.append(scores)
        '''
        id_test = pd.Series(id_test)
        scores = np.asarray(id_test.map(total_frame.set_index('subject_id')['SRS']))
        social_all.append(scores)
  
        if fold == 0:
            #data_all = X_valid
            #labels_all = Y_valid
            sc = StandardScaler()
            sc.fit(Y_train_dev.reshape(-1,1))
            Y_valid = sc.transform(Y_valid.reshape(-1,1))
            data_all = X_valid
            labels_all = Y_valid
        else:
            Y_valid = sc.transform(Y_valid.reshape(-1,1))
            data_all = np.concatenate((data_all,X_valid))
            labels_all = np.concatenate((labels_all,Y_valid))
         
    ids_all = list(chain(*ids_all))
    social_all = np.asarray(list(chain(*social_all)))
    social_all = social_all[~pd.isna(social_all)]
    #features_df = get_and_analyze_features(data_all, labels_all.flatten().astype('float64'),ids_all)
    #features_df.to_csv('stanford_asd_features_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS.csv')
    features_df = pd.read_csv('stanford_asd_features_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS.csv')
    
    #features_df = features_df[~pd.isna(social_all)]
    #social_all = social_all[~pd.isna(social_all)]
 
    features_df['social'] = np.asarray(social_all)
    
    features_df.drop(features_df.columns[0], axis=1, inplace=True)
    features_df.drop(features_df.columns[246],axis=1, inplace=True)
    X = features_df.iloc[:,0:246]
    print(X)
    column_names = features_df.columns[:246]
    Y = features_df.iloc[:,246]
    sc = StandardScaler()
    X = sc.fit_transform(X) 
    
    pca = PCA(n_components=20,random_state=0)
    X_pca = pca.fit_transform(X)
    model = LinearRegression()
    model.fit(X_pca,Y)
    pca.fit(X)
    loadings = pca.components_
    region_names = np.asarray(column_names)
    region_names = region_names.astype('str')

    loading_df = pd.DataFrame(pca.components_,columns=region_names,index=[f'PC{i+1}' for i in range(pca.n_components_)])
    #plt.figure(figsize=(15, 6))
    # Select regions based on maximum absolute loading across the first 5 components
    #top_regions_idx = np.argsort(np.abs(loading_df).max(axis=0))[::-1][:20]  # Top 20 regions
    #loading_df_subset = loading_df.iloc[:, top_regions_idx]
    top_regions_idx = np.argsort(np.abs(loading_df.iloc[:5, :]).max(axis=0))[::-1][:20]
    loading_df_subset = loading_df.iloc[:5, top_regions_idx]
    plt.figure(figsize=(12, 8))
    sns.heatmap(loading_df_subset, cmap='RdBu_r', center=0, annot=True, fmt=".2f", cbar_kws={'shrink': 0.5})
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=12)
    plt.title('Top 20 Brain Regions PCA (PC 1-5) Loadings', fontsize=14)
    plt.tight_layout()

    plt.savefig('stanford_asd_pca_loadings_srs_total.png')
    #model = ElasticNet(alpha=0.01,random_state=0).fit(X,Y)
    #model = RandomForestRegressor(max_depth=2, random_state=0)
    #model.fit(X,Y)
    corr, pvalue = spss.spearmanr(Y, model.predict(X_pca))
    print(corr)
    print(pvalue)
    outputs = model.predict(X_pca)
    labels = Y

    r_squared = corr ** 2
    print(r_squared)
    p = pvalue
    if p < 0.001:
        p_text = r"$\mathit{P} < 0.001$"
    else:
        p_text = rf"$\mathit{{P}} = {p:.3f}$"
    fig,ax = plt.subplots(figsize=(5.5,5.5),dpi=300)
    sns.set_style("white")

    sns.regplot(x=labels, y=outputs, ci=None,
           scatter_kws={'color': 'navy', 'alpha' : 0.6,'s': 40,'edgecolor': 'w', 'linewidth':0.5},
           line_kws={'color': 'red', 'linewidth' : 2},ax=ax)
    ax.text(0.95, 0.05, f"$\mathit{{R}}^2$ = {r_squared:.3f}\n"f"{p_text}",
         transform=ax.transAxes,  # Use axis coordinates (0 to 1)
         horizontalalignment='right',  # Align text to the right at x=0.95
         verticalalignment='bottom',   # Align text to the bottom at y=0.05
         fontsize=12)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel("Observed SRS Total",fontsize=16,labelpad=10)
    ax.set_ylabel("Predicted SRS Total",fontsize=16,labelpad=10)
    ax.set_title("Stanford ASD",fontsize=16,pad=10)
    plt.tight_layout(pad=1.2)
    #plt.savefig('stanford_asd_pca_regression_features_ados_total_scatter.png',format='png')
    pdf.FigureCanvas(fig).print_pdf('stanford_asd_pca_regression_features_ados_total_scatter.ai') 
