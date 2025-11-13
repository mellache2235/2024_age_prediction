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
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from torch.utils.data import DataLoader, TensorDataset
import pdb
sys.path.append('/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/train_regression_models/')
from utility_functions import *
from itertools import chain
from scipy.interpolate import interp1d
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

def get_and_analyze_features(data_all, labels_all,scaler,ids):
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

    #data_tensor = torch.from_numpy(data_all).type(torch.FloatTensor)
    #data_tensor = data_tensor.cuda()
    #predicted_ages = model(data_tensor)
    #predicted_ages = predicted_ages.cpu().detach().numpy()
    #predicted_ages = scaler.inverse_transform(predicted_ages)
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

    features_df['subject_id'] = np.asarray(ids)
    #features_df.to_csv('brain_features_IG_convnet_regressor_adhd200_age_top_regions_wIDS.csv')

    return features_df


def test_model_getVals(x_valid,y_valid,scaler,hyper_parameters, fname_model):
    criterion = nn.MSELoss()
    input_tensor_valid = torch.from_numpy(x_valid).type(torch.FloatTensor)
    label_tensor_valid = torch.from_numpy(y_valid)
    dataset_valid = TensorDataset(input_tensor_valid, label_tensor_valid)
    valid_loader = DataLoader(dataset=dataset_valid, batch_size=x_valid.shape[0], shuffle=False)
    model = ConvNet()
    model.load_state_dict(torch.load(fname_model,map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        for images, labels in valid_loader:
            outputs = model(images)
            outputs = torch.squeeze(outputs)
            #outputs_corrected = (outputs - intercept) / slope
            labels = torch.squeeze(labels)
        return np.squeeze(scaler.inverse_transform(labels.cpu().numpy().reshape(-1,1))), np.squeeze(scaler.inverse_transform(outputs.cpu().numpy().reshape(-1,1)))


if __name__ == '__main__':
    

    hyper_parameters = {}
    hyper_parameters['num_epochs'] = 500
    hyper_parameters['batch_size'] = 32
    hyper_parameters['learning_rate'] = 0.00097119581997096
    hyper_parameters['briannectome'] = True

    font_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/clustering_analysis/arial.ttf'
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()

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
    #PERCENTILE = 80
    PERCENTILE = 0
    FEATURE_SCALE_FACTOR = 10000
    USE_CUDA = True
    ATLAS_NIFTI = '/oak/stanford/groups/menon/projects/sryali/2019_DNN/scripts/features/BN_Atlas_246_2mm.nii'
    FEATURE_FILE_PREFIX = 'bn_features_kfold_all_group_adhd200_TD_percentile_' + str(PERCENTILE)
    OUTPUT_FILE_PREFIX = 'cmihbn_ADHD_age_prediction_output'
 
    #hy_dir = '/oak/stanford/groups/menon/deriveddata/public/adhd200/restfmri/timeseries/group_level/brainnetome/normz/'

    #model_dir = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/results/dnn/models/finetuning_adhd200/regression/'
    model_dir = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/train_regression_models/dev/' 
    
    #hy_data = np.load(hy_dir + 'adhd200_run-rest_brainnetome_mean_regMov-6param_wmcsf_dt1_bpf008-09_normz_246ROIs_nn.pklz',allow_pickle=True)
    
    #hy_data['subject_id'] = hy_data['subject_id'].astype('str')
    #hy_data['Hyper/Impulsive'] = hy_data['Hyper/Impulsive'].astype(float)

    #hy_data['site'] = hy_data['site'].astype('str')

    #hy_data = hy_data.drop_duplicates(subset='subject_id', keep='first')
    #hy_data = hy_data.reset_index()

    #ids_all = []
    #data_all = []
    #labels_all = []
    #hys_all = []

    actual_ages = []
    predicted_ages = []
    
    #lin_params = np.load('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/nki_updated/lin_model_params_nki_single_model.npz')

    #coefs = lin_params['coef']
    #intercepts = lin_params['intercept']

    path = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/data/cmihbn_age_ADHD_wIDs/fold_0.bin'

    X_train, X_valid, id_train, Y_train, Y_valid, id_valid = load_finetune_dataset_wids(path)

    X_total = np.concatenate((X_train,X_valid))
    Y_total = np.concatenate((Y_train,Y_valid))
    id_total = np.concatenate((id_train,id_valid))

    #path_HY = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/data/cmihbn_HY_ADHD_weidong_cutoffs/fold_0.bin'
    #path_IN = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/data/cmihbn_IN_ADHD_weidong_cutoffs/fold_0.bin'
    
    #X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(path_HY)

    #Y_total = np.concatenate((Y_train,Y_valid))

    #X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(path_IN)

    #IN_total = np.concatenate((Y_train,Y_valid))

    path_dev = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/hcp_dev_age_five_fold/fold_0.bin'

    X_train_dev, X_valid_dev, Y_train_dev, Y_valid_dev = load_finetune_dataset(path_dev)
 
    Y_dev_total = np.concatenate((Y_train_dev,Y_valid_dev))

    hcp_dev_model = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/train_regression_models/dev/best_outer_fold_0_hcp_dev_model_2_27_24.pt'
       
    X_total = reshapeData(X_total)
 
    sc = StandardScaler()
    sc.fit(Y_train_dev.reshape(-1,1))
    Y_total = sc.transform(Y_total.reshape(-1,1))
    print(X_total)
    print(Y_total)    
    #actual, predicted = test_model_getVals(X_total,Y_total,sc,hyper_parameters,hcp_dev_model)
  
    #Offset = coefs[0] * actual + intercepts[0]
    #predicted_corrected = predicted - Offset

    #actual_ages = np.concatenate((actual_ages,actual))
    #predicted_ages = np.concatenate((predicted_ages,predicted_corrected))
        
     
    features_df = get_and_analyze_features(X_total,Y_total, sc, id_total)
    features_df.to_csv('cmihbn_adhd_no_cutoffs_features_all_sites_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS_single_model_predictions.csv')
    pdb.set_trace() 
    #features_df = pd.read_csv('cmihbn_adhd_weidong_cutoffs_features_all_sites_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS_single_model_predictions.csv')
   
    #features_df['subject_id'] = features_df['subject_id'].astype('str')
    

    #features_df['site'] = features_df['subject_id'].map(hy_data.set_index('subject_id')['site'])
    #features_df['HY'] = np.asarray(features_df['subject_id'].map(hy_data.set_index('subject_id')['Hyper/Impulsive']))
    features_df['HY'] = HY_total
    #pdb.set_trace()
    #features_df['IN'] = IN_total
    features_df['actual_age'] = actual_ages
    features_df['brain_age'] = predicted_ages
    
    print(features_df)
    ix = features_df['HY'] == -999.
    features_df = features_df[~ix]

    #ix = features_df['IN'] == -999.
    #features_df = features_df[~ix]
    
    print(features_df)

    ''' 
    BAG = np.squeeze(predicted_ages - actual_ages)
    frame = pd.DataFrame({'BAG':BAG,'IN':IN_total})
    print(frame)
    corr, pvalue = spss.spearmanr(frame['BAG'], frame['IN'])
    print(corr)
    print(pvalue)
    
    r_squared = corr ** 2
    p = pvalue
    if p < 0.001:
        p_text = r"$\mathit{P} < 0.001$"
    else:
        p_text = rf"$\mathit{{P}} = {p:.3f}$"
    fig,ax = plt.subplots(figsize=(5.5,5.5),dpi=300)    

    sns.regplot(x=frame['BAG'], y=frame['IN'], ci=None,
           scatter_kws={'color': 'navy', 'alpha' : 0.6,'s': 40,'edgecolor': 'w', 'linewidth':0.5},
           line_kws={'color': 'red', 'linewidth' : 2},ax=ax)

    ax.text(0.95, 0.05, f"$\mathit{{R}}^2$ = {r_squared:.3f}\n"f"{p_text}",
         transform=ax.transAxes,  # Use axis coordinates (0 to 1)
         horizontalalignment='right',  # Align text to the right at x=0.95
         verticalalignment='bottom',   # Align text to the bottom at y=0.05
         fontsize=12)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel("Brain Age Gap",fontsize=16,labelpad=10)
    ax.set_ylabel("Inattention",fontsize=16,labelpad=10)
    ax.set_title("Resilience-Behavior Analysis: CMI-HBN",fontsize=16,pad=10)
    
    plt.tight_layout(pad=1.2)
    pdf.FigureCanvas(fig).print_pdf('cmihbn_BAG_IN_pretty.ai')
    #plt.savefig('cmihbn_ADHD_weidong_cutoffs_BAG_HY_scatter_test_bias_correction_single_model_prediction.png',format='png')
    '''
    #pdb.set_trace()
    
    features_df = features_df.drop(['actual_age','brain_age','Unnamed: 0'],axis=1)
    #print(features_df) 
     
    X = features_df.iloc[:,0:246]
    Y = np.asarray(features_df.iloc[:,246])
    #print(X)
    #print(Y)
    sc = StandardScaler()
    X = sc.fit_transform(X)

    pca = PCA(n_components=70,random_state=0)
    X_pca = pca.fit_transform(X)
    model = LinearRegression()
    print(X_pca)
    model.fit(X_pca,Y)
    ''' 
    bins = 30
    fig,ax = plt.subplots()
    sns.histplot(data=Y,bins=bins, stat='density',alpha=0.8, linewidth=0,label='NYU', color='blue',ax=ax)
    sns.kdeplot(data=Y,color='blue',label='NYU',ax=ax)
    
    Y_peking = np.asarray(features_df_peking.iloc[:,246])
    sns.histplot(data=Y_peking,bins=bins, stat='density',alpha=0.8, linewidth=0,label='Peking', color='red',ax=ax)
    sns.kdeplot(data=Y_peking,color='red',label='Peking',ax=ax)
    plt.savefig('HY_dist_NYU_Peking.png')
    pdb.set_trace()
    '''
    #model = ElasticNet(alpha=0.001,random_state=42).fit(X,Y)
    #score = mean_squared_error(y_test[column], model.predict(X_test))
    corr, pvalue = spss.spearmanr(Y, model.predict(X_pca))
    outputs = model.predict(X_pca)
    #print(outputs)

    labels = Y 
    #np.savez('predicted_adhd200_HY_all_regions_dev_model',output=outputs)
    #np.savez('actual_adhd200_HY_all_regions_dev_model',output=labels)
    print(corr)
    print(pvalue)
    pdb.set_trace()
    r_squared = corr ** 2
    print(r_squared)
    #pdb.set_trace()
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
    ax.set_xlabel("Observed Hyperactivity",fontsize=16,labelpad=10)
    ax.set_ylabel("Predicted Hyperactivity",fontsize=16,labelpad=10)
    ax.set_title("Brain-Behavior Analysis: CMI-HBN",fontsize=16,pad=10)
    plt.tight_layout(pad=1.2)
    #plt.savefig('cmihbn_pca_regression_features_IN_weidong_cutoffs_scatter.png',format='png')
    pdf.FigureCanvas(fig).print_pdf('cmihbn_pca_regression_features_HY_weidong_cutoffs_scatter.ai')
    #print(len(ids_all))
    #print(len(hys_all))

    
