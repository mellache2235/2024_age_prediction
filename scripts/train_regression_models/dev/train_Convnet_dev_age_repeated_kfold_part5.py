import sys
sys.path.append("/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from utility_functions import *
import random
import pdb
import pickle
from scipy.interpolate import interp1d
import wandb
import statistics as st
import shutil
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, StratifiedKFold, StratifiedShuffleSplit
from scipy import signal
import warnings
import time
import os

warnings.filterwarnings("ignore", category=FutureWarning)
USE_CUDA = True
hyper_parameters = {}
hyper_parameters['briannectome'] = True


CUDA_SEED = 2484
NP_RANDOM_SEED = 631
PYTHON_RANDOM_SEED = 239
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
torch.use_deterministic_algorithms(True)
torch.cuda.manual_seed(CUDA_SEED)
torch.manual_seed(CUDA_SEED)
np.random.seed(NP_RANDOM_SEED)
random.seed(PYTHON_RANDOM_SEED)

result_dir = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/'

hyper_parameters = {}
hyper_parameters['num_epochs'] = 200
hyper_parameters['batch_size'] = 16
hyper_parameters['learning_rate'] = 0.0001
hyper_parameters['briannectome'] = True

USE_CUDA = True

temp = 0

####### Prepare Data
data_dir = "/oak/stanford/groups/menon/deriveddata/public/"
datao = np.load(data_dir + 'hcp_dev/restfmri/timeseries/group_level/brainnetome/normz/hcp_dev_run-rfMRI_REST1_AP_brainnetome_mean_regMov-6param_wmcsf_dt1_bpf008-09_normz_246ROIs.pklz',allow_pickle=True)
X = np.asarray([np.asarray(i) for i in datao["data"].values])
Y = np.asarray([i for ind,i in enumerate(datao['age'].values)])

def run_all_folds():
    accs=[]
    rskf = RepeatedKFold(n_splits=5,n_repeats=20,random_state=42)
    #rskf.get_n_splits(X,Y)
    count = 0
    #pdb.set_trace()
    for fold_number, (train_index,val_index) in enumerate(rskf.split(X,Y)):
            
            print('Fold:',fold_number)
            fname_model = result_dir + 'models/dev/repeated_kfold_part5/Convnet_regressor_dev_age_fold_%d.pt'%(fold_number)
            fname_figure = result_dir + 'figures/dev/repeated_kfold_part5/Convnet_regressor_dev_age_fold_%d'%(fold_number)
            
            X_train = X[train_index]
            Y_train = Y[train_index]
 
            X_valid = X[val_index]
            Y_valid = Y[val_index]
            
            X_train = reshapeData(X_train)
            X_valid = reshapeData(X_valid)

            Y_train, Y_valid = np.asarray(Y_train), np.asarray(Y_valid)

            scaler = StandardScaler()
            scaler.fit(Y_train.reshape(-1,1))

            Y_train = scaler.transform(Y_train.reshape(-1,1))
            Y_valid = scaler.transform(Y_valid.reshape(-1,1))

            Data = {}
            Data['train_features'] = X_train
            Data['train_labels'] =  Y_train
            Data['valid_features'] = X_valid
            Data['valid_labels'] = Y_valid
            Data['test_features'] = None
            Data['test_labels'] = None

            train_loader, valid_loader,test_loader = get_data_loaders_forRegression(Data,hyper_parameters)

            model, train_loss_list,  val_loss_list, predicted, actual = train_Regressor_wEmbedding(train_loader,valid_loader,test_loader,hyper_parameters,fname_model,USE_CUDA)
            print('Finished Fold:',fold_number)

if __name__ == "__main__":
    run_all_folds()

