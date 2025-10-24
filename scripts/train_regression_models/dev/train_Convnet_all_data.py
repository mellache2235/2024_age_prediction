import os
import sys
sys.path.append("/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/train_regression_models/")
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
import statistics as st
from scipy import signal
import itertools
from itertools import chain

USE_CUDA = True
# var_name = 'CardSort_Unadj'
#var_name = 'PMAT24_A_CR'
# var_name = 'CardSort_AgeAdj'
# ### Set the hyperparameters
hyper_parameters = {}
hyper_parameters['num_epochs'] = 200
hyper_parameters['batch_size'] = 16
hyper_parameters['learning_rate'] = 0.0009087016915288116
hyper_parameters['briannectome'] = True

if not USE_CUDA:
    data_dir = "/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/embarc_treatment_SER_w8/fold_0.bin"
    result_dir = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/results/dnn/'
else:
    data_dir = "/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/embarc_treatment_SER_w8/fold_0.bin"
    result_dir = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/results/dnn/'



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

class ConvNet(nn.Module):
    def __init__(self,drop_out_rate=0.40581892575490663):
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

        self.drop_out = nn.Dropout(p=drop_out_rate)
        self.regressor = nn.Linear(32, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.drop_out(out)
        out = self.layer2(out)
        out = self.drop_out(out)
        out = out.mean(axis=2)
        out = self.regressor(out)
        return out

path = "/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/hcp_dev_age_five_fold/fold_0.bin"

fname_model = 'hcp_dev_age_model_ba_correction_v2.pt'

fp = open(path, "rb")
data_dict = pickle.load(fp)
fp.close()
X_train, X_valid, Y_train, Y_valid = data_dict["X_train"], data_dict["X_test"], data_dict["Y_train"], data_dict["Y_test"]

X_total = np.concatenate((X_train,X_valid))
Y_total = np.concatenate((Y_train,Y_valid))
            
X_total = reshapeData(X_total)

#scaler = StandardScaler()
#Y_total = scaler.fit_transform(Y_total.reshape(-1,1))

X_train_new, X_valid_new, Y_train_new, Y_valid_new = train_test_split(X_total, Y_total, test_size=0.2,random_state=42)
 
scaler = StandardScaler()
scaler.fit(Y_train_new.reshape(-1,1))
Y_train_new = scaler.fit_transform(Y_train_new.reshape(-1,1))
Y_valid_new = scaler.fit_transform(Y_valid_new.reshape(-1,1))
         
Data = {}
Data['train_features'] = X_train_new
Data['train_labels'] =  Y_train_new
Data['valid_features'] = X_valid_new
Data['valid_labels'] = Y_valid_new
Data['test_features'] = None
Data['test_labels'] = None

train_loader, valid_loader,test_loader = get_data_loaders_forRegression(Data,hyper_parameters)

model,train_loss_list,valid_loss_list,predicted,actual,predicted_train,actual_train = train_Regressor_wEmbedding(train_loader,valid_loader,test_loader,hyper_parameters,fname_model,USE_CUDA)

fig,ax = plt.subplots()
ax.plot(train_loss_list,label="Training Loss")
ax.plot(valid_loss_list,label="Validation Loss")
ax.legend()
ax.spines[['right', 'top']].set_visible(False)
ax.set_xlabel("Step")
ax.set_ylabel("rMSE")
plt.savefig('ConvNet_age_prediction_train_valid_loss_curves.png')



pdb.set_trace()

predicted = scaler.inverse_transform(predicted)
actual = scaler.inverse_transform(actual)
r,p = scipy.stats.pearsonr(np.squeeze(predicted),np.squeeze(actual))
print(r)
print(p)
pdb.set_trace()
predicted_train = scaler.inverse_transform(predicted_train)
actual_train = scaler.inverse_transform(actual_train)
predicted = np.concatenate((predicted_train,predicted))
actual = np.concatenate((actual_train,actual))
print(predicted)
print(actual)
pdb.set_trace()
np.savez('predicted_hcp_dev_ages_model_v2',predicted=predicted)
np.savez('actual_hcp_dev_ages_model_v2',actual=actual)
