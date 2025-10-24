#:x
#n[1]:

import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from utility_functions import *
import random
from scipy.interpolate import interp1d
import pdb
import pandas as pd

USE_CUDA = True
# ### Set the hyperparameters
hyper_parameters = {}
hyper_parameters['num_epochs'] = 500
hyper_parameters['batch_size'] = 16
hyper_parameters['learning_rate'] = 0.0001
hyper_parameters['briannectome'] = True



data_dir = '/oak/stanford/groups/menon/deriveddata/public/adhd200/restfmri/timeseries/group_level/brainnetome/normz/'

data = np.load(data_dir + 'adhd200_run-rest_brainnetome_mean_regMov-6param_wmcsf_dt1_bpf008-09_normz_246ROIs_nn.pklz',allow_pickle=True)
#print(data.keys())
#pdb.set_trace()
#print(data['site'])
#pdb.set_trace()
df = data[data['tr'] != 2.5]
# Remove certain labels
#df = data[data['label'] != 'pending']
df = df[df['label'] != 'pending']
df = df[df['mean_fd']<0.5]

labels = df['label'].values
labels = labels.astype('int')
labels[labels != 0] = 1
df['label'] = labels
df['label'] = df['label'].astype('int')

# Take the last 174 timesteps
X = np.asarray([np.asarray(i)[len(i)-174:] for i in df["data"].values])
Y = df["label"].values

print(X.shape)
fc_data, labels = get_features_labels(X,Y)
print(fc_data.shape)
pdb.set_trace()
# ### Load numpy file for the data and the embeddings model
fname_model = result_dir + 'models/finetuning_adhd200/original/classifier_from_variableintervalmaskEmbeddings_hcp_nki_normz_train_test_split_zero_25_' \
                           'adhd200_fold_0_adjust_tr.pt'
fname_figure = result_dir + 'figures/finetuning_adhd200/original/classifier_from_variableintervalmaskEmbeddings_hcp_nki_normz_adhd200_fold_0_h_adjust_tr'

#table_figure = table_dir + 'classifier_from_variableintervalmaskEmbeddings_hcp_nki_normz_adhd200_fold_0_table'

# Reshape Training Data
X_train, X_valid, tr_train, Y_train, Y_valid, tr_test = load_finetune_dataset_wtrs(path_data)
X_train_tr_adjust = X_train[tr_train == 1.5]
Y_train_tr_adjust = Y_train[tr_train == 1.5]
X_valid_tr_adjust = X_valid[tr_test == 1.5]
Y_valid_tr_adjust = Y_valid[tr_test == 1.5]
X_train_tr_orig = X_train[tr_train == 2]
Y_train_tr_orig = Y_train[tr_train == 2]
X_valid_tr_orig = X_valid[tr_test == 2]
Y_valid_tr_orig = Y_valid[tr_test == 2]
X_train_tr_mid = X_train[tr_train == 1.9]
Y_train_tr_mid = Y_train[tr_train == 1.9]
X_valid_tr_mid = X_valid[tr_test == 1.9]
Y_valid_tr_mid = Y_valid[tr_test == 1.9]
#print(X_train)
#print(Y_train)
interp_data_train = interp1d(np.linspace(0, 1,X_train_tr_orig.shape[1]), X_train_tr_orig, axis=1)
data_extend_train_orig = interp_data_train(np.linspace(0, 1,math.floor(X_train_tr_orig.shape[1] * 2 / 0.72)))
interp_data_valid = interp1d(np.linspace(0, 1,X_valid_tr_orig.shape[1]), X_valid_tr_orig, axis=1)
data_extend_valid_orig = interp_data_valid(np.linspace(0, 1,math.floor(X_valid_tr_orig.shape[1] * 2 / 0.72)))
interp_data_train = interp1d(np.linspace(0, 1,X_train_tr_adjust.shape[1]), X_train_tr_adjust, axis=1)
data_extend_train_adjust = interp_data_train(np.linspace(0, 1,math.floor(X_train_tr_adjust.shape[1] * 1.5 / 0.72)))
interp_data_valid = interp1d(np.linspace(0, 1,X_valid_tr_adjust.shape[1]), X_valid_tr_adjust, axis=1)
data_extend_valid_adjust = interp_data_valid(np.linspace(0, 1,math.floor(X_valid_tr_adjust.shape[1] * 1.5 / 0.72)))
interp_data_train = interp1d(np.linspace(0, 1,X_train_tr_mid.shape[1]), X_train_tr_mid, axis=1)
data_extend_train_mid = interp_data_train(np.linspace(0, 1,math.floor(X_train_tr_mid.shape[1] * 1.9 / 0.72)))
interp_data_valid = interp1d(np.linspace(0, 1,X_valid_tr_mid.shape[1]), X_valid_tr_mid, axis=1)
data_extend_valid_mid = interp_data_valid(np.linspace(0, 1,math.floor(X_valid_tr_mid.shape[1] * 1.9 / 0.72)))
[Ns,Nt,Nr] = data_extend_train_adjust.shape
data_extend_train_orig = data_extend_train_orig[:,0:Nt,:]
data_extend_train_mid = data_extend_train_mid[:,0:Nt,:]
[Ns,Nt,Nr] = data_extend_valid_adjust.shape
data_extend_valid_orig = data_extend_valid_orig[:,0:Nt,:]
data_extend_valid_mid = data_extend_valid_mid[:,0:Nt,:]
data_extend_train = np.concatenate((data_extend_train_orig,data_extend_train_adjust,data_extend_train_mid))
Y_train_new = np.concatenate((Y_train_tr_orig,Y_train_tr_adjust,Y_train_tr_mid))
data_extend_valid = np.concatenate((data_extend_valid_orig,data_extend_valid_adjust,data_extend_valid_mid))
Y_valid_new = np.concatenate((Y_valid_tr_orig,Y_valid_tr_adjust,Y_valid_tr_mid))
X_train = reshapeData(data_extend_train)
X_valid = reshapeData(data_extend_valid)
#keep = ~np.isnan(X_train).any(axis=(-1, -2))
#X_train = X_train[keep]
#keep2 = ~np.isnan(X_valid).any(axis=(-1, -2))
#X_valid = X_valid[keep2]
#Y_train = Y_train[keep]
#Y_valid = Y_valid[keep2]

#X_train_embed = get_embeddings(X_train)
#X_valid_embed = get_embeddings(X_valid)

# ### Prepare data for Training the NN model
Data = {}
Data['train_features'] = X_train
Data['train_labels'] =  Y_train_new
Data['valid_features'] = X_valid
Data['valid_labels'] = Y_valid_new
Data['test_features'] = None
Data['test_labels'] = None


# print(np.unique(Data['train_labels'],return_counts=True))
train_loader, valid_loader,test_loader = get_data_loaders_forClassifiers(Data,hyper_parameters)
# ### Train the model
model, train_loss_list, train_acc_list, val_loss_list, val_acc_list, test_accuracy,valid_f1_score = train_classifier_wEmbedding(train_loader,valid_loader,test_loader,hyper_parameters,fname_model,USE_CUDA)

valid_accuracy, valid_f1_score = test_model(X_valid,Y_valid_new,hyper_parameters,fname_model)
print('Test Accuracy of the model: {} %'.format((valid_accuracy)))
print('Test F1 score of the model: {} %'.format(100*valid_f1_score))
#print(id_table)
#pdb.set_trace()
np.savez(fname_figure, train_loss_list = train_loss_list, val_loss_list = val_loss_list,
         train_acc_list=train_acc_list,val_acc_list=val_acc_list, test_accuracy=test_accuracy,valid_accuracy=valid_accuracy,valid_f1_score=valid_f1_score )
#np.savez(table_figure, df=id_table, ids=id_table.index)
