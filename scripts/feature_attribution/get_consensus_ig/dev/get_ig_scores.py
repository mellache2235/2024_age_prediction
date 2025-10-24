import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from captum.attr import IntegratedGradients, Saliency, DeepLift, DeepLiftShap, GradientShap, InputXGradient, GuidedBackprop, GuidedGradCam, Deconvolution, FeatureAblation, Occlusion, FeaturePermutation, ShapleyValueSampling, Lime, KernelShap, LRP
import sys
from pprint import pprint
from collections import OrderedDict
from torch import nn
import sklearn.metrics as met
from termcolor import colored
import networkx as nx
from functools import reduce
import argparse
import pdb
import re
import pickle

def reshapeData(data):
    no_subjs, no_ts, no_channels = data.shape
    # Reshape data to no_subjs, no_channels, no_ts
    data_reshape = np.empty((no_subjs, no_channels, no_ts))
    for subj in np.arange(no_subjs):
        x_subj = data[subj, :, :]
        x_subj = np.transpose(x_subj)
        data_reshape[subj, :, :] = x_subj
    return data_reshape

def load_finetune_dataset(path):
    """A function to load numpy dictionary for finetune data given path

    Args:
        path (string): A string path to load the numpy dictionary

    Returns:
        Numpy matrices (train_input, test_input, train_target, test_target)
    """
    fp = open(path, "rb")
    data_dict = pickle.load(fp)
    fp.close()
    return data_dict["X_train"], data_dict["X_test"], data_dict["Y_train"], data_dict["Y_test"]

class ConvNet(nn.Module):
    def __init__(self):
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

        self.drop_out = nn.Dropout(p=0.4561228015061742)
        self.regressor = nn.Linear(32, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.drop_out(out)
        out = self.layer2(out)
        out = self.drop_out(out)
        out = out.mean(axis=2)
        out = self.regressor(out)
        return out

def get_and_analyze_features(data_all, labels_all, fname_model, fold_number):
    ig_path = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/figures/dev/ig_files/'
    attr_data = np.zeros((data_all.shape[0], data_all.shape[1], data_all.shape[2]))
    cuda_available = USE_CUDA and torch.cuda.is_available()
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
    fname = f"dev_age_model_{fold_number}_ig" 
    np.savez(ig_path + fname,attr_data)

def load_sorted_files(directory, prefix, suffix):
    # List all files in the directory
    files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(suffix)]
   
    # Define a sorting function to extract the digit
    def extract_digit(filename):
        match = re.search(r'fold_(\d+)', filename)
        return int(match.group(1)) if match else float('inf')  # If no match, place at the end

    # Sort files based on the extracted digit
    sorted_files = sorted(files, key=extract_digit)
    
    return sorted_files

# Example usage
result_dir = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/'

USE_CUDA = True
directory = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/models/dev/repeated_kfold_part1/'
sorted_files = load_sorted_files(directory, prefix='classifier', suffix='.pt')

path = "/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/hcp_dev_age_five_fold/fold_0.bin"
X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(path)
X_train = reshapeData(X_train)
X_valid = reshapeData(X_valid)
X_total = np.concatenate((X_train,X_valid))
Y_total = np.concatenate((Y_train,Y_valid))

'''
for fold in range(250):
    print(f'Fold {fold}:')
    fname_model = directory + str(sorted_files[fold])
    get_and_analyze_features(X_total, Y_total, fname_model, fold)
'''

directory = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/models/dev/repeated_kfold_part1/'
sorted_files = load_sorted_files(directory, prefix='Convnet', suffix='.pt')

for fold in range(100):
    print(f'Fold {fold}:')
    fname_model = directory + str(sorted_files[fold])
    get_and_analyze_features(X_total, Y_total, fname_model, fold)


directory = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/models/dev/repeated_kfold_part2/'
sorted_files = load_sorted_files(directory, prefix='Convnet', suffix='.pt')

for fold in range(100):
    print(f'Fold {fold}:')
    fname_model = directory + str(sorted_files[fold])
    get_and_analyze_features(X_total, Y_total, fname_model, fold+100)

directory = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/models/dev/repeated_kfold_part3/'
sorted_files = load_sorted_files(directory, prefix='Convnet', suffix='.pt')

for fold in range(100):
    print(f'Fold {fold}:')
    fname_model = directory + str(sorted_files[fold])
    get_and_analyze_features(X_total, Y_total, fname_model, fold+200)

directory = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/models/dev/repeated_kfold_part4/'
sorted_files = load_sorted_files(directory, prefix='Convnet', suffix='.pt')

for fold in range(100):
    print(f'Fold {fold}:')
    fname_model = directory + str(sorted_files[fold])
    get_and_analyze_features(X_total, Y_total, fname_model, fold+300)

directory = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/models/dev/repeated_kfold_part5/'
sorted_files = load_sorted_files(directory, prefix='Convnet', suffix='.pt')

for fold in range(100):
    print(f'Fold {fold}:')
    fname_model = directory + str(sorted_files[fold])
    get_and_analyze_features(X_total, Y_total, fname_model, fold+400)
