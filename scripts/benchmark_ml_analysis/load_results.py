import pandas as pd
import numpy as np
import math
import random
import statistics as st
import pickle

dt_original_sample = np.load('/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/results/dnn/figures/finetuning_embarc/ml_models/DT_embarc_hamd_original_sample_manuscript_SI_fig_results_ml_methods.npz',allow_pickle=True)
### DT
'''
dt_acc = st.mean(dt_original_sample['valid_accs'])
dt_f1 = st.mean(dt_original_sample['valid_f1s'])
dt_precision = st.mean(dt_original_sample['valid_precisions'])
dt_recall = st.mean(dt_original_sample['valid_recalls'])
print(dt_recall)
print(dt_precision)
print(dt_f1)
print(dt_acc)
'''
### KNN
'''
knn_original_sample = np.load('/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/results/dnn/figures/finetuning_embarc/ml_models/KNN_embarc_hamd_original_sample_manuscript_SI_fig_results_ml_methods.npz',allow_pickle=True)
knn_acc = st.mean(knn_original_sample['valid_accs'])
knn_f1 = st.mean(knn_original_sample['valid_f1s'])
knn_precision = st.mean(knn_original_sample['valid_precisions'])
knn_recall = st.mean(knn_original_sample['valid_recalls'])
print(knn_recall)
print(knn_precision)
print(knn_f1)
print(knn_acc)
'''

### LASSO
'''
lasso_original_sample = np.load('/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/results/dnn/figures/finetuning_embarc/ml_models/LASSO_embarc_hamd_original_sample_manuscript_SI_fig_results_ml_methods.npz',allow_pickle=True)
lasso_acc = st.mean(lasso_original_sample['valid_accs'])
lasso_f1 = st.mean(lasso_original_sample['valid_f1s'])
lasso_precision = st.mean(lasso_original_sample['valid_precisions'])
lasso_recall = st.mean(lasso_original_sample['valid_recalls'])
print(lasso_recall)
print(lasso_precision)
print(lasso_f1)
print(lasso_acc)
'''

### LinSVM
'''
linSVM_original_sample = np.load('/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/results/dnn/figures/finetuning_embarc/ml_models/linSVM_embarc_hamd_original_sample_manuscript_SI_fig_results_ml_methods.npz',allow_pickle=True)
linSVM_acc = st.mean(linSVM_original_sample['valid_accs'])
linSVM_f1 = st.mean(linSVM_original_sample['valid_f1s'])
linSVM_precision = st.mean(linSVM_original_sample['valid_precisions'])
linSVM_recall = st.mean(linSVM_original_sample['valid_recalls'])
print(linSVM_recall)
print(linSVM_precision)
print(linSVM_f1)
print(linSVM_acc)
'''

### LR
'''
lr_original_sample = np.load('/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/results/dnn/figures/finetuning_embarc/ml_models/LR_embarc_hamd_original_sample_manuscript_SI_fig_results_ml_methods.npz',allow_pickle=True)
lr_acc = st.mean(lr_original_sample['valid_accs'])
lr_f1 = st.mean(lr_original_sample['valid_f1s'])
lr_precision = st.mean(lr_original_sample['valid_precisions'])
lr_recall = st.mean(lr_original_sample['valid_recalls'])
print(lr_recall)
print(lr_precision)
print(lr_f1)
print(lr_acc)
'''

### RC
'''
rc_original_sample = np.load('/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/results/dnn/figures/finetuning_embarc/ml_models/RC_embarc_hamd_original_sample_manuscript_SI_fig_results_ml_methods.npz',allow_pickle=True)
rc_acc = st.mean(rc_original_sample['valid_accs'])
rc_f1 = st.mean(rc_original_sample['valid_f1s'])
rc_precision = st.mean(rc_original_sample['valid_precisions'])
rc_recall = st.mean(rc_original_sample['valid_recalls'])
print(rc_recall)
print(rc_precision)
print(rc_f1)
print(rc_acc)
'''

### RF
rf_original_sample = np.load('/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/results/dnn/figures/finetuning_embarc/ml_models/RF_embarc_hamd_original_sample_manuscript_SI_fig_results_ml_methods.npz',allow_pickle=True)
rf_acc = st.mean(rf_original_sample['valid_accs'])
rf_f1 = st.mean(rf_original_sample['valid_f1s'])
rf_precision = st.mean(rf_original_sample['valid_precisions'])
rf_recall = st.mean(rf_original_sample['valid_recalls'])
print(rf_recall)
print(rf_precision)
print(rf_f1)
print(rf_acc)
