import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
import pickle
from tqdm import tqdm
import pdb
from sklearn.preprocessing import LabelEncoder
from os import listdir
from os.path import isfile, join

def create_kfold_partitions(data, labels, ids,path, window_size, stride, n_splits=5):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=27)
    for index, (train, test) in enumerate(kfold.split(data, labels)):
        X_train, X_test = data[train], data[test]
        Y_train, Y_test = labels[train], labels[test]
        ids_train, ids_test = ids[train], ids[test]
        #id_train, id_test = ids[train], ids[test]
        print(X_train.shape, Y_train.shape)
        # Window the train set
        #if window_size and stride:
            #if window_size < X_train.shape[1]:
                #X_train, Y_train = prepare_data_sliding_window(X_train, Y_train, window_size, stride)
        print(X_train.shape, Y_train.shape,"\n")
        data_dict = dict()
        data_dict["X_train"] = X_train
        data_dict["X_test"] = X_test
        data_dict["id_train"] = ids_train
        data_dict["Y_train"] = Y_train
        data_dict["Y_test"] = Y_test
        data_dict["id_test"] = ids_test
        # Save the dictiona
        fp = open(path+"fold_"+str(index)+".bin", "wb")
        pickle.dump(data_dict, fp, protocol=4)                # Protocol 4 is required to pickle objects larger than 4GB
        fp.close()

data_dir = '/oak/stanford/groups/menon/projects/wdcai/2019_ADHD_NN/data/imaging/timeseries/CMIHBN/restfmri/timeseries/group_level/brainnetome/normz/'
files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
#print(files)

count = 0
### Look only at run1
for i in range(len(files)):
    if 'run1' in files[i]:
        count += 1
        if count == 1:
            data = np.load(data_dir + files[i],allow_pickle=True)
        else:
            data_new = np.load(data_dir + files[i],allow_pickle=True)
            data = pd.concat([data,data_new])
    else:
        continue

#print(data.shape)
data['label'] = data['label'].astype(str).astype(int)
#df = data[data['label'] != 99]
df = data[data['mean_fd']<0.5]
df = df.dropna()
df = df.reset_index()
df['subject_id'] = df['subject_id'].astype('str')

td_list = pd.read_csv('/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/prepare_data/cmihbn/Cmihbn-CustomDECMIHBNClinic_DATA_2024-10-21_1336.csv')
td_list['basic_demos_eid'] = td_list['basic_demos_eid'].astype('str')
td_list['diagnosis_clinicianconsensus_dx_01'] = td_list['diagnosis_clinicianconsensus_dx_01'].astype('str')
ix = td_list['diagnosis_clinicianconsensus_dx_01'] == 'No Diagnosis Given'
td_list = td_list[ix]

mask = np.isin(df['subject_id'],td_list['basic_demos_eid'])
df = df[mask]
df = df.reset_index()

timesteps = []
ts_data = df['data']
for i in range(ts_data.shape[0]):
    data_subj = ts_data[i]
    if len(data_subj) != 375:
        timesteps.append(i)
df = df.drop(timesteps)

'''
srs_scores = pd.read_csv('Cmihbn-SocialResponsiveness_ParentReport_DATA_2024-11-04_1336.csv')
srs_scores = srs_scores.dropna(subset=['srs_srs_awr_t'])
srs_scores = srs_scores.reset_index()
srs_scores['basic_demos_eid'] = srs_scores['basic_demos_eid'].astype('str')
srs_scores = srs_scores.rename(columns={'basic_demos_eid':'subject_id'})
'''
adhd_scores = pd.read_csv('Cmihbn-ConnersADHDRatingSca_DATA_2024-11-01_1206_8-21years.csv')
adhd_scores = adhd_scores.dropna(subset=['c3sr_c3sr_hy_t'])
adhd_scores = adhd_scores.reset_index()
adhd_scores['basic_demos_eid'] = adhd_scores['basic_demos_eid'].astype('str')
adhd_scores = adhd_scores.rename(columns={'basic_demos_eid':'subject_id'})

new_frame = pd.merge(df,adhd_scores,on='subject_id')

X = np.asarray([np.asarray(i) for i in new_frame["data"].values])
Y = np.asarray(new_frame["age"])
ids = np.asarray(new_frame['subject_id'])

# There are NaNs in the data. So drop those samples entirely
samples_to_keep = list()
num_nans = 0
for i in range(X.shape[0]):
    if not np.isnan(np.sum(X[i])):
        samples_to_keep.append(i)
    else:
        num_nans+=1
print("Number of Samples being dropped due to NaNs:", num_nans, "\t", (num_nans/X.shape[0])*100, "%")

X = X[samples_to_keep]
Y = Y[samples_to_keep]
ids = ids[samples_to_keep]

create_kfold_partitions(X,Y, ids,"/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/data/cmihbn_age_TD_noNAs_HY/" , 256,64,5)

