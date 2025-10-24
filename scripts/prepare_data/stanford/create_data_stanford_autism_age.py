import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
import pickle
import pdb

def create_kfold_partitions(data, labels, ids,path, window_size, stride, n_splits=5):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=27)
    for index, (train, test) in enumerate(kfold.split(data, labels)):
        X_train, X_test = data[train], data[test]
        Y_train, Y_test = labels[train], labels[test]
        id_train, id_test = ids[train], ids[test]
        #X_train2, X_test2 = data2[train], data2[test]
        #Y_train2, Y_test2 = labels2[train], labels2[test]
        print(X_train.shape, Y_train.shape)
        # Window the train set
        #if window_size and stride:
            #if window_size < X_train.shape[1]:
                #X_train, Y_train = prepare_data_sliding_window(X_train, Y_train, window_size, stride)
        print(X_train.shape, Y_train.shape,"\n")
        data_dict = dict()
        data_dict["X_train"] = X_train
        data_dict["X_test"] = X_test
        #data_dict["X_train2"] = X_train2
        #data_dict["X_test2"] = X_test2
        data_dict["Y_train"] = Y_train
        data_dict["Y_test"] = Y_test
        data_dict["id_train"] = id_train
        data_dict["id_test"] = id_test
        #data_dict["Y_train2"] = Y_train2
        #data_dict["Y_test2"] = Y_test2
        # Save the dictiona
        fp = open(path+"fold_"+str(index)+".bin", "wb")
        pickle.dump(data_dict, fp, protocol=4)                # Protocol 4 is required to pickle objects larger than 4GB
        fp.close()


def create_pretrain_data(data, path, window_size, stride):
    X = data
    #if data.shape[1] > window_size:
        #X = prepare_data_sliding_window(data, None, window_size, stride)
    #else:
        #X = np.concatenate((data, np.zeros((data.shape[0], window_size-data.shape[1], data.shape[2]))), axis=1)
    data_dict = dict()
    data_dict["X"] = X
    data_dict["Y"] = X
    print("Pre-training Data shape before saving:", X.shape)
    # Save the dictionary
    fp = open(path+"pretrain_data.bin", "wb")
    pickle.dump(data_dict, fp, protocol=4)                # Protocol 4 is required to pickle objects larger than 4GB
    fp.close()


data_dir = '/oak/stanford/groups/menon/projects/sryali/2019_DNN/scripts/daelsaid/output/group/'
data = np.load(data_dir + 'stanford_brainnnetome_6_wmcsf.npz',allow_pickle=True)
#print(data.files)
#pdb.set_trace()
#data_stanford = np.load('/oak/stanford/groups/menon/deriveddata/public/stanford/restfmri/timeseries/group_level/brainnetome/normz/stanford_run-resting_state_brainnetome_mean_regMov-6param_wmcsf_dt1_bpf008-09_normz_246ROIs.pklz',allow_pickle=True)
#data_stanford_id_visit = data_stanford.loc[:,['subject_id','visit']]
#data_stanford_id_visit['subject_id'] = data_stanford_id_visit['subject_id'].astype(int)
ts = data['data']
age = data['ages']
label = data['labels']
ids = data['subjids']
ix = label == 2
ts = ts[ix]
age = age[ix]
ids = ids[ix]
ix = ~pd.isna(age)
ts = ts[ix]
age = age[ix]
ids = ids[ix]
print(ts.shape)
print(age.shape)
print(ids.shape)
create_kfold_partitions(ts,age,ids,"/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/stanford_autism_age_TD_wIDS/" , 256,64,5)
pdb.set_trace()


df = pd.DataFrame({'subject_id':data['subjids'],'data':list(data['data']),'mean_fd':data['mean_fds'],'gender':data['genders'],'age':data['ages'],'site':data['sites'],'label':data['labels']})
df_temp = pd.DataFrame({'subject_id':data['subjids'],'mean_fd':data['mean_fds'],'gender':data['genders'],'age':data['ages'],'site':data['sites'],'label':data['labels']})
##### Drop subjects who have NA for age
df = df.dropna()
df_temp = df_temp.dropna()

subj_list = subj_list.dropna()

df['subject_id'] = df['subject_id'].astype(int)
df_temp['subject_id'] = df_temp['subject_id'].astype(int)
df_temp['gender'] = df_temp['gender'].astype(int)

print(df_temp)
df_temp['label'] = np.abs(df_temp['label'] - 2)
print(df_temp)
df_temp['gender'] = np.asarray(["male" if i == 1 else "female" for ind, i in enumerate(df_temp["gender"].values)],dtype=np.str_)
print(df_temp)
#pdb.set_trace()

X = np.asarray([np.asarray(i) for i in df["data"].values])
labels = np.asarray([np.asarray(i) for i in df["label"].values])

#### Because 2 means HC and 1 means ASD, subtract 2 and take absolute value, so 0 means HC and 1 means ASD
labels = np.abs(labels - 2)

mask = np.isin(df['subject_id'],subj_list['subject_id'])

X_matched = X[mask]
Y_matched = labels[mask]
print(X_matched.shape)

create_kfold_partitions(X_matched,Y_matched, "/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/stanford_autism_final_yuan_list/" , 256,64,5)

pdb.set_trace()

###### If you want to save the time-series and labels to npz
matched_ts = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/prepare_data/stanford_autism/matched_stanford_autism_ts.npz'
matched_labels = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/prepare_data/stanford_autism/matched_stanford_autism_labels.npz'
np.savez(matched_ts,data=X_matched)
np.savez(matched_labels,label=Y_matched)
