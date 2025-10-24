import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
import pickle
from tqdm import tqdm
import pdb
from sklearn.preprocessing import LabelEncoder

def create_kfold_partitions(data, labels,ids,path, window_size, stride, n_splits=5):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=27)
    for index, (train, test) in enumerate(kfold.split(data,labels)):
        X_train, X_test = data[train], data[test]
        Y_train, Y_test = labels[train], labels[test]
        #tr_train, tr_test = trs[train], trs[test]
        #tr_train, tr_test = trs[train], trs[test]
        id_train, id_test = ids[train], ids[test]
        print(X_train.shape, Y_train.shape)
        # Window the train set
        #if window_size and stride:
            #if window_size < X_train.shape[1]:
                #X_train, Y_train = prepare_data_sliding_window(X_train, Y_train, window_size, stride)
        print(X_train.shape, Y_train.shape,"\n")
        data_dict = dict()
        data_dict["X_train"] = X_train
        data_dict["X_test"] = X_test
        data_dict["id_train"] = id_train
        data_dict["Y_train"] = Y_train
        data_dict["Y_test"] = Y_test
        data_dict["id_test"] = id_test
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
    #data_dict["ids"] = ids
    print("Pre-training Data shape before saving:", X.shape)
    # Save the dictionary
    fp = open(path+"pretrain_data.bin", "wb")
    pickle.dump(data_dict, fp, protocol=4)                # Protocol 4 is required to pickle objects larger than 4GB
    fp.close()

data_dir = '/oak/stanford/groups/menon/deriveddata/public/adhd200/restfmri/timeseries/group_level/brainnetome/normz/'

data = np.load(data_dir + 'adhd200_run-rest_brainnetome_mean_regMov-6param_wmcsf_dt1_bpf008-09_normz_246ROIs_nn.pklz',allow_pickle=True)
#print(data)
#pdb.set_trace()
df = data[data['tr'] != 2.5]

### Remove certain labels
df = df[df['label'] != 'pending']
df = df[df['mean_fd'] < 0.5]
#pdb.set_trace()
labels = np.asarray(df['label'])
sites = np.asarray(df['site'])
genders = np.asarray(df['gender'])
genders = genders.astype(str)
ids = np.asarray(df['subject_id'])
ids = ids.astype(int)
mean_fds = np.asarray(df['mean_fd'])
ages = np.asarray(df['age'])
trs = np.asarray(df['tr'])
#print(ids)
#pdb.set_trace()

#df_control = df[df['label'] == 0]
#print(labels)
#print(min(df_control['age']))
#print(max(df_control['age']))
#pdb.set_trace()
#print(sites)
#pdb.set_trace()
ix = labels != 0
labels[ix] = 1
new_df = pd.DataFrame({'subject_id':ids,'data':list(df['data']),'mean_fd':mean_fds,'age':ages,'gender':genders,'site':sites,'label':labels,'trs':trs,'HY':np.asarray(df['Hyper/Impulsive'])})
# Take the last 174 timesteps
new_df['HY'] = new_df['HY'].astype(float)
print(np.asarray(new_df['HY']))

#pdb.set_trace()
ix = new_df['label'] == 1
new_df_ADHD = new_df[ix]
print(new_df_ADHD)
new_df_ADHD = new_df_ADHD.dropna()
#pdb.set_trace()
new_df_ADHD = new_df_ADHD.reset_index()
print(new_df_ADHD)
#pdb.set_trace()
X = np.asarray([np.asarray(i)[len(i)-174:] for i in new_df_ADHD["data"].values])
Y = np.asarray(new_df_ADHD['age'])
#print(Y)
#pdb.set_trace()
adhd_tr = new_df_ADHD["trs"].values
ids = new_df_ADHD['subject_id']
# There are NaNs in the data. So drop those samples entirely
samples_to_keep = list()
num_nans = 0
for i in range(X.shape[0]):
    if not np.isnan(np.sum(X[i])):
        samples_to_keep.append(i)
    else:
        num_nans+=1
print("Number of Samples being dropped due to NaNs:", num_nans, "\t", (num_nans/X.shape[0])*100, "%")
new_df_ADHD = new_df_ADHD.iloc[samples_to_keep,:]
new_df_ADHD = new_df_ADHD.reset_index()
X = np.asarray([np.asarray(i)[len(i)-174:] for i in new_df_ADHD["data"].values])
Y = np.asarray(new_df_ADHD['age'])
ids = new_df_ADHD['subject_id']
#print(samples_to_keep)
#pdb.set_trace()
##print(Y)
### Include tr because dataset contains samples with tr = 1.5, 1.9, 2.0, do separate interpolation
'''
print(new_df)
matched_list = pd.read_csv('final_list/adhd200_matched_motion_n416_with_age_motion_labels.csv',sep=',')
matched_list['subject_id'] = matched_list['subject_id'].astype(int)
print(matched_list)
pdb.set_trace()
X = np.asarray([np.asarray(i)[len(i)-174:] for i in new_df["data"].values])
mask = np.isin(ids,matched_list['subject_id'])
X_matched = X[mask]
Y_matched = Y[mask]
adhd_tr_matched = adhd_tr[mask]
'''
print(X.shape)
print(Y.shape)
print(ids.shape)
print(ids)
pdb.set_trace()
create_kfold_partitions(X,Y,ids,"/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/adhd200_regression_age_ADHD_wIDs/" , 256,64,5)
pdb.set_trace()

