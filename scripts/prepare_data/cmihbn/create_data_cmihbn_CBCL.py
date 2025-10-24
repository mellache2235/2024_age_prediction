import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
import pickle
from tqdm import tqdm
import pdb
from sklearn.preprocessing import LabelEncoder
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir
from os.path import isfile, join

def prepare_data_sliding_window(data, labels, window_size, step):
    """Generates a windowed version of input data

    Args:
        data (numpy matrix): Input data to be windowed
        labels (numpy array): Labels corresponding to each sample
        window_size (int): The size of the window
        step (int): The stride of the window

    Returns:
        (numpy matrix, numpy array): Windowed version of input data
    """
    Nsubjs, N, Nchannels = data.shape
    width = np.int(np.floor(window_size / 2.0))
    labels_window = list()
    data_window = list()
    for subj in tqdm(range(Nsubjs)):
        for k in range(width, N - width - 1, step):
            x = data[subj, k - width : k + width, :]
            x = np.expand_dims(x, axis=0)
            data_window.append(x)
            # window_data = np.concatenate((window_data, x))
            if labels is not None:
                labels_window.append(labels[subj])
    window_data = np.concatenate(data_window, axis=0)

    if labels is not None:
        return (window_data, np.asarray(labels_window, dtype=np.int64))
    else:
        return window_data

def create_kfold_partitions(data, labels, path, window_size, stride, n_splits=5):
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=27)
    for index, (train, test) in enumerate(kfold.split(data, labels)):
        X_train, X_test = data[train], data[test]
        Y_train, Y_test = labels[train], labels[test]
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
        #data_dict["id_train"] = id_train
        data_dict["Y_train"] = Y_train
        data_dict["Y_test"] = Y_test
        #data_dict["id_test"] = id_test
        # Save the dictiona
        fp = open(path+"fold_"+str(index)+".bin", "wb")
        pickle.dump(data_dict, fp, protocol=4)                # Protocol 4 is required to pickle objects larger than 4GB
        fp.close()


def create_pretrain_data(data, path, window_size, stride):
    print(2)
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
df = data[data['mean_fd']<0.5]
df = df.dropna()
df = df.reset_index()
df['subject_id'] = df['subject_id'].astype('str')
df = df.loc[df['label'] == 0,:]

cbcl_data = pd.read_csv('data-2025-02-24T18_41_39.830Z.csv')
cbcl_data['Identifiers'] = cbcl_data['Identifiers'].apply(lambda x : x[0:12])
cbcl_data = cbcl_data.rename(columns={'Identifiers':'subject_id'})


new_frame = pd.merge(df,cbcl_data,on='subject_id')

ocd_features = ['data','CBCL,CBCL_09', 'CBCL,CBCL_31', 'CBCL,CBCL_32', 'CBCL,CBCL_45', 'CBCL,CBCL_50', 'CBCL,CBCL_52', 'CBCL,CBCL_66', 'CBCL,CBCL_84', 'CBCL,CBCL_85','CBCL,CBCL_99','CBCL,CBCL_112']


OCS_2_items = ['CBCL,CBCL_09', 'CBCL,CBCL_66']
OCS_6_items = ['CBCL,CBCL_09', 'CBCL,CBCL_66', 'CBCL,CBCL_31', 'CBCL,CBCL_32', 'CBCL,CBCL_45', 'CBCL,CBCL_50']
OCS_8_items = ['CBCL,CBCL_09', 'CBCL,CBCL_66', 'CBCL,CBCL_31', 'CBCL,CBCL_32', 'CBCL,CBCL_45', 'CBCL,CBCL_50', 'CBCL,CBCL_52', 'CBCL,CBCL_84', 'CBCL,CBCL_85', 'CBCL,CBCL_112']

df_cbcl = new_frame[ocd_features].copy()

df_cbcl["OCS_2_score"] = df_cbcl[OCS_2_items].sum(axis=1)
df_cbcl["OCS_6_score"] = df_cbcl[OCS_6_items].sum(axis=1)
df_cbcl["OCS_8_score"] = df_cbcl[OCS_8_items].sum(axis=1)

df_cbcl["OCD_like"] = (
    (df_cbcl["OCS_2_score"] >= 2) |
    (df_cbcl["OCS_6_score"] >= 4) |
    (df_cbcl["OCS_8_score"] >= 6)
).astype(int)  # 1 = OCD-like, 0 = Healthy Control

print(df_cbcl["OCD_like"].value_counts())
timesteps = []
ts_data = df_cbcl['data']
for i in range(ts_data.shape[0]):
    data_subj = ts_data[i]
    if len(data_subj) != 375:
        timesteps.append(i)
df_cbcl = df_cbcl.drop(timesteps)
print(df_cbcl)
print(df_cbcl["OCD_like"].value_counts())

df_cbcl = df_cbcl.dropna()
df_cbcl = df_cbcl.reset_index()
print(df_cbcl)
print(df_cbcl["OCD_like"].value_counts())

#df_cbcl = df_cbcl[:,[f"Can't get his/her mind off certain thoughts; obsessions",f"Fears he/she might think or do something bad",f"Feels he/she has to be perfect",f"Nervous  highstrung  or tense",f"Too fearful or anxious",f"Feels too guilty",f"Repeats certain acts over and over; compulsions",f"Strange behavior",f"Strange Ideas",f"Worries"]]
'''
linkage_matrix = sch.linkage(df_cbcl,method='ward',metric='euclidean')
#print(linkage_matrix)

plt.figure(figsize=(12, 6))
sch.dendrogram(linkage_matrix,leaf_rotation=90, leaf_font_size=8)
plt.axhline(y=10, color='r', linestyle='--')  # Adjust the threshold based on the plot
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Subjects")
plt.ylabel("Distance")
plt.savefig('OCD_NKI_clustering_refined_HC.png')
pdb.set_trace()

n_clusters = 2
hc = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
new_frame["OCD_subtype"] = hc.fit_predict(df_cbcl)

#new_frame["OCD_subtype"] = new_frame["OCD_subtype"].map({0: "Compulsivity-dominant", 1: "Anxiety-dominant"})
'''

data = np.asarray([np.asarray(i) for ind,i in enumerate(df_cbcl['data'].values)])
labels = np.asarray([i for ind,i in enumerate(df_cbcl['OCD_like'])])

create_kfold_partitions(data,labels,'/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/data/cmihbn_ocd_vs_hc/',256,64)
