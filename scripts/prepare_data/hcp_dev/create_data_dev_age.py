import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
import pickle
from tqdm import tqdm
import pdb
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import pickle
from tqdm import tqdm
import pdb
from sklearn.preprocessing import LabelEncoder

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
    print(1)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=27)
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

data_dir = "/oak/stanford/groups/menon/deriveddata/public/"
datao = np.load(data_dir + 'hcp_dev/restfmri/timeseries/group_level/brainnetome/normz/hcp_dev_run-rfMRI_REST1_AP_brainnetome_mean_regMov-6param_wmcsf_dt1_bpf008-09_normz_246ROIs.pklz',allow_pickle=True)
print(datao)
#labels = np.asarray([np.asarray(i) for ind, i in enumerate(datao["label"].values) if ind not in indices_to_remove])
labels = np.asarray([0 if i == "male" else 1 for i in datao["gender"].values],dtype=np.int64)
data = np.asarray([np.asarray(i) for i in datao["data"].values])
age = np.asarray([i for ind,i in enumerate(datao['age'].values)])
print(data.shape)
print(age.shape)
print(min(age))
print(max(age))
pdb.set_trace()

create_kfold_partitions(data,age, "/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/hcp_dev_age_five_fold/" , 256,64,5)
