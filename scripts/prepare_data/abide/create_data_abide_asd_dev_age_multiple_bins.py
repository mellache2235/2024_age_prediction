import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
import pickle
from tqdm import tqdm
import pdb
from sklearn.preprocessing import LabelEncoder
import fnmatch
import itertools
from itertools import chain
from pathlib import Path

def adjust_timesteps_for_subjects(subjects_data, target_timesteps=180, padding_value=0):
    adjusted_subjects_data = []

    for subject in subjects_data:
        #print(len(subject))
        if len(subject) > target_timesteps:
            # Truncate to the first 180 timesteps
            adjusted_subject = subject[:target_timesteps]
        else:
            # Pad with the padding value to reach 180 timesteps
            pad_length = target_timesteps - len(subject)
            padding = np.full((pad_length, len(subject[0])), padding_value)
            adjusted_subject = np.vstack([subject, padding])

        adjusted_subjects_data.append(adjusted_subject)

    # Convert the list of arrays to a 3D numpy array
    return np.asarray([np.asarray(i) for i in adjusted_subjects_data])

def truncate_predefined_length(jagged_array, length):
    truncated_array = [sublist[:length] for sublist in jagged_array]
    return truncated_array

def create_kfold_partitions(data, labels,ids,path, window_size, stride, n_splits=5):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=27)
    for index, (train, test) in enumerate(kfold.split(data, labels)):
        X_train, X_test = data[train], data[test]
        Y_train, Y_test = labels[train], labels[test]
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
        data_dict["ids_train"] = id_train
        data_dict["Y_train"] = Y_train
        data_dict["Y_test"] = Y_test
        data_dict["ids_test"] = id_test
        # Save the dictiona
        fp = open(path+"fold_"+str(index)+"_wIDS.bin", "wb")
        pickle.dump(data_dict, fp, protocol=4)                # Protocol 4 is required to pickle objects larger than 4GB
        fp.close()


final_sites = ['NYU','SDSU','STANFORD','Stanford','TCD-1','UM','USM','Yale']
common_string = '246ROIs'
path_abide_asd = "/oak/stanford/groups/menon/deriveddata/public/abide/restfmri/timeseries/group_level/brainnetome/normz/"
all_files = os.listdir(path_abide_asd)
filtered_files = []
for file_name in all_files:
    if any(substring in file_name for substring in final_sites):
        filtered_files.append(file_name)
#print(filtered_files)

appended_data = []
for file_name in filtered_files:
    file_path = os.path.join(path_abide_asd, file_name)
    data = np.load(file_path,allow_pickle=True)
    data = data[~pd.isna(data)]
    appended_data.append(data)
appended_data = pd.concat(appended_data)
appended_data['label'] = appended_data['label'].astype(int)
appended_data['age'] = appended_data['age'].astype(float)
ix = appended_data['label'] == 1 
appended_data_td = appended_data[ix]
# you only care up to 21
appended_data_td = appended_data_td[appended_data_td['age'] <= 21].reset_index(drop=True)

# ------------------------------------------------------------------
# 1.  Define developmental-stage bins
# ------------------------------------------------------------------
dev_bins = {
    "child_5_8"           : (5,  8),
    "late_child_8_11"     : (8, 11),
    "early_ado_11_14"     : (11, 14),
    "midlate_ado_14_18"   : (14, 18),
    "emerging_adult_18_21": (18, 22)   # 22 so that 21.999 falls inside
}

ROOT_DIR = Path("/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/data")
# ------------------------------------------------------------------
# 2.  Loop over bins
# ------------------------------------------------------------------
for name, (lo, hi) in dev_bins.items():
    print(f"\n=== Processing bin: {name}  ({lo} – {hi}) ===")

    # subset by age
    age_mask  = (appended_data_td['age'] >= lo) & (appended_data_td['age'] < hi)
    subset    = appended_data_td.loc[age_mask].copy()

    if subset.empty:
        print("  [!] No subjects in this bin – skipping.")
        continue

    # --------------------------------------------------------------
    # 2a. Adjust timesteps & drop bad subjects (same logic as yours)
    # --------------------------------------------------------------
    new_X = adjust_timesteps_for_subjects(subset['data'])
    subset['updated_data'] = list(new_X)

    bad_idx = [
        i for i, arr in enumerate(subset['updated_data'])
        if (len(arr) != 180 or np.isnan(arr).sum() > 0)
    ]
    if bad_idx:
        subset = subset.drop(subset.index[bad_idx])

    if subset.empty:
        print("  [!] All subjects dropped after QC – skipping.")
        continue
    subset = subset.reset_index()
    # stack into numpy arrays
    X = np.stack(subset['updated_data'].values, axis=0)
    Y = subset['age'].values
    ids = subset['subjid']

    print(X.shape)
    print(Y.shape)
    print(ids)
    print(f"  Final N = {len(Y)}  |  X shape = {X.shape}")

    # --------------------------------------------------------------
    # 2b. 5-fold partition + save
    # --------------------------------------------------------------
    out_dir = ROOT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(out_dir)
    create_kfold_partitions(
        X,
        Y,
        ids,
        str(out_dir) + '/',   # convert Path → str for your helper
        256,
        64,
        5
    )


