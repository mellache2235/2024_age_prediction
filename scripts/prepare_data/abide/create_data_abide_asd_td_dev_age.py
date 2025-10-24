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

def create_kfold_partitions(data, labels,path, window_size, stride, n_splits=5):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=27)
    for index, (train, test) in enumerate(kfold.split(data, labels)):
        X_train, X_test = data[train], data[test]
        Y_train, Y_test = labels[train], labels[test]
        #tr_train, tr_test = trs[train], trs[test]
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
        #data_dict["tr_train"] = tr_train
        data_dict["Y_train"] = Y_train
        data_dict["Y_test"] = Y_test
        #data_dict["tr_test"] = tr_test
        # Save the dictiona
        fp = open(path+"fold_"+str(index)+".bin", "wb")
        pickle.dump(data_dict, fp, protocol=4)                # Protocol 4 is required to pickle objects larger than 4GB
        fp.close()


final_sites = ['NYU','SDSU','STANFORD','Stanford','TCD-1','UM','USM','Yale']
common_string = 'acompcor'
path_abide_asd = "/oak/stanford/groups/menon/deriveddata/public/abide/restfmri/timeseries/group_level/brainnetome/normz/"
all_files = os.listdir(path_abide_asd)
filtered_files = []
for file_name in all_files:
    if common_string in file_name and any(substring in file_name for substring in final_sites):
        filtered_files.append(file_name)

appended_data = []
for file_name in filtered_files:
    file_path = os.path.join(path_abide_asd, file_name)
    data = np.load(file_path,allow_pickle=True)
    data = data[~pd.isna(data)]
    appended_data.append(data)
appended_data = pd.concat(appended_data)
ix = appended_data['label'] == 'td'
appended_data_td = appended_data[ix]
ix = appended_data_td['age'] <= 21
appended_data_td_dev = appended_data_td[ix]
appended_data_td_dev = appended_data_td_dev.reset_index()

new_X = adjust_timesteps_for_subjects(appended_data_td_dev['data'])
appended_data_td_dev['updated_data'] = list(new_X)
indices_to_remove = [ind for ind, i in enumerate(appended_data_td_dev['updated_data']) if (len(i)!= 180 or np.sum(np.isnan(i)) > 0)]
appended_data_td_dev = appended_data_td_dev.drop(indices_to_remove)
new_X2 = np.asarray([np.asarray(i) for ind,i in enumerate(appended_data_td_dev['updated_data'].values)])
Y = np.asarray(appended_data_td_dev['age'])
'''
df = pd.DataFrame({'subject_id':appended_data['subject_id'],'data':list(new_X2),'mean_fd':appended_data['mean_fd'],'age':appended_data['age'],'gender':appended_data['gender'],'site':appended_data['site'],'label':appended_data['label']})
print(df)
X = new_X2
appended_data['label'] = appended_data['label'].astype(str)
Y = np.asarray([0 if i == "td" else 1 for ind, i in enumerate(appended_data["label"].values)], dtype=np.int64)
unique,counts = np.unique(Y,return_counts=True)
'''
#unique,counts = np.unique(Y,return_counts=True)
#print(unique)
#print(counts)
print(Y)
print(Y.shape)
pdb.set_trace()

create_kfold_partitions(new_X2,Y,"/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/abide_asd_td_dev_age/" , 256,64,5)
pdb.set_trace()
final_list = pd.read_csv('final_list/abide_noNaNs_matched_motion_age_sex_n690.csv')
final_list['subject_id'] = final_list['subject_id'].astype(int)

mask = np.isin(appended_data['subject_id'],final_list['subject_id'])
appended_data = appended_data[mask]
new_X3 = np.asarray([np.asarray(i) for ind,i in enumerate(appended_data['updated_data'].values)])
Y3 = np.asarray([0 if i == "td" else 1 for ind, i in enumerate(appended_data["label"].values)], dtype=np.int64)
print(new_X3.shape)
print(Y3.shape)
unique,counts = np.unique(Y3,return_counts=True)
print(unique)
print(counts)
pdb.set_trace() 

create_kfold_partitions(new_X3,Y3,"/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/abide_asd_final_yuan_list_noNaNs/" , 256,64,5)
pdb.set_trace()




appended_data = appended_data.rename(columns={'subject_id':'PID'})
appended_data['PID'] = appended_data['PID'].astype(int)
matched_list['PID'] = matched_list['PID'].astype(int)
merged_df = pd.merge(appended_data,matched_list,on='PID')
print(merged_df)
merged_df['label_x'] = merged_df['label_x'].astype(str)
new_X = truncate_predefined_length(merged_df['data'], 180)
X = np.asarray([np.asarray(i) for i in new_X])
pdb.set_trace()
labels = np.asarray([0 if i == "low" else 1 for ind, i in enumerate(merged_df["motion_2class"].values)], dtype=np.int64)
#pdb.set_trace()
indices_to_remove = [ind for ind, i in enumerate(X) if (len(i)!= 180 or np.sum(np.isnan(i)) > 0)]
data = np.asarray([np.asarray(i) for ind, i in enumerate(X) if ind not in indices_to_remove])
labels = np.asarray([0 if i == "td" else 1 for ind, i in enumerate(merged_df["label_x"].values) if ind not in indices_to_remove], dtype=np.int64)
#merged_df = merged_df[~pd.isna(merged_df)]
#print(merged_df)
#print(indices_to_remove)
#pdb.set_trace()
#print(X.shape)
#labels = np.asarray([0 if i == "td" else 1 for ind, i in enumerate(merged_df["label_x"].values)], dtype=np.int64)
print(data)
print(data.shape)
print(labels)
print(labels.shape)
unique,counts = np.unique(labels,return_counts=True)
print(unique)
print(counts)
pdb.set_trace()
create_kfold_partitions(data,labels,"/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/abide_asd_matched_no_window/" , 256,64,5)

#indices_to_remove = [ind for ind, i in enumerate(merged_df["data"].values) if (len(i)!= 180 or np.sum(np.isnan(i)) > 0)]
#print(indices_to_remove)
pdb.set_trace()

ids = list(chain(*ids))
mean_fds = list(chain(*mean_fds))
labels = list(chain(*labels))
genders = list(chain(*genders))
sites = list(chain(*sites))
ages = list(chain(*ages))
df = pd.DataFrame({'PID':ids,'mean_fd':mean_fds,'age':ages,'gender':genders,'site':sites,'label':labels})
df['label'] = df['label'].astype(str)
df['PID'] = df['PID'].astype(int)
final_list_asd = pd.read_excel('ABIDE_ASD_subjectlist.xlsx')
final_list_asd['PID'] = final_list_asd['PID'].astype(int)
final_list_td = pd.read_excel('ABIDE_TD_subjectlist.xlsx')
final_list_td['PID'] = final_list_td['PID'].astype(int)
df_asd = df.loc[df['label'] == 'asd']
df_td = df.loc[df['label'] == 'td']
merged_df_asd = pd.merge(df_asd,final_list_asd,on='PID')
merged_df_td = pd.merge(df_td,final_list_td,on='PID')
print(merged_df_asd)
print(merged_df_td)
pdb.set_trace()
#print(datao.files)
#pdb.set_trace()
#labels = np.asarray([0 if i == "male" else 1 for i in datao["gender"].values], dtype=np.int64)

data = np.asarray([np.asarray(i) for i in datao["data"]])
labels = np.asarray(datao["labels"], dtype=np.int64)
X = data
Y = labels
unique,labels = np.unique(Y,return_counts=True)
print(unique)
print(labels)
pdb.set_trace()
print("Raw Data shape:", X.shape)
print("Labels shape:", Y.shape)

create_pretrain_data(X, "/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/hcp_gender_no_window/" , 256, 64)
create_kfold_partitions(X,Y,"/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/hcp_gender_no_window/" , 256,64,5)
