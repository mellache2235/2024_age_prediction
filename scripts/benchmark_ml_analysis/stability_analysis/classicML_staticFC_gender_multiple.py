import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing, linear_model, model_selection, svm, tree, ensemble, neighbors
import pickle
from sklearn.preprocessing import LabelEncoder
import random
import math
import pdb
import statistics as st

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

def data_cleaning_hcp(path_to_dataset):
    # load & clean data
    datao = pd.read_pickle(path_to_dataset)
    datao.drop(datao[datao['percentofvolsrepaired'] > 10].index, inplace=True)
    datao.drop(datao[datao['mean_fd'] > 0.5].index, inplace=True)
    datao = datao.reset_index()
    data = np.asarray([np.asarray(lst)[:, :] for lst in datao.data])
    labels_gender = datao['gender']
    # subjid = datao['subject_id']

    labels = []
    for i in labels_gender:
        if i == 'male':
            labels.append(0)
        else:
            labels.append(1)
    labels = np.asarray(labels)
    # print("data dimension: {}".format(data.shape)) # no_subj, no_ts, no_roi
    return data, labels


def data_cleaning_nkirs(path_to_dataset):
    # Load and clean
    datao = pd.read_pickle(path_to_dataset)
    datao.drop(datao[datao['percentofvolsrepaired'] > 10].index, inplace=True)
    datao.drop(datao[datao['mean_fd'] > 0.5].index, inplace=True)
    #datao.drop(datao[datao['age'] < 22].index, inplace=True)
    #datao.drop(datao[datao['age'] > 35].index, inplace=True)
    datao.reset_index(inplace=True)

    #data_sel = [idx for idx in range(datao.shape[0]) if len(datao['data'][idx]) == 900]
    #datao = datao.iloc[data_sel]
    #datao.reset_index(inplace=True)
    indices_to_remove = [ind for ind, i in enumerate(datao["data"].values) if (len(i)!= 900 or np.sum(np.isnan(i)) > 0)]
    labels = np.asarray([0 if i == "male" else 1 for ind, i in enumerate(datao["gender"].values) if ind not in indices_to_remove], dtype=np.int64)
    data = np.asarray([np.asarray(i) for ind, i in enumerate(datao["data"].values) if ind not in indices_to_remove])

    #data = np.asarray([np.asarray(lst)[:, :] for lst in datao.data])
    #labels_gender = datao['gender']
    #subjid = datao['subject_id']
    '''
    labels = []
    for i in labels_gender:
        if i == 'male':
            labels.append(0)
        else:
            labels.append(1)
    labels = np.asarray(labels)
    '''
    print("data dimension: {}".format(data.shape))

    return data, labels

def data_cleaning_hcp_dev(path_to_dataset):
    # load & clean data
    datao = pd.read_pickle(path_to_dataset)
    datao.drop(datao[datao['percentofvolsrepaired'] > 10].index, inplace=True)
    datao.drop(datao[datao['mean_fd'] > 0.5].index, inplace=True)
    datao = datao.reset_index()
    labels = np.asarray([0 if i == "male" else 1 for i in datao["gender"].values], dtype=np.int64)
    data = np.asarray([np.asarray(i) for i in datao["data"].values])
    # print("data dimension: {}".format(data.shape)) # no_subj, no_ts, no_roi
    return data, labels

def data_cleaning_leipzig_gender(path_to_dataset):
    # load & clean data
    datao = pd.read_pickle(path_to_dataset)
    datao.drop(datao[datao['percentofvolsrepaired'] > 10].index, inplace=True)
    datao.drop(datao[datao['mean_fd'] > 0.5].index, inplace=True)
    datao = datao.reset_index()
    indices_to_remove = [ind for ind, i in enumerate(datao["data"].values) if (len(i)!= 657 or np.sum(np.isnan(i)) > 0)]
    #print(indices_to_remove)
    #pdb.set_trace()
    data = np.asarray([np.asarray(i) for ind, i in enumerate(datao["data"].values) if ind not in indices_to_remove])
    labels_gender = np.asarray([np.asarray(i) for ind, i in enumerate(datao["gender"].values) if ind not in indices_to_remove])
    #data_sel = [idx for idx in range(datao.shape[0]) if len(datao['data'][idx]) == 657]
    #datao = datao.iloc[data_sel]
    #datao.reset_index(inplace=True)
    #data = np.asarray([np.asarray(lst)[:, :] for lst in datao.data])
    #labels_gender = datao['gender']
    # subjid = datao['subject_id']
    
    labels = []
    for i in labels_gender:
        if i == 'male':
            labels.append(0)
        else:
            labels.append(1)
    labels = np.asarray(labels)
    # print("data dimension: {}".format(data.shape)) # no_subj, no_ts, no_roi
    return data, labels

def data_cleaning_abide_asd(path_to_dataset):
    # load & clean data
    datao = np.load(path_to_dataset,allow_pickle=True)
    #datao.drop(datao[datao['percentofvolsrepaired'] > 10].index, inplace=True)
    #datao.drop(datao[datao['mean_fd'] > 0.5].index, inplace=True)
    #datao = datao.reset_index()
    data = datao['data']
    labels = np.asarray(datao['labels'], dtype=np.int64)
    labels -= 1
    # subjid = datao['subject_id']

    # print("data dimension: {}".format(data.shape)) # no_subj, no_ts, no_roi
    return data, labels

def data_cleaning_abide_asd_matched(path_to_dataset):
    saved_data_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/abide_asd_final_yuan_list_noNaNs/fold_0.bin'
    X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(saved_data_path)

    X_total = np.concatenate((X_train,X_valid))
    Y_total = np.concatenate((Y_train,Y_valid))

    return X_total, Y_total

def data_cleaning_bsnip(path_to_dataset):
    # load & clean data
    datao = pd.read_pickle(path_to_dataset)
    datao.drop(datao[datao['percentofvolsrepaired'] > 10].index, inplace=True)
    datao.drop(datao[datao['mean_fd'] > 0.5].index, inplace=True)
    datao = datao.reset_index()
    #data = np.asarray([np.asarray(lst)[:, :] for lst in datao.data])
    #labels_gender = datao['gender']
    # subjid = datao['subject_id']
    indices_to_remove = [ind for ind, i in enumerate(datao["data"].values) if np.sum(np.isnan(i)) > 0]
    labels = np.asarray([int(i) for ind, i in enumerate(datao["label"].values) if ind not in indices_to_remove], dtype=np.int64)
    data = np.asarray([np.asarray(i) for ind, i in enumerate(datao["data"].values) if ind not in indices_to_remove])
    ix = labels != 2
    labels = labels[ix]
    data = data[ix]


    return data, labels

def data_cleaning_hcp_early_psychosis(path_to_dataset):
    # load & clean data
    datao = pd.read_pickle(path_to_dataset)
    datao.drop(datao[datao['percentofvolsrepaired'] > 10].index, inplace=True)
    datao.drop(datao[datao['mean_fd'] > 0.5].index, inplace=True)
    datao = datao.reset_index()
    data = np.asarray([np.asarray(lst)[:, :] for lst in datao.data])
    #labels_gender = datao['gender']
    label_dict = {'Non-Affective': 0, 'Healthy Control': 1, 'Affective': 2, 'Psychosis': 3}
    # subjid = datao['subject_id']
    labels = np.asarray([label_dict[i] for i in datao["label"].values], dtype=np.int64)
    data = np.asarray([np.asarray(i) for i in datao["data"].values])
    #labels = []
    ix = labels != 3
    data = data[ix]
    labels = labels[ix]
    labels_new = np.zeros(labels.shape[0])
    ix = labels == 1
    labels_new[ix] = 0
    ix = labels == 0
    labels_new[ix] = 1
    ix = labels == 2
    labels_new[ix] = 1
    labels_new = labels_new.astype('int')
    # print("data dimension: {}".format(data.shape)) # no_subj, no_ts, no_roi
    return data, labels_new

def data_cleaning_adhd200(path_to_dataset):
    # load & clean data
    datao = pd.read_pickle(path_to_dataset)
    datao.drop(datao[datao['percentofvolsrepaired'] > 10].index, inplace=True)
    datao.drop(datao[datao['mean_fd'] > 0.5].index, inplace=True)
    datao.drop(datao[datao['tr'] == 2.5].index, inplace=True)
    #datao = datao.reset_index()
    datao.drop(datao[datao['label'] == 'pending'].index, inplace=True)
    #ix = datao['label'] != 0
    #datao['label'][ix] = 1
    datao = datao.reset_index()
    data = np.asarray([np.asarray(i)[len(i)-174:] for i in datao["data"].values])
    labels = datao['label']
    ix = labels != 0
    labels[ix] = 1
    samples_to_keep = list()
    num_nans = 0
    for i in range(data.shape[0]):
        if not np.isnan(np.sum(data[i])):
            samples_to_keep.append(i)
        else:
            num_nans += 1
    data = data[samples_to_keep]
    labels = labels[samples_to_keep]
    # subjid = datao['subject_id']
    labels = labels.astype('int')
    df = pd.DataFrame(labels)
    df = df.reset_index(drop=True)
    labels = df.to_numpy()
    # print("data dimension: {}".format(data.shape)) # no_subj, no_ts, no_roi
    return data, labels

def data_cleaning_oasis_ad(path_to_dataset):
    # load & clean data
    datao = pd.read_pickle(path_to_dataset)
    datao.drop(datao[datao['percentofvolsrepaired'] > 10].index, inplace=True)
    datao.drop(datao[datao['mean_fd'] > 0.5].index, inplace=True)
    datao = datao.reset_index()
    data = datao
    '''
    data = np.asarray([np.asarray(lst)[:, :] for lst in datao.data])
    labels_gender = datao['gender']
    # subjid = datao['subject_id']

    labels = []
    for i in labels_gender:
        if i == 'male':
            labels.append(0)
        else:
            labels.append(1)
    labels = np.asarray(labels)
    # print("data dimension: {}".format(data.shape)) # no_subj, no_ts, no_roi
    '''
    data['days_to_visit'] = data['days_to_visit'].astype(str).astype(int)

    X = np.asarray([np.asarray(i) for i in data["data"].values])

    lst = []
    for i in range(X.shape[0]):
        if X[i].shape[0] != 164:
            lst.append(i)
    data = data.drop(lst)

    X = np.asarray([np.asarray(i) for i in data["data"].values])

    ####### Cases
    y1 = np.asarray(data["PROBAD"])
    y2 = np.asarray(data["POSSAD"])
    y_condition = ((y1 == '1') | (y2 == '1'))
    X_case = X[y_condition]
    data_condition = data[y_condition]
    data_condition['sessionid'] = data_condition['sessionid'].astype(str)
    data_condition['sessionid'] = data_condition['sessionid'].apply(lambda x: x[len(x) - 4:])
    data_condition['sessionid'] = data_condition['sessionid'].astype(int)
    data_condition_limit = data_condition.loc[abs(data_condition['sessionid'] - data_condition['days_to_visit']) < 366]
    X_condition = np.asarray([np.asarray(i) for i in data_condition_limit["data"].values])
    y_case = []
    for i in range(X_condition.shape[0]):
        y_case.append(1)
    y_case = np.asarray(y_case)

    ####### Controls
    #X_control = X_id[y_control_condition]
    y_control = data["NORMCOG"].values
    data_control = data[y_control == '1']
    #data_control_limit = data_control.loc[data_control['days_to_visit'] < 366]
    data_control['sessionid'] = data_control['sessionid'].astype(str)
    data_control['sessionid'] = data_control['sessionid'].apply(lambda x: x[len(x) - 4:])
    data_control['sessionid'] = data_control['sessionid'].astype(int)
    data_control_limit = data_control.loc[abs(data_control['sessionid'] - data_control['days_to_visit']) < 366]
    X_control = np.asarray([np.asarray(i) for i in data_control_limit["data"].values])
    y_control = []
    for i in range(X_control.shape[0]):
        y_control.append(0)
    y_control = np.asarray(y_control)

    ####### Combine Cases and Controls
    X = np.concatenate((X_control,X_condition))
    Y = np.concatenate((y_control,y_case))

    return X, Y

def data_cleaning_pd(path_to_dataset):
    # load & clean data
    datao = pd.read_pickle(path_to_dataset)
    datao.drop(datao[datao['percentofvolsrepaired'] > 10].index, inplace=True)
    datao.drop(datao[datao['mean_fd'] > 0.5].index, inplace=True)
    datao.drop(datao[datao['tr'] == 2.5].index, inplace=True)
    datao.drop(datao[datao['label'] == 'AD'].index, inplace=True)
    datao.drop(datao[datao['label'] == 'PD_on'].index, inplace=True)
    datao = datao.reset_index()
    data = np.asarray([np.asarray(i) for i in datao["data"].values])
    labels = np.zeros((len(datao['label']),),dtype=int)
    ix = datao['label'] == 'HC'
    labels[ix] = 0
    ix = datao['label'] == 'PD_off'
    labels[ix] = 1
    labels = labels.astype('int')
    '''
    labels_gender = datao['gender']
    # subjid = datao['subject_id']

    labels = []
    for i in labels_gender:
        if i == 'male':
            labels.append(0)
        else:
            labels.append(1)
    labels = np.asarray(labels)
    # print("data dimension: {}".format(data.shape)) # no_subj, no_ts, no_roi
    '''
    return data, labels

def data_cleaning_22q(path_to_dataset):
    # load & clean data
    datao = pd.read_pickle(path_to_dataset)
    datao.drop(datao[datao['percentofvolsrepaired'] > 10].index, inplace=True)
    datao.drop(datao[datao['mean_fd'] > 0.5].index, inplace=True)
    datao = datao.reset_index()
    data = np.asarray([np.asarray(lst)[:, :] for lst in datao.data])
    labels = np.asarray([int(i) for i in datao["label"].values],dtype=np.int64)
    # print("data dimension: {}".format(data.shape)) # no_subj, no_ts, no_roi
    return data, labels

def data_cleaning_ucla(path_to_dataset):
    # load & clean data
    datao = pd.read_pickle(path_to_dataset)
    datao.drop(datao[datao['percentofvolsrepaired'] > 10].index, inplace=True)
    datao.drop(datao[datao['mean_fd'] > 0.5].index, inplace=True)
    datao.drop(datao[datao['tr'] == 2.5].index, inplace=True)
    datao.drop(datao[datao['label'] == 'SCHZ'].index, inplace=True)
    datao.drop(datao[datao['label'] == 'ADHD'].index, inplace=True)
    datao = datao.reset_index()
    #data = np.asarray([np.asarray(lst)[:, :] for lst in datao.data])
    #labels_gender = datao['gender']
    # subjid = datao['subject_id']
    indices_to_remove = [ind for ind, i in enumerate(datao["data"].values) if (len(i)!= 152 or np.sum(np.isnan(i)) > 0)]
    #print(indices_to_remove)
    X = np.asarray([np.asarray(i) for ind,i in enumerate(datao["data"].values) if ind not in indices_to_remove])
    labels = np.asarray([0 if i == "CONTROL" else 1 for ind, i in enumerate(datao["label"].values) if ind not in indices_to_remove], dtype=np.int64)
    '''
    labels = []
    for i in labels_gender:
        if i == 'male':
            labels.append(0)
        else:
            labels.append(1)
    labels = np.asarray(labels)
    '''
    # print("data dimension: {}".format(data.shape)) # no_subj, no_ts, no_roi
    return X, labels

def data_cleaning_adhd_fukui(path_to_dataset):
    # load & clean data
    datao = pd.read_pickle(path_to_dataset)
    #datao.drop(datao[datao['percentofvolsrepaired'] > 10].index, inplace=True)
    datao['subject_id'] = datao['subject_id'].astype('str')
    datao['meanFD'] = datao['meanFD'].astype(float)
    datao.drop(datao[datao['meanFD'] > 0.5].index, inplace=True)
    datao = datao.reset_index()
    labels = np.asarray(datao['label'])
    labels = labels.astype(str).astype(int)
    df_meds = datao[labels == 1]
    df_placebo = datao[labels == 2]

    X = np.asarray([np.asarray(i) for i in df_meds["data"].values])
    X_placebo = np.asarray([np.asarray(i) for i in df_placebo["data"].values])
    print(X.shape)
    print(X_placebo.shape)
    iirv_data = pd.read_excel('Fukui_all_data_AllinGrp.xlsx')
    iirv_data['ID'] = iirv_data['ID'].astype('str')
    iirv_data_meds = iirv_data[iirv_data['ID'].isin(df_meds['subject_id'])]
    iirv_data_placebo = iirv_data[iirv_data['ID'].isin(df_placebo['subject_id'])]

    d = {'ids' : iirv_data_meds['ID'], 'time_data' : list(X)}
    d_placebo = {'ids' : iirv_data_placebo['ID'], 'time_data': list(X_placebo)}
    frame = pd.DataFrame(d)
    frame_placebo = pd.DataFrame(d_placebo)

    vals = np.asarray(iirv_data_meds.loc[iirv_data_meds['ID'].isin(frame['ids']),'MED_T.Score.SD_of_reationtime.'])
    vals_placebo = np.asarray(iirv_data_placebo.loc[iirv_data_placebo['ID'].isin(frame_placebo['ids']),'MED_T.Score.SD_of_reationtime.'])
    ### Target is delta
    delta = vals - vals_placebo
    #print(delta)
    #pdb.set_trace()
    ### If you want to look at placebo instead of meds, replace frame with frame_placebo
    X = np.asarray([np.asarray(i) for i in frame['time_data'].values])
    #print(X.shape)
    #pdb.set_trace()
    #print(frame['ids'])
    #print(iirv_data_placebo['ID'])
    #pdb.set_trace()

    time_series_data = X[frame['ids'].isin(iirv_data['ID'])]
    #print(time_series_data)
    #pdb.set_trace()
    ### Perform median split of IIRV
    truther = lambda t: 0 if t else 1
    vfunc = np.vectorize(truther)
    frame["median_split_IIRV"] = vfunc(delta < np.median(delta))

    #frame["IIRV"] = delta
    # 0: Good Response
    # 1: Poor Response
    X = time_series_data
    Y = np.asarray(frame["median_split_IIRV"])
    # subjid = datao['subject_id']
    '''
    labels = []
    for i in labels_gender:
        if i == 'male':
            labels.append(0)
        else:
            labels.append(1)
    labels = np.asarray(labels)
    # print("data dimension: {}".format(data.shape)) # no_subj, no_ts, no_roi
    '''
    return X, Y

def data_cleaning_pd_MDS(path_to_dataset):
    # load & clean data
    datao = pd.read_pickle(path_to_dataset)
    datao.drop(datao[datao['percentofvolsrepaired'] > 10].index, inplace=True)
    datao.drop(datao[datao['mean_fd'] > 0.5].index, inplace=True)
    #datao = datao.reset_index()
    mds_data = pd.read_csv('HCPDcombined_DemosNtaskscores_info_4_MDS_Analysis_v2.csv')
    mds_data = mds_data.dropna(subset=['MDS_updrs_ON'])
    mds_data = mds_data.drop_duplicates(subset=['PID'],keep='last')
    mds_data['PID'] = mds_data['PID'].astype(str)
    datao['subject_id'] = datao['subject_id'].astype(str)
    df = datao.loc[datao['subject_id'].isin(mds_data['PID']),:]
    df = df.reset_index()
    df_on = df[df['label'] == 'PD_on']
    df_off = df[df['label'] == 'PD_off']

    # Get patients who have pd_on and pd_off status (to compute MDS delta)
    ids = list(set(df_on['subject_id']) & set(df_off['subject_id']))
    df = df.loc[df['subject_id'].isin(ids),:]

    # Keep patients with pd_on status
    df = df.drop_duplicates(subset=['subject_id'],keep='first')

    # Get MDS on score
    mds_scores_on = np.asarray(mds_data.loc[mds_data['PID'].isin(df['subject_id']),'MDS_updrs_ON'])
    # Get MDS off score
    mds_scores_off = np.asarray(mds_data.loc[mds_data['PID'].isin(df['subject_id']),'MDS_updrs_OFF'])
    # Compute MDS on - MDS off
    delta = mds_scores_on - mds_scores_off
    #print(delta)
    #pdb.set_trace()
    # Perform median split of MDS
    X = np.asarray([np.asarray(i) for i in df["data"].values])
    #truther = lambda t: 0 if t else 1
    #vfunc = np.vectorize(truther)
    #df["median_split_MDS"] = vfunc(delta)
    truther = lambda t: 0 if t else 1
    vfunc = np.vectorize(truther)
    df["median_split_MDS"] = vfunc(delta < np.median(delta))
    Y = np.asarray(df["median_split_MDS"])

    #data = np.asarray([np.asarray(lst)[:, :] for lst in datao.data])
    #labels_gender = datao['gender']
    # subjid = datao['subject_id']
    '''
    labels = []
    for i in labels_gender:
        if i == 'male':
            labels.append(0)
        else:
            labels.append(1)
    labels = np.asarray(labels)
    # print("data dimension: {}".format(data.shape)) # no_subj, no_ts, no_roi
    '''
    return X, Y

def data_cleaning_embarc(path_to_dataset):
    # load & clean data
    datao = pd.read_pickle(path_to_dataset)
    datao.drop(datao[datao['percentofvolsrepaired'] > 10].index, inplace=True)
    datao.drop(datao[datao['mean_fd'] > 0.5].index, inplace=True)
    datao = datao.reset_index()
    data_subset = datao[
        (datao["Stage1TX"] == "SER")
        & (datao["phenotype"] == "MDD")
        & (datao["mean_fd"] < 0.5)
        & (~pd.isna(datao["data"]))
    ]
    data_subset = data_subset.dropna(subset=['w8_responder','hamd_36_week_0','hamd_36_week_8'])


    ### If target variable is week 8 - week 0, do median split
    delta = data_subset['hamd_36_week_8'] - data_subset['hamd_36_week_0']
    data_subset["median_split"] = (delta<delta.quantile()).replace({True:0, False:1})
    indices_to_remove = [ind for ind, i in enumerate(data_subset["data"].values) if (len(i)!= 180 or np.sum(np.isnan(i)) > 0)]
    Y = np.asarray([int(i) for ind, i in enumerate(data_subset["median_split"].values) if ind not in indices_to_remove], dtype=np.int64)
    #Y = np.asarray([int(i) for ind, i in enumerate(delta) if ind not in indices_to_remove], dtype=np.int64)
    X = np.asarray([np.asarray(i) for ind, i in enumerate(data_subset["data"].values) if ind not in indices_to_remove])

    '''
    data = np.asarray([np.asarray(lst)[:, :] for lst in datao.data])
    labels_gender = datao['gender']
    # subjid = datao['subject_id']

    labels = []
    for i in labels_gender:
        if i == 'male':
            labels.append(0)
        else:
            labels.append(1)
    labels = np.asarray(labels)
    # print("data dimension: {}".format(data.shape)) # no_subj, no_ts, no_roi
    '''
    return X, Y

def get_features_labels(path_to_dataset, site):
    if site == 'hcp':
        data, labels = data_cleaning_hcp(path_to_dataset)
    elif site == 'nkirs':
        data, labels = data_cleaning_nkirs(path_to_dataset)
    elif site == 'abide_asd':
        data, labels = data_cleaning_abide_asd_matched(path_to_dataset)
    elif site == 'bsnip':
        data, labels = data_cleaning_bsnip(path_to_dataset)
    elif site == 'hcp_early_psychosis':
        data, labels = data_cleaning_hcp_early_psychosis(path_to_dataset)
    elif site == 'adhd200':
        data, labels = data_cleaning_adhd200(path_to_dataset)
    elif site == '22q':
        data, labels = data_cleaning_22q(path_to_dataset)
    elif site == 'oasis_ad':
        data, labels = data_cleaning_oasis_ad(path_to_dataset)
    elif site == 'pd':
        data, labels = data_cleaning_pd(path_to_dataset)
    elif site == 'hcp_dev':
        data, labels = data_cleaning_hcp_dev(path_to_dataset)
    elif site == 'leipzig_gender':
        data, labels = data_cleaning_leipzig_gender(path_to_dataset)
    elif site == 'ucla_bipolar':
        data, labels = data_cleaning_ucla(path_to_dataset)
    elif site == 'adhd_fukui':
        data, labels = data_cleaning_adhd_fukui(path_to_dataset)
    elif site == 'pd_MDS':
        data, labels = data_cleaning_pd_MDS(path_to_dataset)
    elif site == 'embarc':
        data, labels = data_cleaning_embarc(path_to_dataset)
 
    print("data dimension: {}".format(data.shape))  # no_subj, no_ts, no_roi
    print("labels dimension: {}".format(labels.shape))

    # generate static FC features
    no_subjs, no_ts, no_rois = data.shape
    data_fcz = np.empty((no_subjs, int(no_rois * (no_rois - 1) / 2)))
    print('data_fcz dimension {}'.format(data_fcz.shape))

    for subj in range(no_subjs):
        # print(subj)
        x_subj = data[subj, :, :]
        df_subj = pd.DataFrame(x_subj)
        fc_subj = df_subj.corr('pearson')  # get correlation matrix
        fc_subj = fc_subj.to_numpy()
        # get upper tri elements of the FC matrix and apply fisher z transformation
        data_fcz[subj, :] = np.arctanh(fc_subj[np.triu_indices(fc_subj.shape[0], k=1)])

    return data_fcz, labels

def perform_stability_analysis(path_to_dataset,site,sample_size,times_to_sample,imbalance_ratio,output_path,model_ss):
    data, labels = get_features_labels(path_to_dataset,site)
    ix = labels == 1
    X_class_1 = data[ix]
    labels_class_1 = labels[ix]
    ix = labels == 0
    X_class_0 = data[ix]
    labels_class_0 = labels[ix]
    
    #stability_frame_acc = np.zeros((times_to_sample,7))
    #stability_frame_f1 = np.zeros((times_to_sample,7))
    #stability_frame_acc_sd = np.zeros((times_to_sample,7))
    #stability_frame_f1_sd = np.zeros((times_to_sample,7))
    #results = {}
    total_accs = []
    total_f1s = []
    models = []
    #models.append(('linSVM', svm.SVC(kernel='linear')))
    #models.append(('KNN', (neighbors.KNeighborsClassifier())))
    #models.append(('DT', tree.DecisionTreeClassifier()))
    #models.append(('LR', linear_model.LogisticRegression()))
    # models.append(('rbfSVM', svm.SVC(kernel='rbf')))
    #models.append(('RC', linear_model.RidgeClassifier(alpha=0.5)))
    #models.append(('LASSO', linear_model.LogisticRegression(penalty='l1', solver='liblinear')))
    # models.append(('ELNet', linear_model.LogisticRegression(penalty='elasticnet', l1_ratio=0.5, solver='saga')))
    models.append(('RF', ensemble.RandomForestClassifier()))
    for i in range(times_to_sample):
        print("Sampling:",i)
        #repetition_results = {}
        indexes = random.sample([i for i in range(len(X_class_1))],math.floor(sample_size/imbalance_ratio))
        X_class_1_subset = X_class_1[indexes]
        Y_class_1_subset = np.ones((math.floor(sample_size/imbalance_ratio),),dtype=int)
        indexes = random.sample([i for i in range(len(X_class_0))],sample_size)
        X_class_0_subset = X_class_0[indexes]
        Y_class_0_subset = np.zeros((sample_size,),dtype=int)
        new_X = np.concatenate((X_class_1_subset,X_class_0_subset))
        new_Y = np.concatenate((Y_class_1_subset,Y_class_0_subset))
        new_X_train,X_valid,new_Y_train,Y_valid = train_test_split(new_X,new_Y,test_size=0.2,random_state=42)
        accs_model = []
        f1s_model = []
        for name,model in models:
            print("Running:",name)
            fname_model = output_path + 'model_' + name + '_' + model_ss + '_sampling_' + str(times_to_sample) + '.sav'
            print('*** Classifier: ', name)
            modelfit = model.fit(new_X_train, new_Y_train)
            y_val_predicted = modelfit.predict(X_valid)
            pickle.dump(modelfit, open(fname_model, 'wb'))
            classifreport_dict = classification_report(Y_valid, y_val_predicted,
                                                       output_dict=True)
            accuracy = classifreport_dict['accuracy'] * 100
            #precision = classifreport_dict['macro avg']['precision'] * 100
            #recall = classifreport_dict['macro avg']['recall'] * 100
            f1score = classifreport_dict['macro avg']['f1-score'] * 100
            accs_model.append(accuracy)
            f1s_model.append(f1score)
            print("Finished running:",name)
            #repetition_results[f"{name}"] = {"Accuracy": accuracy, "F1-Score":f1score}
         #results[f"Repitition_{i+1}"] = repitition_results
        total_accs.append(accs_model)
        total_f1s.append(f1s_model)
    mean_acc = np.mean(total_accs,axis=0)
    std_acc = np.std(total_accs,axis=0)
    mean_f1 = np.mean(total_f1s,axis=0)
    std_f1 = np.std(total_f1s,axis=0)
    np.savez(f"stability_analysis_rf_model_{site}_final_yuan_list_sample_size_{sample_size}_control_asd_ratio_{imbalance_ratio}_1_sample_{times_to_sample}_times_results.npz",mean_accuracy = mean_acc,std_accuracy=std_acc,mean_f1=mean_f1,std_f1=std_f1)
 
    return mean_acc,std_acc,mean_f1,std_f1

if __name__ == '__main__':
    abide_asd_data = '/oak/stanford/groups/menon/projects/sryali/2021_foundation_model/data/imaging/for_dnn/abide_resting/clean_abide.npz'
    leipzig_gender_data = '/oak/stanford/groups/menon/deriveddata/public/mpi_leipzig/restfmri/timeseries/group_level/brainnetome/normz/mpi_leipzig_run-AP_run-01_brainnetome_mean_regMov-6param_wmcsf_dt1_bpf008-09_normz_246ROIs.pklz'
    output_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/results/dnn/models/finetuning_abide_asd/ml_models/'
    model_ss = 'abide_asd_final_yuan_list_rf_only'
    site = 'abide_asd'
    times_to_sample = 40
    for sample_size in [50,100,200,340]:
        for imbalance_ratio in [1,2,3,4]:
            print("Sample Size:",sample_size)
            print("Imbalance Ratio:",imbalance_ratio)
            accs,acc_stds,f1s,f1_stds = perform_stability_analysis(abide_asd_data,site,sample_size,times_to_sample,imbalance_ratio,output_path,model_ss)
            print(accs)
            print(acc_stds)
            print(f1s)
            print(f1_stds)
