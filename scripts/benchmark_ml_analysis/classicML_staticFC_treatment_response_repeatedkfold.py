import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing, linear_model, model_selection, svm, tree, ensemble, neighbors
import pickle
from sklearn.preprocessing import LabelEncoder
import pdb
import random

NP_RANDOM_SEED = 652
PYTHON_RANDOM_SEED = 819

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


def data_cleaning_adhd_fukui_matched(path_to_dataset):
    saved_data_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/adhd_fukui_IIRV_final_yuan_list/fold_0.bin'

    X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(saved_data_path)

    X_total = np.concatenate((X_train,X_valid))
    Y_total = np.concatenate((Y_train,Y_valid))

    return X_total, Y_total


def data_cleaning_pd_MDS_easy(path_to_dataset):
    saved_data_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/pd_on_MDS/fold_0.bin'

    X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(saved_data_path)

    X_total = np.concatenate((X_train,X_valid))
    Y_total = np.concatenate((Y_train,Y_valid))

    return X_total, Y_total


def data_cleaning_embarc_matched(path_to_dataset):
    saved_data_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/embarc_treatment_SER_w8_final_yuan_list/fold_0.bin'

    X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(saved_data_path)

    X_total = np.concatenate((X_train,X_valid))
    Y_total = np.concatenate((Y_train,Y_valid))

    return X_total, Y_total

def data_cleaning_embarc_original_sample(path_to_dataset):
    saved_data_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/embarc_treatment_SER_w8_original_sample/fold_0.bin'

    X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(saved_data_path)

    X_total = np.concatenate((X_train,X_valid))
    Y_total = np.concatenate((Y_train,Y_valid))

    return X_total, Y_total

def get_features_labels(path_to_dataset, site):
    if site == 'adhd_fukui':
        data, labels = data_cleaning_adhd_fukui_matched(path_to_dataset)
    elif site == 'pd_MDS':
        data, labels = data_cleaning_pd_MDS_easy(path_to_dataset)
    elif site == 'embarc':
        data, labels = data_cleaning_embarc_original_sample(path_to_dataset)
 
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

if __name__ == '__main__':
    np.random.seed(NP_RANDOM_SEED)
    random.seed(PYTHON_RANDOM_SEED)
    #print("models trained on HCP Session 3, tested on HCP Session 3, Session 4, and NKI-RS")
    adhd_fukui_data = '/oak/stanford/groups/menon/projects/wdcai/2019_ADHD_NN/data/imaging/timeseries/Fukui/group_level/brainnetome/normz/fukui_mph_run-resting_brainnetome_mean_regMov-6param_wmcsf_dt1_bpf008-09_normz_246ROIs.pklz'
    pd_MDS_data = '/oak/stanford/groups/menon/deriveddata/public/pd/restfmri/timeseries/group_level/brainnetome/normz/pd_run-resting_brainnetome_mean_regMov-6param_wmcsf_dt1_bpf008-09_normz_246ROIs.pklz'
    embarc_data = '/oak/stanford/groups/menon/deriveddata/public/embarc/restfmri/timeseries/group_level/brainnetome/normz/embarc_run-run-01_brainnetome_mean_regMov-6param_wmcsf_dt1_bpf008-09_normz_246ROIs_behav.pklz'
    
    model_ss = 'pd_MDS_seed_set'

    output_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/results/dnn/models/finetuning_pd/ml_models/'
    #output_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/results/dnn/models/'
    #f1 = output_path + 'sFC_HCP_Session3.npz'
    #f2 = output_path + 'sFC_HCP_Session4.npz'
    #f3 = output_path + 'sFC_NKI-RS.npz'
    f1 = output_path + 'sFC_abide_asd_matched.npz'
    f2 = output_path + 'sFC_bsnip.npz'
    f3 = output_path + 'sFC_adhd200.npz'
    f4 = output_path + 'sFC_early_psychosis.npz'
    f5 = output_path + 'sFC_22q.npz'
    f6 = output_path + 'sFC_oasis_ad.npz'
    f7 = output_path + 'sFC_pd.npz'
    f8 = output_path + 'sFC_hcp_gender_final.npz'
    f9 = output_path + 'sFC_nki_gender.npz'
    f10 = output_path + 'sFC_hcp_dev.npz'
    f11 = output_path + 'sFC_leipzig_gender.npz'
    f12 = output_path + 'sFC_ucla_bipolar.npz'
    f13 = output_path + 'sFC_adhd_fukui_IIRV.npz'
    f14 = output_path + 'sFC_pd_MDS.npz'
    f15 = output_path + 'sFC_embarc_hamd.npz'
    f16 = output_path + 'sFC_ucla_schiz.npz'
    f17 = output_path + 'sFC_stanford_autism_matched.npz'
    f18 = output_path + 'sFC_bsnip_matched.npz'
    f19 = output_path + 'sFC_adhd200_matched.npz'
    f20 = output_path + 'sFC_early_psychosis_matched.npz'
    f21 = output_path + 'sFC_22q_matched.npz'
    f22 = output_path + 'sFC_oasis_ad_matched.npz'
    f23 = output_path + 'sFC_pd_matched.npz'
    f24 = output_path + 'sFC_ucla_bipolar_matched.npz'
    f25 = output_path + 'sFC_ucla_schizophrenia_matched.npz'
    f26 = output_path + 'sFC_nki_gender_matched.npz'
    f27 = output_path + 'sFC_leipzig_gender_matched.npz'
    f28 = output_path + 'sFC_dev_gender_matched.npz'
    f29 = output_path + 'sFC_embarc_hamd_matched.npz'
    f30 = output_path + 'sFC_adhd_fukui_IIRV_matched.npz'
    f31 = output_path + 'sFC_pd_MDS_final.npz'
    f32 = output_path + 'sFC_ucla_schizophrenia_full_no_duplicates.npz'
    f33 = output_path + 'sFC_ucla_bipolar_full_no_duplicates.npz'
    f34 = output_path + 'sFC_leipzig_gender_full_visit1.npz'
    f35 = output_path + 'sFC_abide_asd_original_sample.npz'
    f36 = output_path + 'sFC_bearden_22q_original_sample.npz'
    f37 = output_path + 'sFC_pd_original_sample.npz'
    f38 = output_path + 'sFC_oasis_ad_original_sample.npz'
    f39 = output_path + 'sFC_stanford_autism_original_sample.npz'
    f40 = output_path + 'sFC_bsnip_original_sample.npz'
    f41 = output_path + 'sFC_nki_gender_original_sample.npz'
    f42 = output_path + 'sFC_adhd200_original_sample.npz'
    f43 = output_path + 'sFC_embarc_MDD_original_sample.npz'
    # data1 = np.load(f1)
    # data2 = np.load(f2)
    # data3 = np.load(f3)
    #
    # data_features = data1['features']
    # labels = data1['labels']
    # testData1_features = data2['features']
    # testData1_labels = data2['labels']
    # testData2_features = data3['features']
    # testData2_labels = data3['labels']


    #####################################
    K = 5
    # Perform classification and compute classification performance metrics
    #accuracy = np.zeros(K)
    #precision = np.zeros(K)
    #recall = np.zeros(K)
    #f1score = np.zeros(K)
    #test1_accuracy = np.zeros(K)
    #test1_precision = np.zeros(K)
    #test1_recall = np.zeros(K)
    #test1_f1score = np.zeros(K)
    #test2_accuracy = np.zeros(K)
    #test2_precision = np.zeros(K)
    #test2_recall = np.zeros(K)
    #test2_f1score = np.zeros(K)

    # prepare models
    models = []
    models.append(('linSVM', svm.SVC(kernel='linear')))
    models.append(('KNN', (neighbors.KNeighborsClassifier())))
    models.append(('DT', tree.DecisionTreeClassifier()))
    models.append(('LR', linear_model.LogisticRegression()))
    # models.append(('rbfSVM', svm.SVC(kernel='rbf')))
    models.append(('RC', linear_model.RidgeClassifier(alpha=0.5)))
    models.append(('LASSO', linear_model.LogisticRegression(penalty='l1', solver='liblinear')))
    # models.append(('ELNet', linear_model.LogisticRegression(penalty='elasticnet', l1_ratio=0.5, solver='saga')))
    models.append(('RF', ensemble.RandomForestClassifier()))

    
    #data_features, labels = get_features_labels(adhd_fukui_data, "adhd_fukui")
    data_features, labels = get_features_labels(pd_MDS_data, "pd_MDS")
    #data_features, labels = get_features_labels(embarc_data, "embarc")
    #testData1_features, testData1_labels = get_features_labels(testData1, 'hcp')
    #testData2_features, testData2_labels = get_features_labels(testData2, 'nkirs')
    np.savez(f31, features=data_features, labels=labels)
    #np.savez(f9, features=data_features, labels=labels)
    #np.savez(f5, features=data_features, labels=labels)
    #np.savez(f8, features=data_features, labels=labels)
    #np.savez(f2, features=testData1_features, labels=testData1_labels)
    #np.savez(f3, features=testData2_features, labels=testData2_labels)
    seed = 42
    repeated_acc = []
    repeated_f1 = []
    repeated_precision = []
    repeated_recall = []
    model_names = ['linSVM','KNN','DT','LR','RC','LASSO','RF']
    all_accuracies = {i: [] for i in model_names}
    all_f1s = {i: [] for i in model_names}
    all_precisions = {i: [] for i in model_names}
    all_recalls = {i: [] for i in model_names}
    for split in range(11,21):
        accuracy = np.zeros(K)
        precision = np.zeros(K)
        recall = np.zeros(K)
        f1score = np.zeros(K)
        split_seed = seed + split 
        kf = StratifiedShuffleSplit(n_splits=5, random_state=split_seed)
        train_index_list = []
        val_index_list = []
        for train_index, val_index in kf.split(data_features, labels):
            train_index_list.append(train_index)
            val_index_list.append(val_index)
        for name, model in models:
            print('*** Classifier: ', name)
            for foldid in range(K):
                print('** Evaluating: Fold {}'.format(foldid))
            
                fname_model = output_path + 'model_' + name + '_' + model_ss + '_' + str(foldid) + '_' + 'seed_' + str(split_seed) + '.sav'
                print('model name {}'.format(fname_model))

                idx = foldid
                train_index = train_index_list[idx]
                val_index = val_index_list[idx]
                x_train, x_val = data_features[train_index], data_features[val_index]
                y_train, y_val = labels[train_index], labels[val_index]
            
                modelfit = model.fit(x_train, y_train)
                y_val_predicted = modelfit.predict(x_val)

                # save the model
                pickle.dump(modelfit, open(fname_model, 'wb'))

                classifreport_dict = classification_report(y_val, y_val_predicted,
                                                       output_dict=True)
                accuracy[foldid] = classifreport_dict['accuracy'] * 100
                precision[foldid] = classifreport_dict['macro avg']['precision'] * 100
                recall[foldid] = classifreport_dict['macro avg']['recall'] * 100
                f1score[foldid] = classifreport_dict['macro avg']['f1-score'] * 100
            
        
            ''' 
            print('** Mean ADHD Fukui IIRV (training session) accuracy across 5 folds {0:.2f} +/- {1:.2f} %'.format(
                np.mean(accuracy), np.std(accuracy)))
            print('** Mean ADHD Fukui IIRV (training session) precision across 5 folds {0:.2f} +/- {1:.2f} %'.format(
                np.mean(precision), np.std(precision)))
            print('** Mean ADHD Fukui IIRV (training session) recall across 5 folds {0:.2f} +/- {1:.2f} %'.format(
                np.mean(recall), np.std(recall)))
            print('** Mean ADHD Fukui IIRV (training session) f1-score across 5 folds {0:.2f} +/- {1:.2f} %'.format(
                np.mean(f1score),np.std(f1score)))
            print( '** F1-scores across 5 folds **:',f1score)
            print( '** Accs across 5 folds **:',accuracy)
            all_accuracies[name].append(np.mean(accuracy))
            all_f1s[name].append(np.mean(f1score))
            all_precisions[name].append(np.mean(precision))
            all_recalls[name].append(np.mean(recall))
            '''

            
            print('** Mean PD MDS (training session) accuracy across 5 folds {0:.2f} +/- {1:.2f} %'.format(
                np.mean(accuracy), np.std(accuracy)))
            print('** Mean PD MDS (training session) precision across 5 folds {0:.2f} +/- {1:.2f} %'.format(
                np.mean(precision), np.std(precision)))
            print('** Mean PD MDS (training session) recall across 5 folds {0:.2f} +/- {1:.2f} %'.format(
                np.mean(recall), np.std(recall)))
            print('** Mean PD MDS (training session) f1-score across 5 folds {0:.2f} +/- {1:.2f} %'.format(
                np.mean(f1score),np.std(f1score)))
            print( '** F1-scores across 5 folds **:',f1score)
            print( '** Accs across 5 folds **:',accuracy)
            all_accuracies[name].append(np.mean(accuracy))
            all_f1s[name].append(np.mean(f1score))
            all_precisions[name].append(np.mean(precision))
            all_recalls[name].append(np.mean(recall)) 

            '''
            print('** Mean EMBARC HAMD (training session) accuracy across 5 folds {0:.2f} +/- {1:.2f} %'.format(
                np.mean(accuracy), np.std(accuracy)))
            print('** Mean EMBARC HAMD (training session) precision across 5 folds {0:.2f} +/- {1:.2f} %'.format(
                np.mean(precision), np.std(precision)))
            print('** Mean EMBARC HAMD (training session) recall across 5 folds {0:.2f} +/- {1:.2f} %'.format(
                np.mean(recall), np.std(recall)))
            print('** Mean EMBARC HAMD (training session) f1-score across 5 folds {0:.2f} +/- {1:.2f} %'.format(
                np.mean(f1score),np.std(f1score)))
            print( '** F1-scores across 5 folds **:',f1score)
            print( '** Accs across 5 folds **:',accuracy)
            repeated_acc.append(np.mean(accuracy))
            repeated_precision.append(np.mean(precision))
            repeated_recall.append(np.mean(precision))
            repeated_f1.append(np.mean(precision))
            '''
    mean_accuracy = {idx: np.mean(acc) for idx,acc in all_accuracies.items()}
    std_accuracy = {idx: np.std(acc) for idx,acc in all_accuracies.items()}
    mean_f1 = {idx: np.mean(f1) for idx,f1 in all_f1s.items()}
    std_f1 = {idx: np.std(f1) for idx,f1 in all_f1s.items()}
    mean_precision = {idx: np.mean(precision) for idx,precision in all_precisions.items()}
    std_precision = {idx: np.std(precision) for idx,precision in all_precisions.items()}
    mean_recall = {idx: np.mean(recall) for idx,recall in all_recalls.items()}
    std_recall = {idx: np.std(recall) for idx,recall in all_recalls.items()}
    pdb.set_trace()
