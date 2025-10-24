import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing, linear_model, model_selection, svm, tree, ensemble, neighbors
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.preprocessing import LabelEncoder
import pdb
import scipy
from scipy import stats
import joblib

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


def data_cleaning_nkirs_age(path_to_dataset):
    saved_data_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/nki_age_cog_dev_wIDs/fold_0.bin'
    X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(saved_data_path)

    X_total = np.concatenate((X_train,X_valid))
    Y_total = np.concatenate((Y_train,Y_valid))

    return X_total, Y_total


def data_cleaning_hcp_dev_age(path_to_dataset):
    saved_data_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/hcp_dev_age_five_fold/fold_0.bin'
    X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(saved_data_path)

    X_total = np.concatenate((X_train,X_valid))
    Y_total = np.concatenate((Y_train,Y_valid))

    return X_total, Y_total


def data_cleaning_adhd200_td_age(path_to_dataset):
    saved_data_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/adhd200_regression_age_TD_wIDs/fold_0.bin'
    X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(saved_data_path)

    X_total = np.concatenate((X_train,X_valid))
    Y_total = np.concatenate((Y_train,Y_valid))

    return X_total, Y_total

def data_cleaning_cmi_td_age(path_to_dataset):
    saved_data_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/cmihbn_age_TD/fold_0.bin'
    X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(saved_data_path)

    X_total = np.concatenate((X_train,X_valid))
    Y_total = np.concatenate((Y_train,Y_valid))

    return X_total, Y_total


def data_cleaning_abide_asd_age(path_to_dataset):
    saved_data_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/abide_asd_asd_dev_age_wIDs/fold_0.bin'
    X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(saved_data_path)
    
    X_total = np.concatenate((X_train,X_valid))
    Y_total = np.concatenate((Y_train,Y_valid))

    return X_total, Y_total


def data_cleaning_stanford_autism_age(path_to_dataset):
    saved_data_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/stanford_autism_age_wIDS/fold_0.bin'
    X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(saved_data_path)

    X_total = np.concatenate((X_train,X_valid))
    Y_total = np.concatenate((Y_train,Y_valid))
 
    return X_total, Y_total


def data_cleaning_adhd200_age(path_to_dataset):
    saved_data_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/adhd200_regression_age_ADHD_wIDs/fold_0.bin'

    X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(saved_data_path)

    X_total = np.concatenate((X_train,X_valid))
    Y_total = np.concatenate((Y_train,Y_valid))

    return X_total, Y_total

def data_cleaning_cmi_age(path_to_dataset):
    saved_data_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/adhd_cmihbn_age_ADHD_wIDS/fold_0.bin'

    X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(saved_data_path)

    X_total = np.concatenate((X_train,X_valid))
    Y_total = np.concatenate((Y_train,Y_valid))

    return X_total, Y_total


def get_features_labels(path_to_dataset, site):
    if site == 'nkirs':
        data, labels = data_cleaning_nkirs_age(path_to_dataset)
    elif site == 'abide_asd':
        data, labels = data_cleaning_abide_asd_age(path_to_dataset)
    elif site == 'stanford_autism':
        data, labels = data_cleaning_stanford_autism_age(path_to_dataset)
    elif site == 'adhd200':
        data, labels = data_cleaning_adhd200_age(path_to_dataset)
    elif site == 'hcp_dev':
        data, labels = data_cleaning_hcp_dev_age(path_to_dataset)
    elif site == 'adhd200_td':
        data, labels = data_cleaning_adhd200_td_age(path_to_dataset)
    elif site == 'cmihbn_td':
        data, labels = data_cleaning_cmi_td_age(path_to_dataset)
 
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
    
    model_ss = 'nki_age'

    output_path = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/models/dev/ml_models/'
    
    f1 = output_path + 'sFC_abide_asd_matched.npz'
    f3 = output_path + 'sFC_adhd200.npz'
    f10 = output_path + 'sFC_hcp_dev_age.npz'
    f17 = output_path + 'sFC_stanford_autism_matched.npz'
    f26 = output_path + 'sFC_nki_cog_dev_age.npz'
    f27 = output_path + 'sFC_cmi_td_age.npz'
    f39 = output_path + 'sFC_stanford_autism_original_sample.npz'
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
    accuracy = np.zeros(K)
    pvals = np.zeros(K)

    '''
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
    '''

    regressors = []
    # linear SVM classifier → linear SVR
    regressors.append(('linSVR', svm.SVR(kernel='linear')))
    # KNN classifier → KNN regressor
    regressors.append(('KNN', neighbors.KNeighborsRegressor()))
    # Decision tree classifier → decision tree regressor
    regressors.append(('DT', tree.DecisionTreeRegressor()))
    # logistic regression (classification) → ordinary least squares regression
    regressors.append(('LR', linear_model.LinearRegression()))
    # ridge classifier → ridge regression
    regressors.append(('RC', linear_model.Ridge(alpha=0.5)))
    # L1-penalized logistic → LASSO regression
    regressors.append(('LASSO', linear_model.Lasso(alpha=0.1)))
    # regressors.append(('ELNet', linear_model.ElasticNet(alpha=1.0, l1_ratio=0.5)))
    # random forest classifier → random forest regressor
    regressors.append(('RF', ensemble.RandomForestRegressor()))
    # get features and labels (0: male, 1: female)
    
    #data_features, labels = get_features_labels(hcp_data, 'hcp')
    #data_features, labels = get_features_labels(nki_data, "nkirs")
    data_features, labels = get_features_labels('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/models/cmihbn/ml_models/', "cmihbn_td")
    #data_features, labels = get_features_labels(leipzig_gender_data, "leipzig_gender")
    #data_features, labels = get_features_labels(ucla_bipolar_data, "ucla_bipolar")
    #data_features, labels = get_features_labels(adhd_fukui_data, "adhd_fukui")
    #data_features, labels = get_features_labels(pd_MDS_data, "pd_MDS")
    #data_features, labels = get_features_labels(embarc_data, "embarc")
    #data_features, labels = get_features_labels(abide_asd_data, 'abide_asd')
    #data_features, labels = get_features_labels(bsnip_data, 'bsnip')
    #data_features, labels = get_features_labels(adhd200_data, 'adhd200')
    #data_features, labels = get_features_labels(hcp_early_psychosis_data, 'hcp_early_psychosis')
    #data_features, labels = get_features_labels(bearden_22q_data, '22q')
    #data_features, labels = get_features_labels(oasis_ad_data, 'oasis_ad')
    #data_features, labels = get_features_labels(pd_data, 'pd')
    #data_features, labels = get_features_labels(ucla_schizophrenia_data,'ucla_schizophrenia')
    #data_features, labels = get_features_labels(stanford_autism_data, 'stanford_autism')
    #testData1_features, testData1_labels = get_features_labels(testData1, 'hcp')
    #testData2_features, testData2_labels = get_features_labels(testData2, 'nkirs')
    np.savez(f26, features=data_features, labels=labels)
    #np.savez(f9, features=data_features, labels=labels)
    #np.savez(f5, features=data_features, labels=labels)
    #np.savez(f8, features=data_features, labels=labels)
    #np.savez(f2, features=testData1_features, labels=testData1_labels)
    #np.savez(f3, features=testData2_features, labels=testData2_labels)
    
    kf = KFold(n_splits=5, random_state=6666, shuffle=True)
    train_index_list = []
    val_index_list = []
    for train_index, val_index in kf.split(data_features, labels):
        train_index_list.append(train_index)
        val_index_list.append(val_index)
    for name, model in regressors:
        #fname_figure = 
        print('*** Regressor: ', name)
        for foldid in range(K):
            print('** Evaluating: Fold {}'.format(foldid))
            
            fname_model = output_path + 'model_' + name + '_' + model_ss + '_' + str(foldid) + '.sav'
            print('model name {}'.format(fname_model))

            idx = foldid
            train_index = train_index_list[idx]
            val_index = val_index_list[idx]
            x_train, x_val = data_features[train_index], data_features[val_index]
            y_train, y_val = labels[train_index], labels[val_index]
            
            ####### Uncomment this block below for Stanford Autism only, because it says there are NaNs in the data and this block removes them
            
            ''' 
            train_indices_to_remove = []
            val_indices_to_remove = []
            for i in range(len(x_train)):
                for j in range(len(x_train[i])):
                    if np.isnan(x_train[i][j]):
                        train_indices_to_remove.append(i)
                        break
                        #print("train NaN found at index ({},{})".format(i,j))
            for i in range(len(x_val)):
                for j in range(len(x_val[i])):
                    if np.isnan(x_val[i][j]):
                        val_indices_to_remove.append(i)
                        break
                        #print("val NaN found at index ({},{})".format(i,j))
            #print(train_indices_to_remove)
            #print(val_indices_to_remove)
            x_train_new = np.asarray([np.asarray(i) for ind,i in enumerate(x_train) if ind not in train_indices_to_remove])
            x_val_new = np.asarray([np.asarray(i) for ind,i in enumerate(x_val) if ind not in val_indices_to_remove])
            y_train_new = []
            y_val_new = []
            for index, element in enumerate(y_train):
                if index not in train_indices_to_remove:
                    y_train_new.append(element)
            for index, element in enumerate(y_val):
                if index not in val_indices_to_remove:
                     y_val_new.append(element)
            #print(x_train_new.shape)
            #print(x_val_new.shape)
            #pdb.set_trace()
            '''
            
            #x_train = x_train_new
            #x_val = x_val_new
            #y_train = y_train_new
            #y_val = y_val_new
            
            modelfit = model.fit(x_train, y_train)
            y_val_predicted = modelfit.predict(x_val)

            BAG = (y_val_predicted - y_val).reshape(-1,1)

            lin_model = LinearRegression().fit(y_val.reshape(-1,1),BAG)

            Offset = lin_model.coef_[0][0] * y_val + lin_model.intercept_[0]
            # save the model
            pickle.dump(modelfit, open(fname_model, 'wb'))

            y_val_predicted = y_val_predicted - Offset
            #classifreport_dict = classification_report(y_val, y_val_predicted,
                                                       #output_dict=True)
            r, p = stats.pearsonr(y_val,y_val_predicted)
            accuracy[foldid] = r
            pvals[foldid] = p
        



        ##### Performance on validation set, uncomment block that corresponds to data you are working with
        
        ''' 
        fname_figure = f'/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/results/dnn/figures/finetuning_hcp_gender/ml_models/{name}_hcp_gender_final_manuscript_fig_results_ml_methods' 
        print('** Mean HCP (training session) accuracy across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(accuracy), np.std(accuracy)))
        print('** Mean HCP (training session) precision across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(precision), np.std(precision)))
        print('** Mean HCP (training session) recall across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(recall), np.std(recall)))
        print('** Mean HCP (training session) f1-score across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(f1score),np.std(f1score)))
        print( '** F1-scores across 5 folds **:',f1score)
        print( '** Accs across 5 folds **:',accuracy)
        np.savez(fname_figure,valid_accs=accuracy,valid_f1s=f1score,valid_precisions=precision,valid_recalls=recall)
        '''
        
        ''' 
        print('** Mean NKI (training session) accuracy across 5 folds {0:.4f} +/- {1:.4f} %'.format(
            np.mean(accuracy), np.std(accuracy)))
        print( '** Accs across 5 folds **:',accuracy)
        print('** Mean NKI (training session) p-value across 5 folds {0:.4f} +/- {1:.4f} %'.format(
            np.mean(pvals), np.std(pvals)))
        print( '** P-values across 5 folds **:',accuracy)
        '''
        
        '''  
        fname_figure = f'/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/results/dnn/figures/finetuning_dev/ml_models/{name}_dev_gender_final_yuan_list_manuscript_fig_results_ml_methods' 
        print('** Mean HCP-Dev (training session) accuracy across 5 folds {0:.4f} +/- {1:.4f} %'.format(
            np.mean(accuracy), np.std(accuracy)))
        print( '** Accs across 5 folds **:',accuracy)
        print('** Mean HCP-Dev (training session) p-value across 5 folds {0:.4f} +/- {1:.4f} %'.format(
            np.mean(pvals), np.std(pvals)))
        print( '** P-values across 5 folds **:',accuracy)
        np.savez(fname_figure,valid_accs=accuracy,valid_pvals=pvals)
        '''
        
        
        print('** Mean CMI TD (training session) accuracy across 5 folds {0:.4f} +/- {1:.4f} %'.format(
            np.mean(accuracy), np.std(accuracy)))
        print( '** Accs across 5 folds **:',accuracy)
        print('** Mean CMI TD (training session) p-value across 5 folds {0:.4f} +/- {1:.4f} %'.format(
            np.mean(pvals), np.std(pvals)))
        print( '** P-values across 5 folds **:',accuracy)
        
        
        ''' 
        fname_figure = f'/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/results/dnn/figures/finetuning_abide_asd/ml_models/{name}_abide_asd_final_yuan_list_noNaNs_manuscript_fig_results_ml_methods' 
        print('** Mean ABIDE ASD (training session) accuracy across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(accuracy), np.std(accuracy)))
        print('** Mean ABIDE ASD (training session) precision across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(precision), np.std(precision)))
        print('** Mean ABIDE ASD (training session) recall across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(recall), np.std(recall)))
        print('** Mean ABIDE ASD (training session) f1-score across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(f1score),np.std(f1score)))
        print( '** F1-scores across 5 folds **:',f1score)
        print( '** Accs across 5 folds **:',accuracy)
        np.savez(fname_figure,valid_accs=accuracy,valid_f1s=f1score,valid_precisions=precision,valid_recalls=recall)
        '''

        ''' 
        fname_figure = f'/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/results/dnn/figures/finetuning_stanford_autism/ml_models/{name}_stanford_autism_original_sample_manuscript_SI_fig_results_ml_methods'
        print('** Mean Stanford Autism (training session) accuracy across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(accuracy), np.std(accuracy)))
        print('** Mean Stanford Autism (training session) precision across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(precision), np.std(precision)))
        print('** Mean Stanford Autism (training session) recall across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(recall), np.std(recall)))
        print('** Mean Stanford Autism (training session) f1-score across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(f1score),np.std(f1score)))
        print( '** F1-scores across 5 folds **:',f1score)
        print( '** Accs across 5 folds **:',accuracy)
        np.savez(fname_figure,valid_accs=accuracy,valid_f1s=f1score,valid_precisions=precision,valid_recalls=recall)
        '''

        ''' 
        fname_figure = f'/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/results/dnn/figures/finetuning_bsnip/ml_models/{name}_bsnip_original_sample_manuscript_SI_fig_results_ml_methods'

        print('** Mean BSNIP (training session) accuracy across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(accuracy), np.std(accuracy)))
        print('** Mean BSNIP (training session) precision across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(precision), np.std(precision)))
        print('** Mean BSNIP (training session) recall across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(recall), np.std(recall)))
        print('** Mean BSNIP (training session) f1-score across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(f1score),np.std(f1score)))
        print( '** F1-scores across 5 folds **:',f1score)
        print( '** Accs across 5 folds **:',accuracy)
        np.savez(fname_figure,valid_accs=accuracy,valid_f1s=f1score,valid_precisions=precision,valid_recalls=recall)
        '''

        ''' 
        fname_figure = f'/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/results/dnn/figures/finetuning_adhd200/ml_models/{name}_adhd200_original_sample_manuscript_SI_fig_results_ml_methods'
        print('** Mean ADHD200 (training session) accuracy across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(accuracy), np.std(accuracy)))
        print('** Mean ADHD200 (training session) precision across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(precision), np.std(precision)))
        print('** Mean ADHD200 (training session) recall across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(recall), np.std(recall)))
        print('** Mean ADHD200 (training session) f1-score across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(f1score),np.std(f1score)))
        print( '** F1-scores across 5 folds **:',f1score)
        print( '** Accs across 5 folds **:',accuracy)
        np.savez(fname_figure,valid_accs=accuracy,valid_f1s=f1score,valid_precisions=precision,valid_recalls=recall)
        '''

        ''' 
        fname_figure = f'/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/results/dnn/figures/finetuning_early_psychosis/ml_models/{name}_hcp_early_psychosis_original_sample_manuscript_fig_results_ml_methods'
        print('** Mean Early Psychosis (training session) accuracy across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(accuracy), np.std(accuracy)))
        print('** Mean Early Psychosis (training session) precision across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(precision), np.std(precision)))
        print('** Mean Early Psychosis (training session) recall across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(recall), np.std(recall)))
        print('** Mean Early Psychosis (training session) f1-score across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(f1score),np.std(f1score)))
        print( '** F1-scores across 5 folds **:',f1score)
        print( '** Accs across 5 folds **:',accuracy)
        np.savez(fname_figure,valid_accs=accuracy,valid_f1s=f1score,valid_precisions=precision,valid_recalls=recall)
        '''

        ''' 
        fname_figure = f'/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/results/dnn/figures/finetuning_22q/ml_models/{name}_22q_original_sample_manuscript_SI_fig_results_ml_methods'
        print('** Mean 22q (training session) accuracy across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(accuracy), np.std(accuracy)))
        print('** Mean 22q (training session) precision across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(precision), np.std(precision)))
        print('** Mean 22q (training session) recall across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(recall), np.std(recall)))
        print('** Mean 22q (training session) f1-score across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(f1score),np.std(f1score)))
        print( '** F1-scores across 5 folds **:',f1score)
        print( '** Accs across 5 folds **:',accuracy)
        np.savez(fname_figure,valid_accs=accuracy,valid_f1s=f1score,valid_precisions=precision,valid_recalls=recall)
        '''

        '''         
        fname_figure = f'/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/results/dnn/figures/finetuning_oasis_ad/ml_models/{name}_ad_original_sample_manuscript_SI_fig_results_ml_methods'
        print('** Mean oasis ad (training session) accuracy across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(accuracy), np.std(accuracy)))
        print('** Mean oasis ad (training session) precision across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(precision), np.std(precision)))
        print('** Mean oasis ad (training session) recall across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(recall), np.std(recall)))
        print('** Mean oasis ad (training session) f1-score across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(f1score),np.std(f1score)))
        print( '** F1-scores across 5 folds **:',f1score)
        print( '** Accs across 5 folds **:',accuracy)
        np.savez(fname_figure,valid_accs=accuracy,valid_f1s=f1score,valid_precisions=precision,valid_recalls=recall)
        '''

        ''' 
        fname_figure = f'/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/results/dnn/figures/finetuning_pd/ml_models/{name}_pd_original_sample_manuscript_SI_fig_results_ml_methods'
        print('** Mean pd (training session) accuracy across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(accuracy), np.std(accuracy)))
        print('** Mean pd (training session) precision across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(precision), np.std(precision)))
        print('** Mean pd (training session) recall across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(recall), np.std(recall)))
        print('** Mean pd (training session) f1-score across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(f1score),np.std(f1score)))
        print( '** F1-scores across 5 folds **:',f1score)
        print( '** Accs across 5 folds **:',accuracy)
        np.savez(fname_figure,valid_accs=accuracy,valid_f1s=f1score,valid_precisions=precision,valid_recalls=recall)
        '''

        '''
        fname_figure = f'/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/results/dnn/figures/finetuning_ucla/ml_models/{name}_bipolar_full_list_no_duplicates_manuscript_SI_fig_results_ml_methods'
        print('** Mean bipolar (training session) accuracy across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(accuracy), np.std(accuracy)))
        print('** Mean bipolar (training session) precision across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(precision), np.std(precision)))
        print('** Mean bipolar (training session) recall across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(recall), np.std(recall)))
        print('** Mean bipolar (training session) f1-score across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(f1score),np.std(f1score)))
        print( '** F1-scores across 5 folds **:',f1score)
        print( '** Accs across 5 folds **:',accuracy)
        np.savez(fname_figure,valid_accs=accuracy,valid_f1s=f1score,valid_precisions=precision,valid_recalls=recall) 
        '''

        '''
        fname_figure = f'/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/results/dnn/figures/finetuning_ucla/ml_models/{name}_ucla_schizophrenia_full_list_no_duplicates_manuscript_SI_fig_results_ml_methods'
        print('** Mean ucla schizophrenia (training session) accuracy across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(accuracy), np.std(accuracy)))
        print('** Mean ucla schizophrenia (training session) precision across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(precision), np.std(precision)))
        print('** Mean ucla schizophrenia (training session) recall across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(recall), np.std(recall)))
        print('** Mean ucla schizophrenia (training session) f1-score across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(f1score),np.std(f1score)))
        print( '** F1-scores across 5 folds **:',f1score)
        print( '** Accs across 5 folds **:',accuracy)
        np.savez(fname_figure,valid_accs=accuracy,valid_f1s=f1score,valid_precisions=precision,valid_recalls=recall)
        '''
        
        ''' 
        fname_figure = f'/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/results/dnn/figures/finetuning_adhd_fukui/ml_models/{name}_adhd_fukui_IIRV_matched_final_yuan_list_manuscript_fig_results_ml_methods'
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
        np.savez(fname_figure,valid_accs=accuracy,valid_f1s=f1score,valid_precisions=precision,valid_recalls=recall)
        '''

        '''
        fname_figure = f'/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/results/dnn/figures/finetuning_pd/ml_models/{name}_pd_MDS_final_manuscript_fig_results_ml_methods'
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
        np.savez(fname_figure,valid_accs=accuracy,valid_f1s=f1score,valid_precisions=precision,valid_recalls=recall)
        '''

        ''' 
        fname_figure = f'/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/results/dnn/figures/finetuning_embarc/ml_models/{name}_embarc_hamd_original_sample_manuscript_SI_fig_results_ml_methods'
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
        np.savez(fname_figure,valid_accs=accuracy,valid_f1s=f1score,valid_precisions=precision,valid_recalls=recall)
        '''       
 
