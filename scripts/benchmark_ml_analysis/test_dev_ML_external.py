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

def data_cleaning_abide_td_age(path_to_dataset):
    saved_data_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/abide_asd_td_dev_age/fold_0.bin'
    X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(saved_data_path)

    X_total = np.concatenate((X_train,X_valid))
    Y_total = np.concatenate((Y_train,Y_valid))
   
    return X_total, Y_total

def data_cleaning_stanford_td_age(path_to_dataset):
    saved_data_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/stanford_autism_age_TD_wIDS/fold_0.bin'
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
    elif site == 'cmihbn':
        data, labels = data_cleaning_cmi_age(path_to_dataset)
    elif site == 'abide_td':
        data, labels = data_cleaning_abide_td_age(path_to_dataset)
    elif site == 'stanford_td':
        data, labels = data_cleaning_stanford_td_age(path_to_dataset)

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
    
    model_ss = 'dev_age'

    output_path = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/models/dev/ml_models/'
    
    #####################################
    K = 5
    # Perform classification and compute classification performance metrics

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
   
    #### loop through datasets ###

    #data_features, labels = get_features_labels('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/models/cmihbn/ml_models/', "adhd200_td")
    #data_features, labels = get_features_labels('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/results/models/cmihbn/ml_models/', "stanford_autism")
    for name, model in regressors:
        #fname_figure = 
        all_folds_pred = []
        print('*** Regressor: ', name)
        for foldid in range(K):
            print('** Evaluating: Fold {}'.format(foldid))
            
            fname_model = output_path + 'model_' + name + '_' + model_ss + '_' + str(foldid) + '.sav'
            print('model name {}'.format(fname_model))

            x_total, y_total = data_features, labels 
            
            #### Set if condition if dataset is stanford_autism, code below deals with NaNs #### 
            '''
            indices_to_remove = []
            for i in range(len(x_total)):
                for j in range(len(x_total[i])):
                    if np.isnan(x_total[i][j]):
                        indices_to_remove.append(i)
                        break
                        #print("train NaN found at index ({},{})".format(i,j))
            x_total_new = np.asarray([np.asarray(i) for ind,i in enumerate(x_total) if ind not in indices_to_remove])
            y_total_new = []
            for index, element in enumerate(y_total):
                if index not in indices_to_remove:
                     y_total_new.append(element)
            #print(x_train_new.shape)
            #print(x_val_new.shape)
            #pdb.set_trace()
            
            x_total = x_total_new
            y_total = np.asarray(y_total_new)
            '''
            modelfit = joblib.load(fname_model) 
            y_total_predicted = modelfit.predict(x_total)
            all_folds_pred.append(y_total_predicted)

        ensemble_predictions = np.mean(all_folds_pred, axis=0)
        
        ##### Set bias correction to do for each external TD dataset, for ADHD and ASD, call data load function for TD samples in ADHD and ASD cohort, derive correction parameters and apply 
        BAG = (ensemble_predictions - y_total).reshape(-1,1)

        lin_model = LinearRegression().fit(y_total.reshape(-1,1),BAG)

        Offset = lin_model.coef_[0][0] * y_total + lin_model.intercept_[0]
        # save the model

        y_total_predicted = ensemble_predictions - Offset
        #classifreport_dict = classification_report(y_val, y_val_predicted,
                                                       #output_dict=True)
        r, p = stats.pearsonr(y_total,y_total_predicted)



        ### Print for all datasets 
        
        ''' 
        print('** NKI (training session) accuracy {0:.4f}'.format(
            r))
        print('** NKI (training session) p-value {0:.4f}'.format(
            p))
        '''

        ''' 
        print('** CMI TD (training session) accuracy {0:.4f}'.format(
            r))
        print('** CMI TD (training session) p-value {0:.4f}'.format(
            p))
        
        '''

        '''
        print('** ADHD200 TD (training session) accuracy {0:.4f}'.format(
            r ** 2))
        print('** ADHD200 TD (training session) p-value {0:.4f}'.format(
            p)) 
        '''

    
        ''' 
        print('** Stanford ASD (training session) accuracy {0:.4f}'.format(
            r ** 2))
        print('** Stanford ASD (training session) p-value {0:.4f}'.format(
            p))  
        '''


 
