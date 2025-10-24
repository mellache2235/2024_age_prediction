import sys
sys.path.append('/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/train_regression_models/')
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import pickle
from tqdm import tqdm
import pdb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utility_functions import *
import scipy
from scipy.interpolate import interp1d
from itertools import chain
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from scipy import signal

USE_CUDA = False
# ### Set the hyperparameters
hyper_parameters = {}
hyper_parameters['num_epochs'] = 500
hyper_parameters['batch_size'] = 32
hyper_parameters['learning_rate'] = 0.00097119581997096
hyper_parameters['briannectome'] = True

def standardize_timeseries(data):
    '''Remove mean and standard deviation from each timeseries'''
    data_standard = np.empty((0, data.shape[1], data.shape[2]))
    for subj in np.arange(data.shape[0]):
        x = np.squeeze(data[subj,:,:])
        sum_x = np.abs(np.sum(x))
        if sum_x > 0:
            print("Subject = ",subj)
            data_subj = np.zeros((data.shape[1],data.shape[2]))
            for channel in np.arange(data.shape[2]):
                ts = data[subj,:,channel]
                ts = np.squeeze(ts)
                mean_ts = np.mean(ts[ts != 0])
                std_ts = np.std(ts[ts != 0])
                if std_ts != 0:
                    ts = (ts - mean_ts)/std_ts
                data_subj[:,channel] = ts
            data_subj = np.expand_dims(data_subj,axis=0)
            data_standard = np.concatenate((data_standard, data_subj))
                # data[subj,:,channel] = ts
    return data_standard[:,:,:]

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(246, 32, kernel_size=5, stride=1, bias=False),
            nn.BatchNorm1d(32),
            nn.PReLU(32),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=7, stride=1, bias=False),
            nn.BatchNorm1d(32),
            nn.PReLU(32),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.drop_out = nn.Dropout(p=0.4561228015061742)
        self.regressor = nn.Linear(32, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.drop_out(out)
        out = self.layer2(out)
        out = self.drop_out(out)
        out = out.mean(axis=2)
        out = self.regressor(out)
        return out

def test_model(x_valid,y_valid,scaler,hyper_parameters, fname_model):
    criterion = nn.MSELoss()
    input_tensor_valid = torch.from_numpy(x_valid).type(torch.FloatTensor)
    label_tensor_valid = torch.from_numpy(y_valid).type(torch.FloatTensor)
    dataset_valid = TensorDataset(input_tensor_valid, label_tensor_valid)
    valid_loader = DataLoader(dataset=dataset_valid, batch_size=x_valid.shape[0], shuffle=False)
    model = ConvNet()
    model.load_state_dict(torch.load(fname_model,map_location=torch.device('cpu')))

    model.eval()
    with torch.no_grad():
        for images, labels in valid_loader:
            outputs = model(images)
            outputs = torch.squeeze(outputs)
            labels = torch.squeeze(labels)
            loss = criterion(outputs, labels).item()
            test_corr_coeff = np.corrcoef(labels.cpu(),outputs.cpu())[0,1]
            pval = stats.pearsonr(labels.cpu(),outputs.cpu())
            test_mae = np.mean(np.abs(scaler.inverse_transform(outputs.cpu().numpy().reshape(-1,1)) - scaler.inverse_transform(labels.cpu().numpy().reshape(-1,1)))) 
    return test_corr_coeff, test_mae, pval

def test_model_getVals(x_valid,y_valid,scaler,hyper_parameters, fname_model):
    criterion = nn.MSELoss()
    input_tensor_valid = torch.from_numpy(x_valid).type(torch.FloatTensor)
    label_tensor_valid = torch.from_numpy(y_valid)
    dataset_valid = TensorDataset(input_tensor_valid, label_tensor_valid)
    valid_loader = DataLoader(dataset=dataset_valid, batch_size=x_valid.shape[0], shuffle=False)
    model = ConvNet()
    model.load_state_dict(torch.load(fname_model,map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        for images, labels in valid_loader:
            outputs = model(images)
            outputs = torch.squeeze(outputs)
            labels = torch.squeeze(labels)
        return np.squeeze(scaler.inverse_transform(labels.cpu().numpy().reshape(-1,1))), np.squeeze(scaler.inverse_transform(outputs.cpu().numpy().reshape(-1,1)))


corrs = []
pvals = []
maes = []
scalers = []
actual_ages = []
predicted_ages = []
hys = []
ids = []
genders_all = []

lin_params = np.load('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/nki_updated/lin_model_params_nki_5folds.npz')

coefs = lin_params['coef']
intercepts = lin_params['intercept']

for fold in range(5):

    path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/stanford_autism_age_wIDS/fold_%d.bin'%(fold)
 
    X_train, X_valid, id_train, Y_train, Y_valid, id_test = load_finetune_dataset_wids(path)
   
    epsilon = 1e-8
 
    mean_per_subject_region = np.mean(X_train, axis=1, keepdims=True)
    std_per_subject_region = np.std(X_train, axis=1, keepdims=True)
    X_train = (X_train - mean_per_subject_region) / (std_per_subject_region + epsilon)

    mean_per_subject_region = np.mean(X_valid, axis=1, keepdims=True)
    std_per_subject_region = np.std(X_valid, axis=1, keepdims=True)
    X_valid = (X_valid - mean_per_subject_region) / (std_per_subject_region + epsilon)

    path_dev = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/hcp_dev_age_five_fold/fold_%d.bin'%(fold)
    X_train_dev, X_valid_dev, Y_train_dev, Y_valid_dev = load_finetune_dataset(path_dev)
    
    hcp_dev_model = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/train_regression_models/dev/best_outer_fold_%d_hcp_dev_model_2_27_24.pt'%(fold)

    sc = StandardScaler()
    sc.fit(Y_train_dev.reshape(-1,1))

    Y_train_dev = sc.transform(Y_train_dev.reshape(-1,1))
    Y_valid = np.squeeze(sc.transform(Y_valid.reshape(-1,1)))
 
    X_valid_resampled = signal.resample(X_valid,math.floor(X_valid.shape[1] * 0.49 / 0.8),axis=1)

    X_train_dev = reshapeData(X_train_dev)
    X_valid = reshapeData(X_valid_resampled)
    
 
    #actual_train, predicted_train = test_model_getVals(X_train_dev,Y_train_dev,sc,hyper_parameters,hcp_dev_model)

    actual, predicted = test_model_getVals(X_valid,Y_valid,sc,hyper_parameters,hcp_dev_model)

    #BAG = (predicted_train - actual_train).reshape(-1,1)

    #lin_model = LinearRegression().fit(actual_train.reshape(-1,1),BAG)

    Offset = coefs[0] * actual + intercepts[0]

    predicted_corrected = predicted - Offset

    actual_ages = np.concatenate((actual_ages,actual))
    predicted_ages = np.concatenate((predicted_ages,predicted_corrected))
    
    id_test = id_test.astype('str')
    #print(id_test)
    ids.append(id_test)
    #mask = np.isin(adhd_ids,id_test)
    #hy = hyp_scores[mask]
    #gender = label[mask]
    #hys.append(hy)
    #genders_all.append(gender)

#ids = list(chain(*ids))
#hys = list(chain(*hys))

actual_ages = np.squeeze(actual_ages)
predicted_ages = np.squeeze(predicted_ages)

print(actual_ages)
print(predicted_ages)

print(np.mean(np.abs(predicted_ages - actual_ages)))

r,p = scipy.stats.pearsonr(x=actual_ages,y=predicted_ages)

print(r)
print(p)


#r,p = scipy.stats.pearsonr(x=actual_ages,y=predicted_ages-actual_ages)

#print(r)
#print(p)

r_squared = r ** 2

fig,ax = plt.subplots()
sns.regplot(x=actual_ages, y=predicted_ages, ci=None,
           scatter_kws={'color': 'blue', 's': 50},
           line_kws={'color': 'red'},ax=ax)
ax.text(0.95, 0.05, f"$r^2 = {r_squared:.3f}$\n$p = {p:.2e}$",
         transform=ax.transAxes,  # Use axis coordinates (0 to 1)
         horizontalalignment='right',  # Align text to the right at x=0.95
         verticalalignment='bottom',   # Align text to the bottom at y=0.05
         fontsize=12)

ax.spines[['right', 'top']].set_visible(False)
ax.set_xlabel("Chronological Age",fontsize=16)
ax.set_ylabel("Brain Age",fontsize=16)
ax.set_title("Stanford ASD",fontsize=16)
plt.savefig('stanford_asd_brain_age_scatter_test_bias_correction.png',format='png')

pdb.set_trace()
#np.savez('stanford_asd_predicted_ages',predicted=predicted_ages)
#np.savez('stanford_asd_actual_ages',actual=actual_ages)
#pdb.set_trace()


total_frame = pd.DataFrame({'subject_id':ids,'Chronological Age':actual_ages,'Brain Age':predicted_ages,'BAG':predicted_ages-actual_ages})
print(total_frame)

data_dir_SRS = "/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/prepare_data/stanford_autism/"
SRS_file = pd.read_csv(data_dir_SRS + 'SRS_data_20230925.csv',skiprows=[0])
SRS_file = SRS_file.drop_duplicates(subset=['record_id'],keep='last')
SRS_file['record_id'] = SRS_file['record_id'].astype('str')
ids_2 = np.asarray(SRS_file['record_id'])
print(SRS_file)

intersection = np.asarray(np.intersect1d(ids, ids_2))
total_frame_sub = total_frame.loc[total_frame['subject_id'].isin(intersection),:]
print(total_frame_sub)
scores = np.asarray(SRS_file.loc[SRS_file['record_id'].isin(total_frame_sub['subject_id']),'srs_awr_standard'])
total_frame_sub['SRS'] = scores
print(total_frame_sub)
pdb.set_trace()

#total_frame = pd.DataFrame({'subject_id':ids,'Chronological Age':actual_ages,'Brain Age':predicted_ages,'Gender':genders_all,'Hyperactivity':hys})

total_frame = total_frame.dropna()
ix = total_frame['Hyperactivity'] == -999.
total_frame = total_frame[~ix]

total_frame['BAG'] = total_frame['Brain Age'] - total_frame['Chronological Age']

test_corr_coeff = np.corrcoef(total_frame['BAG'],total_frame['Hyperactivity'])[0,1]
pval = stats.pearsonr(total_frame['BAG'],total_frame['Hyperactivity'])
print(test_corr_coeff)
print(pval)

data = pd.DataFrame({'BAG':total_frame['BAG'],'Gender':total_frame['Gender'],'ChronologicalAge':total_frame['Chronological Age']})
model = smf.ols('BAG ~ ChronologicalAge * Gender', data=data).fit()
print(model.summary())
'''
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
interaction_terms = poly.fit_transform(total_frame[['Chronological Age', 'Gender']])
interaction_df = pd.DataFrame(interaction_terms, columns=['Chronological Age', 'Gender', 'Chronological Age*Gender'])

# Fit the linear regression model
X = interaction_df[['Chronological Age', 'Gender', 'Chronological Age*Gender']]
y = total_frame['BAG']

model = LinearRegression().fit(X, y)

# Display coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
'''
print("Mean Correlation:",np.mean(corrs))
print("Mean MAE:",np.mean(maes))
