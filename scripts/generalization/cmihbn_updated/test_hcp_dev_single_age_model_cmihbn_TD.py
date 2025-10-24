import sys
sys.path.append('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/')
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import pickle
from tqdm import tqdm
import pdb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from utility_functions import *
import scipy
from scipy.interpolate import interp1d
from itertools import chain
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import scipy.stats as spss

USE_CUDA = False
# ### Set the hyperparameters
hyper_parameters = {}
hyper_parameters['num_epochs'] = 500
hyper_parameters['batch_size'] = 32
hyper_parameters['learning_rate'] = 0.00097119581997096
hyper_parameters['briannectome'] = True

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

def test_model(x_valid,y_valid,scaler,hyper_parameters, fname_model,intercept,slope):
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
            outputs_corrected = (outputs - intercept) / slope
            loss = criterion(outputs, labels).item()
            test_corr_coeff = np.corrcoef(labels.cpu(),outputs.cpu())[0,1]
            pval = stats.pearsonr(labels.cpu(),outputs.cpu())
            predicted_ages = scaler.inverse_transform(outputs.cpu().numpy().reshape(-1,1))
            actual_ages = scaler.inverse_transform(labels.cpu().numpy().reshape(-1,1))
            print('Predicted:',predicted_ages)
            print('Actual:',actual_ages)
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
            #outputs_corrected = (outputs - intercept) / slope
            labels = torch.squeeze(labels)
        return np.squeeze(scaler.inverse_transform(labels.cpu().numpy().reshape(-1,1))), np.squeeze(scaler.inverse_transform(outputs.cpu().numpy().reshape(-1,1)))



corrs = []
pvals = []
maes = []
scalers = []
actual_ages = []
predicted_ages = []
hys_all = []
ids_all = []
coef_list = []
intercept_list = []

path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/cmihbn_age_TD/fold_0.bin'

X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(path)
X_total = np.concatenate((X_train,X_valid))
Y_total = np.concatenate((Y_train,Y_valid))

path_dev = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/hcp_dev_age_five_fold/fold_0.bin'
X_train_dev, X_valid_dev, Y_train_dev, Y_valid_dev = load_finetune_dataset(path_dev)

sc = StandardScaler()
sc.fit(Y_train_dev.reshape(-1,1))

Y_total = sc.transform(Y_total.reshape(-1,1))

hcp_dev_model = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/train_regression_models/dev/best_outer_fold_0_hcp_dev_model_2_27_24.pt'

X_total = reshapeData(X_total)
    
actual, predicted = test_model_getVals(X_total,Y_total,sc,hyper_parameters,hcp_dev_model)

BAG = (predicted - actual).reshape(-1,1)

lin_model = LinearRegression().fit(actual.reshape(-1,1),BAG)

Offset = lin_model.coef_[0][0] * actual + lin_model.intercept_[0]

coef_list.append(lin_model.coef_[0][0])
intercept_list.append(lin_model.intercept_[0])

predicted_corrected = predicted - Offset

actual_ages = np.concatenate((actual_ages,actual))
predicted_ages = np.concatenate((predicted_ages,predicted_corrected))

print(actual_ages)
print(predicted_ages)
print(np.mean(np.abs(predicted_ages-actual_ages)))
print(np.mean(predicted_ages-actual_ages))

r,p = scipy.stats.pearsonr(x=actual_ages,y=predicted_ages)
print(r)
print(p)

#r,p = scipy.stats.pearsonr(actual_ages,predicted_ages-actual_ages)
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
ax.set_title("CMI-HBN",fontsize=16)
plt.savefig('cmihbn_td_brain_age_scatter_test_bias_correction_single_model.png',format='png')


pdb.set_trace()

actual_ages = np.squeeze(actual_ages)
predicted_ages = np.squeeze(predicted_ages)
print(np.mean(np.abs(predicted_ages-actual_ages)))

pdb.set_trace()

np.savez('nki_predicted_ages',predicted=predicted_ages)
np.savez('nki_actual_ages',actual=actual_ages)
pdb.set_trace()
#genders_all = list(chain(*genders_all))

#print(len(ids_all))
#print(len(hys_all))
#print(len(genders_all))
#print(actual_ages.shape)
#print(predicted_ages.shape)

total_frame = pd.DataFrame({'subject_id':ids_all,'Chronological Age':actual_ages,'Brain Age':predicted_ages,'BAG':predicted_ages-actual_ages,'HY':hys_all})
print(total_frame)
pdb.set_trace()
corr, pvalue = spss.spearmanr(total_frame['BAG'], total_frame['HY'])
print(corr)
print(pvalue)
