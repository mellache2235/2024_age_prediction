import sys
sys.path.append('/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/train_regression_models/')
import os
from os import listdir
from os.path import isfile, join
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
import scipy.stats as spss
from scipy.interpolate import interp1d
from itertools import chain
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

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

def test_model(x_valid,y_valid,scaler,hyper_parameters, fname_model):
    criterion = nn.MSELoss()
    input_tensor_valid = torch.from_numpy(x_valid).type(torch.FloatTensor)
    label_tensor_valid = torch.from_numpy(y_valid).type(torch.FloatTensor)
    dataset_valid = TensorDataset(input_tensor_valid, label_tensor_valid)
    valid_loader = DataLoader(dataset=dataset_valid, batch_size=x_valid.shape[0], shuffle=False)
    #model = CovnetRegressor_brainnectome_wEmbedder(hyper_parameters['fname_masked_model'],False)
    model = ConvNet()
    model.load_state_dict(torch.load(fname_model,map_location=torch.device('cpu')))

    model.eval()
    with torch.no_grad():
        for images, labels in valid_loader:
            outputs = model(images)
            outputs = torch.squeeze(outputs)
            labels = torch.squeeze(labels)
            #print(outputs)
            #print(labels)
            #print(outputs.shape)
            #print(labels.shape)
            loss = criterion(outputs, labels).item()
            #print(loss)
            test_corr_coeff = np.corrcoef(labels.cpu(),outputs.cpu())[0,1]
            pval = stats.pearsonr(labels.cpu(),outputs.cpu())
            test_mae = np.mean(np.abs(scaler.inverse_transform(outputs.cpu().numpy().reshape(-1,1)) - scaler.inverse_transform(labels.cpu().numpy().reshape(-1,1)))) 
            #print('Test Accuracy of the model: {} %'.format(abs(test_corr_coeff)))
            #print('P-value:',stats.pearsonr(labels.cpu(),outputs.cpu()))
        # print('Test Accuracy of the model: {} %'.format((correct / total) * 100))
        # print('Test F1 score of the model: {} %'.format(100*test_f1_score))
        #plot_ages(labels.cpu(),outputs.cpu())
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


data_dir = '/oak/stanford/groups/menon/projects/wdcai/2019_ADHD_NN/data/imaging/timeseries/CMIHBN/restfmri/timeseries/group_level/brainnetome/normz/'
files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
#print(files)
#pdb.set_trace()
count = 0
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

### Regression Target
c3sr = pd.read_csv('/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/prepare_data/adhd/C3SR.csv')
c3sr['Identifiers'] = c3sr['Identifiers'].apply(lambda x : x[0:12])

data['label'] = data['label'].astype(str).astype(int)

#hcp_dev_model = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/train_regression_models/dev/hcp_dev_age_model_generalize_ready.pt'
#nki_model = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/train_regression_models/nki/nki_age_cog_dev_model_generalize_ready_v1.pt'

corrs = []
pvals = []
maes = []
scalers = []
actual_ages = []
predicted_ages = []
hys_all = []
ids_all = []
genders_all = []

lin_params = np.load('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/nki_updated/lin_model_params_nki_5folds.npz')

coefs = lin_params['coef']
intercepts = lin_params['intercept']

for fold in range(5):

    path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/adhd_cmihbn_age_ADHD_wIDS/fold_%d.bin'%(fold)
 
    X_train, X_valid, id_train, Y_train, Y_valid, id_test = load_finetune_dataset_wids(path)

    path_dev = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/hcp_dev_age_five_fold/fold_%d.bin'%(fold)
    X_train_dev, X_valid_dev, Y_train_dev, Y_valid_dev = load_finetune_dataset(path_dev)

    hcp_dev_model = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/train_regression_models/dev/best_outer_fold_%d_hcp_dev_model_2_27_24.pt'%(fold)
     
    sc = StandardScaler()
    sc.fit(Y_train_dev.reshape(-1,1))

    Y_train_dev = sc.transform(Y_train_dev.reshape(-1,1))
    Y_valid = sc.transform(Y_valid.reshape(-1,1))
    
    #interp_data_valid = interp1d(np.linspace(0, 1,X_valid.shape[1]), X_valid, axis=1)
    #data_extend_valid = interp_data_valid(np.linspace(0, 1,math.floor(X_valid.shape[1] * 2 / 0.8)))
    X_train_dev = reshapeData(X_train_dev)
    X_valid = reshapeData(X_valid)

    id_test = id_test.astype('str')
    ids_all.append(list(id_test))

    #actual_train, predicted_train = test_model_getVals(X_train_dev,Y_train_dev,sc,hyper_parameters,hcp_dev_model)

    actual, predicted = test_model_getVals(X_valid,Y_valid,sc,hyper_parameters,hcp_dev_model)

    #BAG = (predicted_train - actual_train).reshape(-1,1)

    #lin_model = LinearRegression().fit(actual_train.reshape(-1,1),BAG)

    Offset = coefs[0] * actual + intercepts[0]

    predicted_corrected = predicted - Offset
 
    actual_ages = np.concatenate((actual_ages,actual))
    predicted_ages = np.concatenate((predicted_ages,predicted_corrected))
    
    #mask = np.isin(adhd_ids,id_test)
    #hy = hyp_scores[mask]
    #gender = label[mask]
    #hys.append(hy)
    #genders_all.append(gender)

ids_all = list(chain(*ids_all))


actual_ages = np.squeeze(actual_ages)
predicted_ages = np.squeeze(predicted_ages)

'''
print(actual_ages)
print(predicted_ages)

print(np.mean(np.abs(predicted_ages - actual_ages)))

r,p = scipy.stats.pearsonr(x=actual_ages,y=predicted_ages)
print(r)
print(p)

r_squared = r ** 2
#r,p = scipy.stats.pearsonr(x=actual_ages,y=predicted_ages-actual_ages)
#print(r)
#print(p)

#pdb.set_trace()
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
ax.set_title("CMI-HBN ADHD",fontsize=16)
plt.savefig('cmihbn_adhd_brain_age_scatter_test_bias_correction.png',format='png')

pdb.set_trace()
'''
np.savez('cmihbn_adhd_predicted_ages',predicted=predicted_ages)
np.savez('cmihbn_adhd_actual_ages',actual=actual_ages)
pdb.set_trace()

scores = np.asarray(c3sr.loc[c3sr['Identifiers'].isin(ids_all),'C3SR,C3SR_HY_T'])
c3sr_sub = c3sr.loc[c3sr['Identifiers'].isin(ids_all),:]

common_ids = [id_ for id_ in ids_all if id_ in list(c3sr_sub['Identifiers'])]
actual_subset = [actual_ages[ids_all.index(id_)] for id_ in common_ids]
predicted_subset = [predicted_ages[ids_all.index(id_)] for id_ in common_ids]

frame = pd.DataFrame({'ID':common_ids,'BAG':np.asarray(predicted_subset)-np.asarray(actual_subset),'HY':scores})
print(frame)
bag_vals = np.asarray(frame['BAG'])
print(np.mean(bag_vals))
#pdb.set_trace()
corr,pvalue = spss.spearmanr(frame['BAG'],frame['HY'])
print(corr)
print(pvalue)
