import pandas as pd
import numpy as np
import math
import random
import pdb
import seaborn as sns
import matplotlib.pyplot as plt
import random
import scipy
from sklearn.metrics import mean_absolute_error
import sys
sys.path.append('/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/train_regression_models/')
from utility_functions import *
import scipy
from sklearn.preprocessing import StandardScaler, LabelEncoder

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

actual_ages = []
predicted_ages = []
corrs = []
pvals = []
maes = []

for fold in range(5):

    path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/hcp_dev_age_five_fold/fold_%d.bin'%(fold)

    hcp_dev_model = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/train_regression_models/dev/best_outer_fold_%d_hcp_dev_model_2_6_24.pt'%(fold)
   
    X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(path)

    sc = StandardScaler()
    sc.fit(Y_train.reshape(-1,1))

    X_valid = reshapeData(X_valid)
    Y_valid = sc.transform(Y_valid.reshape(-1,1))

    valid_corr_coef, valid_mae, valid_pval = test_model(X_valid,Y_valid,sc,hyper_parameters,hcp_dev_model)

    corrs.append(valid_corr_coef)
    pvals.append(valid_pval)
    maes.append(valid_mae)

    actual, predicted = test_model_getVals(X_valid,Y_valid,sc,hyper_parameters,hcp_dev_model)
    actual_ages = np.concatenate((actual_ages,actual))
    predicted_ages = np.concatenate((predicted_ages,predicted))

print('Mean Correlation:',np.mean(corrs))
print('Mean MAE:',np.mean(maes))
pdb.set_trace()

actual_ages = np.squeeze(actual_ages)
predicted_ages = np.squeeze(predicted_ages)
print(np.mean(predicted_ages-actual_ages))

np.savez('hcp_dev_nested_predicted_ages',predicted=predicted_ages)
np.savez('hcp_dev_nested_actual_ages',actual=actual_ages)

    
