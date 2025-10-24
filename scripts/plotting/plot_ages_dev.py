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
import os
from sklearn.linear_model import LinearRegression
import matplotlib.backends.backend_pdf as pdf
from matplotlib import font_manager
from itertools import chain

USE_CUDA = False
# ### Set the hyperparameters
hyper_parameters = {}
hyper_parameters['num_epochs'] = 500
hyper_parameters['batch_size'] = 32
hyper_parameters['learning_rate'] = 0.00097119581997096
hyper_parameters['briannectome'] = True

CUDA_SEED = 2344
NP_RANDOM_SEED = 652
PYTHON_RANDOM_SEED = 819
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
os.environ["WANDB__SERVICE_WAIT"] = "300"
torch.use_deterministic_algorithms(True)
torch.cuda.manual_seed(CUDA_SEED)
torch.manual_seed(CUDA_SEED)
np.random.seed(NP_RANDOM_SEED)
random.seed(PYTHON_RANDOM_SEED)

font_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/clustering_analysis/arial.ttf'
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()

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
all_folds_pred = []


for fold in range(5):

    path_dev = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/hcp_dev_age_five_fold/fold_%d.bin'%(fold)
    X_train_dev, X_valid_dev, Y_train_dev, Y_valid_dev = load_finetune_dataset(path_dev)

    hcp_dev_model = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/train_regression_models/dev/best_outer_fold_%d_hcp_dev_model_2_27_24.pt'%(fold)

    sc = StandardScaler()
    sc.fit(Y_train_dev.reshape(-1,1))

    Y_train_dev = sc.transform(Y_train_dev.reshape(-1,1))
    Y_valid_dev = sc.transform(Y_valid_dev.reshape(-1,1))
    
    X_valid_dev = reshapeData(X_valid_dev)

    actual, predicted = test_model_getVals(X_valid_dev,Y_valid_dev,sc,hyper_parameters,hcp_dev_model)
    
    BAG = (predicted - actual).reshape(-1,1)

    lin_model = LinearRegression().fit(actual.reshape(-1,1),BAG)

    Offset = lin_model.coef_[0][0] * actual + lin_model.intercept_[0]

    predicted = predicted - Offset
    
    predicted_ages.append(predicted)
    actual_ages.append(actual)
    

#print('Mean Correlation:',np.mean(corrs))
#print('Mean MAE:',np.mean(maes))

actual_ages = np.asarray(list(chain(*actual_ages)))
predicted_ages = np.asarray(list(chain(*predicted_ages)))
#actual_ages = np.asarray(actual)
#all_folds_pred = np.asarray(all_folds_pred)

#ensemble_predictions = np.mean(all_folds_pred, axis=0)
#print(ensemble_predictions)
#print(actual_ages)
#pdb.set_trace()
#BAG = (predicted_ages - actual_ages).reshape(-1,1)

#lin_model = LinearRegression().fit(actual_ages.reshape(-1,1),BAG)

#Offset = lin_model.coef_[0][0] * actual_ages + lin_model.intercept_[0]

#predicted_ages = predicted_ages - Offset

r,p = scipy.stats.pearsonr(x=actual_ages,y=predicted_ages)
print(r ** 2)
print(p)

np.savez('actual_hcp_dev_ages_most_updated',actual=actual_ages)
np.savez('predicted_hcp_dev_ages_most_updated',predicted=predicted_ages)
pdb.set_trace()


r,p = scipy.stats.pearsonr(x=actual_ages,y=predicted_ages)
print(r)
print(p)
mae = np.mean(np.abs(predicted_ages-actual_ages))
mean_bag = np.mean(predicted_ages-actual_ages)
print(mae)
print(mean_bag)

r1,p1 = scipy.stats.pearsonr(actual_ages,predicted_ages-actual_ages)
print(r1)
print(p1)

r_squared = r ** 2

# Conditional formatting for p-value:
if p < 0.001:
    p_text = r"$\mathit{P} < 0.001$"
else:
    p_text = rf"$\mathit{{P}} = {p:.3f}$"

sns.set_style("white")
fig,ax = plt.subplots(figsize=(5.5,5.5),dpi=300)
sns.regplot(x=actual_ages, y=predicted_ages, ci=None,
           scatter_kws={'color': 'navy', 'alpha' : 0.6,'s': 40,'edgecolor': 'w', 'linewidth':0.5},
           line_kws={'color': 'red', 'linewidth' : 2},ax=ax)

lims = [5,23]
ax.plot(lims, lims, linestyle='--', color='gray', linewidth=1.2, label='Identity line')
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xticks(np.arange(6, 23, 2))
ax.set_yticks(np.arange(6, 23, 2))

# Final annotation with proper italicization and upright formatting:

ax.text(0.95, 0.05,
        f"$\mathit{{R}}^2$ = {r_squared:.3f}\n"
        f"{p_text}\n"
        f"$\mathrm{{MAE}} = {mae:.2f}\;\mathrm{{years}}$",
        transform=ax.transAxes,
        horizontalalignment='right',
        verticalalignment='bottom',
        fontsize=11)

ax.set_xlabel("Chronological Age",fontsize=15,labelpad=10)
ax.set_ylabel("Brain Age",fontsize=15,labelpad=10)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.tick_params(axis='both', which='major', length=6, width=1)
for spine in ['bottom', 'left']:
    ax.spines[spine].set_linewidth(1.5)
ax.spines[['right', 'top']].set_visible(False)
ax.set_title("HCP-Dev",fontsize=15,pad=10)
plt.tight_layout(pad=1.2)
#plt.savefig('dev_brain_age_scatter_high_quality.png',format='png')
#plt.savefig('dev_brain_age_scatter_high_quality.svg',format='svg')
pdf.FigureCanvas(fig).print_pdf('dev_brain_age_scatter_high_quality.ai')
pdb.set_trace()

print(np.mean(predicted_ages-actual_ages))

np.savez('hcp_dev_nested_predicted_ages',predicted=predicted_ages)
np.savez('hcp_dev_nested_actual_ages',actual=actual_ages)

    
