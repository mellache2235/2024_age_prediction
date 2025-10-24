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
import matplotlib.backends.backend_pdf as pdf
from matplotlib import font_manager

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
hys_all = []
ids_all = []
coef_list = []
intercept_list = []
all_folds_pred = []

#path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/abide_asd_asd_dev_age_wIDs/fold_0.bin'
path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/abide_asd_td_dev_age/fold_0.bin'

X_train, X_valid, Y_train, Y_valid = load_finetune_dataset(path)

X_total = np.concatenate((X_train,X_valid))
Y_total = np.concatenate((Y_train,Y_valid))

#ix = ((Y_total >= 7) & (Y_total <= 18))
#X_total = X_total[ix]
#Y_total = Y_total[ix]

for fold in range(5):

    path_dev = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/hcp_dev_age_five_fold/fold_%d.bin'%(fold)
    X_train_dev, X_valid_dev, Y_train_dev, Y_valid_dev = load_finetune_dataset(path_dev)

    hcp_dev_model = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/train_regression_models/dev/best_outer_fold_%d_hcp_dev_model_2_27_24.pt'%(fold)
    
    sc = StandardScaler()
    sc.fit(Y_train_dev.reshape(-1,1))
      
    Y_train_dev = sc.transform(Y_train_dev.reshape(-1,1)) 
    Y_total_new = sc.transform(Y_total.reshape(-1,1))
    
    interp_data_total = interp1d(np.linspace(0, 1,X_total.shape[1]), X_total, axis=1)
    data_extend_total = interp_data_total(np.linspace(0, 1,math.floor(X_total.shape[1] * 2 / 0.8)))

    X_total_new = reshapeData(data_extend_total)

    actual, predicted = test_model_getVals(X_total_new,Y_total_new,sc,hyper_parameters,hcp_dev_model)

    #BAG = (predicted - actual).reshape(-1,1)

    #lin_model = LinearRegression().fit(actual.reshape(-1,1),BAG)

    #Offset = lin_model.coef_[0][0] * actual + lin_model.intercept_[0]

    #coef_list.append(lin_model.coef_[0][0])
    #intercept_list.append(lin_model.intercept_[0])

    #predicted_corrected = predicted - Offset

    #actual_ages = np.concatenate((actual_ages,actual))
    all_folds_pred.append(predicted)


actual_ages = np.asarray(actual)
all_folds_pred = np.asarray(all_folds_pred)

ensemble_predictions = np.mean(all_folds_pred, axis=0)

BAG = (ensemble_predictions - actual_ages).reshape(-1,1)

lin_model = LinearRegression().fit(actual_ages.reshape(-1,1),BAG)

Offset = lin_model.coef_[0][0] * actual_ages + lin_model.intercept_[0]

coef_list.append(lin_model.coef_[0][0])
intercept_list.append(lin_model.intercept_[0])

#np.savez('lin_params_td_abide',coef=coef_list,intercept=intercept_list)
#pdb.set_trace()
#lin_params = np.load('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/nki_updated/lin_params_nki_ensemble.npz')

#coefs = lin_params['coef']
#intercepts = lin_params['intercept']


Offset = coef_list[0] * actual_ages + intercept_list[0]

predicted_ages = ensemble_predictions - Offset

print(actual_ages)
print(predicted_ages)

np.savez('actual_abide_td_ages_most_updated',actual=actual_ages)
np.savez('predicted_abide_td_ages_most_updated',predicted=predicted_ages)
pdb.set_trace()

r,p = scipy.stats.pearsonr(x=actual_ages,y=predicted_ages)
print(r)
print(p)
mae = np.mean(np.abs(predicted_ages-actual_ages))
mean_bag = np.mean(predicted_ages-actual_ages)
print(mae)
print(mean_bag)

pdb.set_trace()

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

padding = 1
lims = [
    min(actual_ages.min(), predicted_ages.min()) - padding,
    max(actual_ages.max(), predicted_ages.max()) + padding
]
#ax.plot(lims, lims, linestyle='--', color='gray', linewidth=1.2, label='Identity line')
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xticks(np.arange(int(lims[0]), int(lims[1])+1, 2))
ax.set_yticks(np.arange(int(lims[0]), int(lims[1])+1, 2))

# Final annotation with proper italicization and upright formatting:
ax.text(0.95, 0.05,
        f"$\mathit{{R}}^2$ = {r_squared:.3f}\n"
        f"{p_text}\n"
        f"$\mathrm{{MAE}} = {mae:.2f}\;\mathrm{{years}}$",
        transform=ax.transAxes,
        horizontalalignment='right',
        verticalalignment='bottom',
        fontsize=11)

ax.set_xlabel("Chronological Age",fontsize=16,labelpad=10)
ax.set_ylabel("Brain Age",fontsize=16,labelpad=10)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.tick_params(axis='both', which='major', length=6, width=1)
for spine in ['bottom', 'left']:
    ax.spines[spine].set_linewidth(1.5)
ax.spines[['right', 'top']].set_visible(False)
ax.set_title("ABIDE ASD Brain Age Prediction",fontsize=16,pad=10)
plt.tight_layout(pad=1.2)
#plt.savefig('abide_asd_brain_age_scatter_ensemble_prediction_test_bias_correction_high_quality.png',format='png')
pdf.FigureCanvas(fig).print_pdf('abide_asd_brain_age_scatter_ensemble_prediction_test_bias_correction_high_quality.ai')

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
