import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import KFold, StratifiedKFold
import wandb
from sklearn.metrics import accuracy_score, f1_score
import pickle
import math
import random
import pandas as pd
import pdb
from sklearn.preprocessing import StandardScaler
import scipy
import statistics as st
from sklearn.linear_model import LinearRegression

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

def reshapeData(data):
    no_subjs, no_ts, no_channels = data.shape
    # Reshape data to no_subjs, no_channels, no_ts
    data_reshape = np.empty((no_subjs, no_channels, no_ts))
    for subj in np.arange(no_subjs):
        x_subj = data[subj, :, :]
        x_subj = np.transpose(x_subj)
        data_reshape[subj, :, :] = x_subj
    return data_reshape

class Conv1dSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.cut_last_element = (kernel_size % 2 == 0 and stride == 1 and dilation % 2 == 1)
        self.padding = math.ceil((1 - stride + dilation * (kernel_size-1))/2)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, stride=stride, dilation=dilation,bias=False)

    def forward(self, x):
        if self.cut_last_element:
            return self.conv(x)[:, :, :-1]
        else:
            return self.conv(x)

class ConvNet(nn.Module):
    def __init__(self,drop_out_rate=0.6):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(246, 32, kernel_size=5, stride=1,bias=False),
            nn.BatchNorm1d(32),
            nn.PReLU(32),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=7, stride=1,bias=False),
            nn.BatchNorm1d(32),
            nn.PReLU(32),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.drop_out = nn.Dropout(p=drop_out_rate)
        self.regressor = nn.Linear(32, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.drop_out(out)
        out = self.layer2(out)
        out = self.drop_out(out)
        out = out.mean(axis=2)
        out = self.regressor(out)
        return out

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

def train(model, dataloader_1,dataloader_2, criterion, optimizer, fname_model,val_loss_temp,device):
    global scaler
    print('Training model...')
    model.train()
    all_labels_train = []
    all_preds_train = []
    for i, (inputs, labels) in enumerate(dataloader_1):
        inputs, labels = inputs.to(device), labels.to(device)
        #inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        all_labels_train.append(labels.cpu().detach().numpy())
        all_preds_train.append(outputs.cpu().detach().numpy())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            total_valid_loss = 0.0
            cnt = 0
            for v_inputs,v_labels in dataloader_2:
                v_inputs, v_labels = v_inputs.to(device),v_labels.to(device)
                v_outputs = model(v_inputs)
                loss = criterion(v_outputs,v_labels).item()
                total_valid_loss += loss
                cnt += 1
            total_valid_loss = total_valid_loss / cnt
        if val_loss_temp > total_valid_loss:
           val_loss_temp = total_valid_loss
           print('**Saving Model on Drive**')
           torch.save(model.state_dict(),fname_model)
           #artifact = wandb.Artifact('best_model',type='model')
           #artifact.add_file(fname_model)
           #wandb.log_artifact(artifact)
        if (i + 1) % 1 == 0:
            print(
                'Valid Loss: {:.4f}'.format(total_valid_loss))
    all_labels_train = np.concatenate(all_labels_train)
    all_preds_train = np.concatenate(all_preds_train)
    unscaled_labels = np.squeeze(scaler.inverse_transform(all_labels_train.reshape(-1,1)))
    unscaled_preds = np.squeeze(scaler.inverse_transform(all_preds_train.reshape(-1,1)))
    BAG = (unscaled_preds - unscaled_labels).reshape(-1,1)
    lin_model = LinearRegression().fit(unscaled_labels.reshape(-1,1),BAG)
    return val_loss_temp, lin_model

def evaluate(model, dataloader, criterion, fname_model,lin_model,device):
    global scaler
    model = ConvNet()
    model.load_state_dict(torch.load(fname_model))
    model = model.cuda()
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            all_labels.append(labels.cpu().detach().numpy())
            all_preds.append(outputs.cpu().detach().numpy())
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
    unscaled_labels = np.squeeze(scaler.inverse_transform(all_labels.reshape(-1,1)))
    unscaled_preds = np.squeeze(scaler.inverse_transform(all_preds.reshape(-1,1)))
    Offset = lin_model.coef_[0][0] * unscaled_labels + lin_model.intercept_[0]
    preds_corrected = unscaled_preds - Offset

    return preds_corrected, unscaled_labels

# Set up nested cross-validation
inner_cv = KFold(n_splits=3, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define sweep configuration for Weights and Biases
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'avg_val_corr',
        'goal': 'maximize'
    },
    'parameters': {
        'epochs': {
            'values': [200]
        },
        'dropout_rate': {
            'min': 0.0,
            'max': 0.95
        },
        'batch_size': {
            'values': [16, 32]
        },
        'weight_decay': {
            'min': 1e-4,
            'max': 1e-3
        },
        'learning_rate': {
            'min': 1e-4,
            'max': 1e-3
        }
    }
}

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

entity = 'mellache'
project = 'nested_cv_hcp_dev_age_session2_model_bias_correction_4_28_25'
sweep_id = wandb.sweep(sweep_config, project=project,entity=entity)
print(f"Created sweep with ID: {sweep_id}")

def train_with_wandb(config=None):
    best_inner_score = float('inf')
    with wandb.init(config=config):
        config = wandb.config

        # Set up data for inner cross-validation
        inner_fold_corr = []
        for inner_train_idx, inner_val_idx in inner_cv.split(X_train,y_train):
            X_inner_train, X_val = X_train[inner_train_idx], X_train[inner_val_idx]
            y_inner_train, y_val = y_train[inner_train_idx], y_train[inner_val_idx]

            # Create model, optimizer, criterion
            model = ConvNet(config.dropout_rate).to(device)
            model = model.cuda()
 
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            criterion = RMSELoss()

            # Create DataLoader for training and validation
            input_tensor = torch.from_numpy(X_inner_train).type(torch.FloatTensor)
            label_tensor = torch.from_numpy(y_inner_train).type(torch.FloatTensor)
            dataset_train = TensorDataset(input_tensor, label_tensor)

            input_tensor_valid = torch.from_numpy(X_val).type(torch.FloatTensor)
            label_tensor_valid = torch.from_numpy(y_val).type(torch.FloatTensor)
            dataset_valid = TensorDataset(input_tensor_valid, label_tensor_valid)
            train_loader = DataLoader(dataset=dataset_train, batch_size=config.batch_size, shuffle=True)
            val_loader = DataLoader(dataset=dataset_valid, batch_size=config.batch_size, shuffle=False)
            #train_loader = torch.utils.data.DataLoader(list(zip(X_inner_train, y_inner_train)), batch_size=config.batch_size, shuffle=True)
            #val_loader = torch.utils.data.DataLoader(list(zip(X_val, y_val)), batch_size=config.batch_size, shuffle=False)

            # Train the model
            fname_model = 'best_inner_fold_hcp_dev_age_session2_model_4_28_25.pt'
            val_loss_temp = 100000.0
            for epoch in range(config.epochs):
                print(f'Epoch {epoch + 1}/{config.epochs}')
                val_loss_temp,lin_model = train(model, train_loader, val_loader,criterion, optimizer, fname_model,val_loss_temp,device)
            predicted, actual = evaluate(model, val_loader, criterion, fname_model,lin_model,device)
            corr_coef,p = scipy.stats.pearsonr(x=np.squeeze(actual),y=np.squeeze(predicted))
            inner_fold_corr.append(corr_coef)
            print(f'Inner Fold - Validation Correlation: {corr_coef:.4f}')
            wandb.log({'val_correlation':corr_coef})
            if val_loss_temp < best_inner_score:
                best_inner_score = val_loss_temp
                best_model_filename = f'best_model_inner_fold_4_28_25.pt'
                torch.save(model.state_dict(),best_model_filename)
                artifact = wandb.Artifact(f'best_model_inner_fold_4_28_25',type='model')
                artifact.add_file(best_model_filename)
                wandb.log_artifact(artifact) 
        avg_inner_fold_corr = np.mean(inner_fold_corr)
        wandb.log({'avg_val_corr':avg_inner_fold_corr})

outer_results = []
best_outer_hyperparams = None
best_outer_corr = 0.0
# Start outer cross-validation
for fold in range(5):
    #path = "/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/data/imaging/for_dnn/hcp_dev_age_five_fold/fold_%d.bin"%(fold)
    path = "/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/data/hcp_dev_age_run1_PA/fold_%d.bin"%(fold)

    X_train, X_test, y_train, y_test = load_finetune_dataset(path)
    
    X_train = reshapeData(X_train)
    X_test = reshapeData(X_test)

    scaler = StandardScaler()
    scaler.fit(y_train.reshape(-1,1))

    y_train = scaler.transform(y_train.reshape(-1,1))
    y_test = scaler.transform(y_test.reshape(-1,1))
    # Run Weights and Biases agent for Bayesian Optimization
    wandb.agent(sweep_id, function=train_with_wandb, count=50)

    # Get best hyperparameters from the sweep
    best_run = wandb.Api().sweep(f'{entity}/{project}/{sweep_id}').best_run()
    best_hyperparams = best_run.config
    #best_model_artifact = wandb.Api().artifact(f'{best_run.entity}/{best_run.project}/best_model_inner_fold:latest')
    #best_model_artifact = best_run.use_artifact('best_model_inner_fold:model')
    #best_model_dir = best_model_artifact.download() 
    # Train final model on outer fold with best hyperparameters
    model = ConvNet(best_hyperparams['dropout_rate']).to(device)
    model = model.cuda()
    #fname_model_outer = f'outer_fold_{fold}_model.pt'
    #state_dict = torch.load(f'{best_model_dir}/best_model_inner_fold.pt')
    #model.load_state_dict(state_dict)
    #torch.save(model.state_dict(),fname_model_outer)

    optimizer = torch.optim.AdamW(model.parameters(), lr=best_hyperparams['learning_rate'], weight_decay=best_hyperparams['weight_decay'])
    criterion = RMSELoss()

    input_tensor = torch.from_numpy(X_train).type(torch.FloatTensor)
    label_tensor = torch.from_numpy(y_train).type(torch.FloatTensor)
    dataset_train = TensorDataset(input_tensor, label_tensor)

    input_tensor_test = torch.from_numpy(X_test).type(torch.FloatTensor)
    label_tensor_test = torch.from_numpy(y_test).type(torch.FloatTensor)
    dataset_test = TensorDataset(input_tensor_test, label_tensor_test)
    
    train_loader = DataLoader(dataset=dataset_train, batch_size=best_hyperparams['batch_size'], shuffle=True)
    test_loader = DataLoader(dataset=dataset_test, batch_size=best_hyperparams['batch_size'], shuffle=False)
    
    fname_model_outer = f'best_outer_fold_{fold}_hcp_dev_session2_model_4_28_25.pt'
    print('Outer Fold Training!')
    test_loss_temp = 1000000.0
    for epoch in range(best_hyperparams['epochs']):
        test_loss_temp, lin_model = train(model, train_loader, test_loader,criterion, optimizer, fname_model_outer,test_loss_temp,device)
    
    # Evaluate on outer test set
    predicted,actual = evaluate(model, test_loader, criterion,fname_model_outer,lin_model,device)
    corr_coef,p = scipy.stats.pearsonr(x=np.squeeze(actual),y=np.squeeze(predicted))
    outer_results.append(corr_coef)
    print('Test Correlation:',corr_coef)
    if corr_coef > best_outer_corr:
        best_outer_corr = corr_coef
        best_outer_hyperparams = best_run.config
# Calculate average performance across outer folds
average_corr = np.mean(outer_results)
std_corr = np.std(outer_results)
print(f'Average Performance across Outer Folds: Correlation: {average_corr}')
print(f'Standard Deviation across Outer Folds: {std_corr}')
#print('Best Dropout:',best_outer_hyperparams['dropout_rate'])
#print('Best Learning Rate:',best_outer_hyperparams['learning_rate'])
#print('Best Batch Size:',best_outer_hyperparams['batch_size'])
#print('Best Decay:',best_outer_hyperparams['weight_decay'])
#print('Best Epochs:',best_outer_hyperparams['epochs'])
