import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
import torch.nn.functional as F
import math
import pdb
import pickle
from sklearn.linear_model import LinearRegression
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as st
from scipy import stats
import wandb
import torch.distributions as dist

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

def remove_nans(data,labels):
    ix_nan = np.isnan(labels)
    labels = labels[~ix_nan]
    data = data[~ix_nan, :, :]
    return data, labels

def reshapeData(data):
    no_subjs, no_ts, no_channels = data.shape
    # Reshape data to no_subjs, no_channels, no_ts
    data_reshape = np.empty((no_subjs, no_channels, no_ts))
    for subj in np.arange(no_subjs):
        x_subj = data[subj, :, :]
        x_subj = np.transpose(x_subj)
        data_reshape[subj, :, :] = x_subj
    return data_reshape


def get_data_loaders_forRegression(Data,hyper_parameters):
    # Prepare data for data loader
    x_train = Data['train_features']
    y_train = Data['train_labels']

    x_valid = Data['valid_features']
    y_valid = Data['valid_labels']

    x_test = Data['test_features']
    y_test = Data['test_labels']

    batch_size = hyper_parameters['batch_size']

    # Train Data
    input_tensor = torch.from_numpy(x_train).type(torch.FloatTensor)
    label_tensor = torch.from_numpy(y_train).type(torch.FloatTensor)
    dataset_train = TensorDataset(input_tensor, label_tensor)

    # Validation Data
    input_tensor_valid = torch.from_numpy(x_valid).type(torch.FloatTensor)
    label_tensor_valid = torch.from_numpy(y_valid).type(torch.FloatTensor)
    dataset_valid = TensorDataset(input_tensor_valid, label_tensor_valid)

    # Test Data
    if x_test != None:
        input_tensor_test = torch.from_numpy(x_test).type(torch.FloatTensor)
        label_tensor_test = torch.from_numpy(y_test).type(torch.FloatTensor)
        dataset_test = TensorDataset(input_tensor_test, label_tensor_test)

    # Load Train and Test data into the loader
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=dataset_valid, batch_size=batch_size, shuffle=False)
    if x_test != None:
        test_loader = DataLoader(dataset=dataset_test, batch_size=x_test.shape[0], shuffle=False)
    else:
        test_loader = None
    return train_loader, valid_loader, test_loader

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(246, 32, kernel_size=5, stride=1, bias=False),
            nn.BatchNorm1d(32),
            nn.PReLU(32),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(32,32, kernel_size=7, stride=1, bias=False),
            nn.BatchNorm1d(32),
            nn.PReLU(32),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.drop_out = nn.Dropout(p=0.6)
        self.regressor = nn.Linear(32, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.drop_out(out)
        out = self.layer2(out)
        out = self.drop_out(out)
        out = out.mean(axis=2)
        out = self.regressor(out)
        return out


def train_Regressor_wEmbedding(train_loader,valid_loader,test_loader, hyper_parameters,fname_model, USE_CUDA=False):

    if not hyper_parameters['briannectome']:
        model = CovnetClassifier_working_memory_wEmbedder(hyper_parameters['fname_masked_model'])
    else:
        #model = CovnetRegressor_brainnectome_wEmbedder(hyper_parameters['fname_masked_model'],USE_CUDA)
        model = ConvNet()

    if USE_CUDA:
        model.cuda()
    # Loss and optimizer
    #criterion = RMSELoss()
    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(),
                                     lr=hyper_parameters['learning_rate'], weight_decay=0.0001)
    total_step = len(train_loader)

    # Train the model

    train_loss_list = []
    val_loss_list = []
    val_loss_temp = 100000000.0
    num_epochs = hyper_parameters['num_epochs']
    for epoch in range(num_epochs):
        # Put the model into the training mode
        model.train()
        for i, (data_ts, labels) in enumerate(train_loader):
            if USE_CUDA:
                data_ts = data_ts.cuda()
                labels = labels.cuda()
            # Run the forward pas
            outputs = model(data_ts)
            loss = torch.sqrt(criterion(outputs, labels))
            # Track the Training Loss
            train_loss_list.append(loss.item())
            train_loss = loss.item()

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the Training accuracy
            total = labels.size(0)
            # Validation Loss and Accuracy
            model.eval()
            with torch.no_grad():
                total_valid_loss = 0.0
                cnt = 0
                for images, labels in valid_loader:
                    if USE_CUDA:
                        images = images.cuda()
                        labels = labels.cuda()
                    outputs = model(images)
                    loss = criterion(outputs, labels).item()
                    total_valid_loss += loss
                    cnt += 1
                total_valid_loss = total_valid_loss / cnt
                val_loss_list.append(total_valid_loss)

            if val_loss_temp > total_valid_loss:
                val_loss_temp = total_valid_loss
                print('**Saving Model on Drive**')
                torch.save(model.state_dict(), fname_model)
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f} '
                      .format(epoch + 1, num_epochs, i + 1, total_step, train_loss,
                              total_valid_loss))
            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f} '
                      .format(epoch + 1, num_epochs, i + 1, total_step, train_loss,
                              total_valid_loss))
    #Apply on the Test Data
    if not hyper_parameters['briannectome']:
        model = CovnetClassifier_working_memory_wEmbedder(hyper_parameters['fname_masked_model'])
    else:
        model = ConvNet()
    if USE_CUDA:
        model.load_state_dict(torch.load(fname_model))
        model.cuda()
    else:
        model.load_state_dict(torch.load(fname_model,map_location=torch.device('cpu')))
    model.eval()
    targets_store = []
    outputs_store = []
    with torch.no_grad():
        for images, labels in valid_loader:
            if USE_CUDA:
                images = images.cuda()
                labels = labels.cuda()
            outputs = model(images)
            outputs_store.append(outputs.cpu().detach().numpy())
            targets_store.append(labels.cpu().detach().numpy())
        
        outputs_store = np.concatenate(outputs_store)
        targets_store = np.concatenate(targets_store)

    return model, train_loss_list, val_loss_list,outputs_store,targets_store


def test_model(x_valid,y_valid,hyper_parameters, fname_model):
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
            
            test_corr_coeff = np.corrcoef(labels.cpu(),outputs.cpu())[0,1]
            pval = stats.pearsonr(labels.cpu(),outputs.cpu())
            
            print('Test Accuracy of the model: {} %'.format(abs(test_corr_coeff)))
            print('P-value:',stats.pearsonr(labels.cpu(),outputs.cpu()))
    return test_corr_coeff, pval

def test_model_getVals(x_valid,y_valid,hyper_parameters, fname_model):
    criterion = nn.MSELoss()
    input_tensor_valid = torch.from_numpy(x_valid).type(torch.FloatTensor)
    label_tensor_valid = torch.from_numpy(y_valid)
    dataset_valid = TensorDataset(input_tensor_valid, label_tensor_valid)
    valid_loader = DataLoader(dataset=dataset_valid, batch_size=x_valid.shape[0], shuffle=False)
    model = CovnetRegressor_brainnectome_wEmbedder(hyper_parameters['fname_masked_model'],False)
    model.load_state_dict(torch.load(fname_model,map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            outputs = model(images)
            outputs = torch.squeeze(outputs)
            labels = torch.squeeze(labels)
            #print(outputs.shape)
            #print(labels.shape)
            loss = criterion(outputs, labels).item()
            #print(loss)
        return labels.cpu(), outputs.cpu()

def load_finetune_dataset_w_sites(path):
    """A function to load numpy dictionary for finetune data given path

    Args:
        path (string): A string path to load the numpy dictionary

    Returns:
        Numpy matrices (train_input, test_input, train_target, test_target)
    """
    fp = open(path, "rb")
    data_dict = pickle.load(fp)
    fp.close()
    return data_dict["X_train"], data_dict["X_test"], data_dict["site_train"], data_dict["Y_train"], data_dict["Y_test"], data_dict["site_test"]

def load_finetune_dataset_w_ids(path):
    """A function to load numpy dictionary for finetune data given path

    Args:
        path (string): A string path to load the numpy dictionary

    Returns:
        Numpy matrices (train_input, test_input, train_target, test_target)
    """
    fp = open(path, "rb")
    data_dict = pickle.load(fp)
    fp.close()
    return data_dict["X_train"], data_dict["X_test"], data_dict["id_train"], data_dict["Y_train"], data_dict["Y_test"], data_dict["id_test"]

