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
#import statsmodels.api as sm

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

class Conv1dSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.cut_last_element = (kernel_size % 2 == 0 and stride == 1 and dilation % 2 == 1)
        self.padding = math.ceil((1 - stride + dilation * (kernel_size-1))/2)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, stride=stride, dilation=dilation)

    def forward(self, x):
        if self.cut_last_element:
            return self.conv(x)[:, :, :-1]
        else:
            return self.conv(x)

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

        self.drop_out = nn.Dropout(p=0.40581892575490663)
        self.regressor = nn.Linear(32, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.drop_out(out)
        out = self.layer2(out)
        out = self.drop_out(out)
        out = out.mean(axis=2)
        out = self.regressor(out)
        return out

class ConvNet_v2(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1dSame(246, 128, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3, stride=1, padding = int(3/2)))
        self.layer2 = nn.Sequential(
            nn.Conv1dSame(128 ,128, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=5, stride=1, padding = int(5/2)))

        self.drop_out = nn.Dropout(p=0.40581892575490663)
        self.regressor = nn.Linear(128, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.drop_out(out)
        out = self.layer2(out)
        out = self.drop_out(out)
        out = out.mean(axis=2)
        out = self.regressor(out)
        return out

class ConvNet_resting_data_mask(nn.Module):
    def __init__(self,drop_out_rate=0.5):
        super(ConvNet_resting_data_mask, self).__init__()
        self.drop_out_rate = drop_out_rate
        #CNN Block 1
        self.layer1 = Conv1dSame(246, 256, kernel_size=3, stride=1)
        self.layer2 = nn.ReLU()
        self.layer3 = Conv1dSame(256, 256, kernel_size=3, stride=1)
        self.layer4 = nn.ReLU()
        self.layer5 = nn.AvgPool1d(kernel_size=3, stride=1, padding=int(3 / 2))
        # CNN Block 2
        self.layer6 = Conv1dSame(256, 512, kernel_size=10, stride=1)
        self.layer7 = nn.ReLU()
        self.layer8 = Conv1dSame(512, 512, kernel_size=10, stride=1)
        self.layer9 = nn.ReLU()
        self.layer10 = nn.AvgPool1d(kernel_size=7, stride=1, padding=int(7 / 2))
        # CNN Block 3: Bring back to Channel Size
        self.layer11 = Conv1dSame(512, 246, kernel_size=12, stride=1)
        self.layer12 = nn.ReLU()
        self.layer13 = Conv1dSame(246, 246, kernel_size=12, stride=1)
        self.layer14 = nn.ReLU()
        self.layer15 = nn.AvgPool1d(kernel_size=7, stride=1, padding=int(7 / 2))
        # Dropout
        self.drop_out = nn.Dropout(p=self.drop_out_rate)
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        #out = self.layer11(out)
        #out = self.layer12(out)
        #out = self.layer13(out)
        #out = self.layer14(out)
        #out = self.layer15(out)
        #out = self.drop_out(out)
        return out


def modified_mse_loss(data_ts_mask,data_ts_original,mask_indices):
    mask_indices = mask_indices.long()
    x = data_ts_mask[:,:,mask_indices]
    y = data_ts_original[:,:,mask_indices]
    loss = torch.mean((x - y)**2)
    return loss

class ConvNet_scsnl_tdmd_reg_2(nn.Module):
    def __init__(self):
        super(ConvNet_scsnl_tdmd_reg_2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(246, 128, kernel_size=8, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(128, 32, kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.drop_out = nn.Dropout(p=0.8)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out = self.layer1(x)
        # out = self.drop_out(out)
        out = self.layer2(out)
        # out = self.drop_out(out)
        out = out.mean(axis=2)
        out = self.drop_out(out)
        out = self.fc(out)
        return out

# Create the Classifier class with the Embedding Model
class CovnetRegressor_brainnectome_wEmbedder(nn.Module):
    def __init__(self, fname_model,USE_CUDA,freeze_embedder=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(CovnetRegressor_brainnectome_wEmbedder, self).__init__()
        # Instantiate Embedding model
        self.embedder = ConvNet_resting_data_mask()
        if USE_CUDA:
            self.embedder.load_state_dict(torch.load(fname_model))
        else:
            self.embedder.load_state_dict(torch.load(fname_model,map_location=torch.device('cpu')))

        # Instantiate an one-layer feed-forward classifier
        self.cnn = nn.Sequential(
            nn.Conv1d(246, 246, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1))
        self.drop_out = nn.Dropout(p=0.95)
        self.regressor =  nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
        # # Freeze the embedding model
        # if freeze_embedder:
        #     for param in self.embedder.parameters():
        #         param.requires_grad = False

    def forward(self, x):
        # Feed input to Embedder
        out = self.embedder(x)
        # 1D CNN Layer
        #out = self.cnn(out)
        out = out.mean(axis=2)
        #Dropout
        out = self.drop_out(out)
        # Feed input to classifier to compute logits
        out = self.regressor(out)
        #out = self.sigmoid(out)
        return out

class MeanReduction(nn.Module):
    def __init__(self, dim):
        super(MeanReduction, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.mean(dim=self.dim)

class CovnetRegressor_brainnectome_wEmbedder_probabilistic(nn.Module):
    def __init__(self, fname_model,USE_CUDA,freeze_embedder=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(CovnetRegressor_brainnectome_wEmbedder_probabilistic, self).__init__()
        # Instantiate Embedding model
        self.embedder = ConvNet_resting_data_mask()
        if USE_CUDA:
            self.embedder.load_state_dict(torch.load(fname_model))
        else:
            self.embedder.load_state_dict(torch.load(fname_model,map_location=torch.device('cpu')))

        # Instantiate an one-layer feed-forward classifier
        self.cnn = nn.Sequential(
            nn.Conv1d(512, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3, stride=1,padding=int(3/2)),
        )
        self.jitter = 1e-6
        '''
        self.drop_out = nn.Dropout()
        self.jitter = 1e-6
        
        self.mean_layer = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=7, stride=1),
            nn.ReLU(),
            MeanReduction(dim=2),
            nn.Dropout(),
            nn.Linear(32,1),
        )
        self.std_layer = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=7, stride=1),
            nn.ReLU(),
            MeanReduction(dim=2),
            nn.Dropout(),
            nn.Linear(32,1),
        )
        '''
        self.drop_out = nn.Dropout(p=0.95)
        self.mean_linear = nn.Linear(512,1)
        self.std_linear = nn.Linear(512,1)

    def forward(self, x):
        # Feed input to Embedder
        out = self.embedder(x)
        out = out.mean(axis=2)
        out = self.drop_out(out)
        mean = self.mean_linear(out)
     
        std = F.softplus(self.std_linear(out)) + self.jitter
        return torch.distributions.Normal(mean,std)

class CovnetRegressor_brainnectome_wEmbedder_wandb(nn.Module):
    def __init__(self, fname_model,USE_CUDA,drop_out=0.95,freeze_embedder=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(CovnetRegressor_brainnectome_wEmbedder_wandb, self).__init__()
        # Instantiate Embedding model
        self.embedder = ConvNet_resting_data_mask()
        if USE_CUDA:
            self.embedder.load_state_dict(torch.load(fname_model))
        else:
            self.embedder.load_state_dict(torch.load(fname_model,map_location=torch.device('cpu')))

        # Instantiate an one-layer feed-forward classifier
        self.cnn = nn.Sequential(
            nn.Conv1d(246, 246, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1))
        self.drop_out_layer = nn.Dropout(p = drop_out)
        self.regressor =  nn.Linear(512, 1)

        # # Freeze the embedding model
        # if freeze_embedder:
        #     for param in self.embedder.parameters():
        #         param.requires_grad = False

    def forward(self, x):
        # Feed input to Embedder
        out = self.embedder(x)
        # 1D CNN Layer
        #out = self.cnn(out)
        out = out.mean(axis=2)
        #Dropout
        out = self.drop_out_layer(out)
        # Feed input to classifier to compute logits
        out = self.regressor(out)
        return out

class CovnetRegressor_brainnectome(nn.Module):
    def __init__(self):
        super(CovnetRegressor_brainnectome, self).__init__()
        #CNN Block 1
        self.layer1 = nn.Sequential(
            nn.Conv1d(246, 256, kernel_size=7, stride=1),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=7, stride=1),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=2))
        # CNN Block 1
        self.layer4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=5, stride=1),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=5, stride=1),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=2))
        # Dropout
        self.drop_out = nn.Dropout(p=0.95)
        # Output FC Layer
        self.fc1 = nn.Linear(512, 1)
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        #Temporal Averaging
        out = out.mean(axis=2)
        out = self.drop_out(out)
        out = self.fc1(out)
        return out



def train_classifier_scratch(train_loader,valid_loader,test_loader, hyper_parameters,fname_model, USE_CUDA=False):

    model = CovnetRegressor_brainnectome()


    if USE_CUDA:
        model.cuda()
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss()

    optimizer = torch.optim.Adam(model.parameters(),
                                     lr=hyper_parameters['learning_rate'], weight_decay=0.5 * 1e-1)
    total_step = len(train_loader)

    # Train the model

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    val_acc_temp = 0.0
    for epoch in range(hyper_parameters['num_epochs']):
        # Put the model into the training mode
        model.train()
        for i, (data_ts, labels) in enumerate(train_loader):
            if USE_CUDA:
                data_ts = data_ts.cuda()
                labels = labels.cuda()
            # Run the forward pas
            outputs = model(data_ts)
            loss = criterion(outputs, labels)
            # Track the Training Loss
            train_loss_list.append(loss.item())
            train_loss = loss.item()

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the Training accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            train_acc_list.append(correct / total)

            train_acc = 100.0 * correct / total

            # Validation Loss and Accuracy
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                total_valid_loss = 0.0
                cnt = 0
                for images, labels in valid_loader:
                    if USE_CUDA:
                        images = images.cuda()
                        labels = labels.cuda()
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    loss = criterion(outputs, labels).item()
                    total_valid_loss += loss
                    cnt += 1
                total_valid_loss = total_valid_loss / cnt
                val_loss_list.append(total_valid_loss)
                val_acc = (correct / total)
                val_acc_list.append(val_acc)

            if val_acc_temp < val_acc:
                val_acc_temp = val_acc
                print('**Saving Model on Drive**')
                torch.save(model.state_dict(), fname_model)
            if (i + 1) % 10 == 0:
                print(
                    'Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.2f}%,Valid Loss: {:.4f}, Valid Accuracy: {:.2f} '
                    .format(epoch + 1, hyper_parameters['num_epochs'], i + 1, total_step, train_loss,
                            train_acc, total_valid_loss, 100.0 * val_acc))
    #Apply on the Test Data
    model = CovnetClassifier_brainnectome()
    if USE_CUDA:
        model.load_state_dict(torch.load(fname_model))
        model.cuda()
    else:
        model.load_state_dict(torch.load(fname_model,map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            if USE_CUDA:
                images = images.cuda()
                labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels).item()
        test_accuracy = (correct / total) * 100
        print('Test Accuracy of the model: {} %'.format((correct / total) * 100))

    return model, train_loss_list, train_acc_list, val_loss_list, val_acc_list,test_accuracy


def train_Regressor_wEmbedding_train_only(train_loader,valid_loader,test_loader,hyper_parameters,fname_model,USE_CUDA=False):
    model = ConvNet()
    if USE_CUDA:
        model.cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=hyper_parameters['learning_rate'],weight_decay=0.0006830904696091078)
    total_step = len(train_loader)
    train_loss_list = []
    val_loss_list = []
    val_loss_temp = 1000000000000.0
    num_epochs = hyper_parameters['num_epochs']
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for i, (data_ts, labels) in enumerate(train_loader):
            if USE_CUDA:
                data_ts = data_ts.cuda()
                labels = labels.cuda()
            outputs = model(data_ts)
            loss = torch.sqrt(criterion(outputs,labels))
            train_loss_list.append(loss.item())
            train_loss = loss.item()
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_train_loss += train_loss

            total = labels.size(0)
            model.eval()
            with torch.no_grad():
                total_valid_loss = 0.0
                cnt = 0
                for images, labels in valid_loader:
                    if USE_CUDA:
                        images = images.cuda()
                        labels = labels.cuda()
                    outputs = model(images)
                    loss = torch.sqrt(criterion(outputs, labels))
                    total_valid_loss += loss.item()
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

    return model, train_loss_list


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
        #model = CovnetRegressor_brainnectome_wEmbedder(hyper_parameters['fname_masked_model'],USE_CUDA)
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
            #print(outputs)
            #print(labels)
            #outputs = torch.squeeze(outputs)
            #labels = torch.squeeze(labels)
            outputs_store.append(outputs.cpu().detach().numpy())
            targets_store.append(labels.cpu().detach().numpy())
            #print(labels)
            #print(outputs)
            #_, predicted = outputs.data
            #loss = criterion(outputs, labels).item()
            #test_corr_coeff = np.corrcoef(labels.cpu(),outputs.cpu())[0,1]
            #coef.append(abs(test_corr_coeff))
        #print(outputs_store)
        #print(targets_store)
        outputs_store = np.concatenate(outputs_store)
        targets_store = np.concatenate(targets_store)
        #corr_coef = np.corrcoef(outputs_store,targets_store)[0,1]
        #corr_pval = stats.pearsonr(outputs_store,targets_store)[1]
        #test_corr_coeff = np.corrcoef(labels.cpu(),outputs.cpu())[0,1]
        #print('Test Accuracy/Correlation of the model: {} %'.format(corr_coef))
    #Apply on the Train Data
    targets_store_train = []
    outputs_store_train = []
    with torch.no_grad():
        for images, labels in train_loader:
            if USE_CUDA:
                images = images.cuda()
                labels = labels.cuda()
            outputs = model(images)
            outputs_store_train.append(outputs.cpu().detach().numpy())
            targets_store_train.append(labels.cpu().detach().numpy())
        targets_store_train = np.concatenate(targets_store_train)
        outputs_store_train = np.concatenate(outputs_store_train)
        #X_train_const = sm.add_constant(targets_store_train)
        #lin_model = sm.OLS(outputs_store_train,X_train_const).fit()
        #intercept,slope = lin_model.params
        #outputs_store = (outputs_store - intercept) / slope
        #outputs_store_train = (outputs_store_train - intercept) / slope

    return model, train_loss_list, val_loss_list,outputs_store,targets_store,outputs_store_train,targets_store_train

def plot_ages(actual,predicted,tmp,tmp2):
    fig, ax = plt.subplots(1,2,sharex=True,sharey=True,figsize=(10,5))
    model = LinearRegression()
    model.fit(actual.reshape((-1,1)),predicted)
    line = model.predict(actual.reshape((-1,1)))
    p = sns.scatterplot(x=actual,y=predicted,ax=ax[0])
    ax[0].set(xlabel="Actual Age",ylabel="Predicted Age")
    ax[0].plot(actual.reshape((-1,1)),line,color="blue")
    ax[0].set_title(f"NKI Actual vs Predicted Age")
    model2 = LinearRegression()
    model2.fit(tmp.reshape((-1,1)),tmp2)
    line2 = model2.predict(tmp.reshape((-1,1)))
    p2 = sns.scatterplot(x=tmp,y=tmp2,ax=ax[1])
    ax[1].set(xlabel="Actual Age",ylabel="Predicted Age")
    ax[1].plot(tmp.reshape((-1,1)),line2,color="blue")
    ax[1].set_title(f"Leipzig Actual vs Predicted Age")
    plt.show()

def test_model(x_valid,y_valid,hyper_parameters, fname_model):
    criterion = nn.MSELoss()
    input_tensor_valid = torch.from_numpy(x_valid).type(torch.FloatTensor)
    label_tensor_valid = torch.from_numpy(y_valid).type(torch.FloatTensor)
    dataset_valid = TensorDataset(input_tensor_valid, label_tensor_valid)
    valid_loader = DataLoader(dataset=dataset_valid, batch_size=x_valid.shape[0], shuffle=False)
    model = CovnetRegressor_brainnectome_wEmbedder(hyper_parameters['fname_masked_model'],False)
    #model = ConvNet()
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
            #print('Test Accuracy of the model: {} %'.format(abs(test_corr_coeff)))
            #print('P-value:',stats.pearsonr(labels.cpu(),outputs.cpu()))
        # print('Test Accuracy of the model: {} %'.format((correct / total) * 100))
        # print('Test F1 score of the model: {} %'.format(100*test_f1_score))
        #plot_ages(labels.cpu(),outputs.cpu())
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

def load_finetune_dataset_wids(path):
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

