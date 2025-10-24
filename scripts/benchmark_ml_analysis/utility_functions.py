import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
import torch.nn.functional as F
import math
import pickle
from sklearn.metrics import f1_score
import pandas as pd
from tqdm import tqdm
import pdb

def add_zeros(X,k):
    [N,T,C] = np.shape(X)
    X_extend = np.zeros((N,k*T,C))
    X_extend[:,0:k*T:k,:] = X
    return X_extend
       
    
def remove_nans(data,labels):
    ix_nan = np.isnan(labels)
    labels = labels[~ix_nan]
    data = data[~ix_nan, :, :]
    return data, labels

def get_thresholds(var,low_percentile,high_percentile):
    ix_nan = np.isnan(var)
    var = var[~ix_nan]
    thresh_low = np.percentile(var, low_percentile)
    thresh_high = np.percentile(var, high_percentile)
    return [thresh_low, thresh_high]

def binarize_labels(data,labels,thresholds):
    ix_neg = labels < thresholds[0]
    ix_pos = labels >= thresholds[1]
    data_neg = data[ix_neg, :, :]
    data_pos = data[ix_pos, :, :]
    print("Number of Positive and Negative Classes")
    print(sum(ix_neg),sum(ix_pos))
    y_neg = np.zeros(sum(ix_neg))
    y_pos = np.ones(sum(ix_pos))
    data_classif = np.concatenate((data_neg, data_pos))
    y = np.concatenate((y_neg, y_pos))
    y = y.astype('int')
    return data_classif,y


def reshapeData(data):
    no_subjs, no_ts, no_channels = data.shape
    # Reshape data to no_subjs, no_channels, no_ts
    data_reshape = np.empty((no_subjs, no_channels, no_ts))
    for subj in np.arange(no_subjs):
        x_subj = data[subj, :, :]
        x_subj = np.transpose(x_subj)
        data_reshape[subj, :, :] = x_subj
    return data_reshape


def get_data_loaders(Data,hyper_parameters, model_type = 'classification'):
    # Prepare data for data loader
    x_train = Data['train_features']
    y_train = Data['train_labels']
    x_valid = Data['valid_features']
    y_valid = Data['valid_labels']
    batch_size = hyper_parameters['batch_size']
    # Train Data
    input_tensor = torch.from_numpy(x_train).type(torch.FloatTensor)
    if model_type == 'regression':
        label_tensor = torch.from_numpy(y_train).type(torch.FloatTensor)
    elif model_type == 'classification':
        label_tensor = torch.from_numpy(y_train)
    dataset_train = TensorDataset(input_tensor, label_tensor)
    # Validation Data
    input_tensor_valid = torch.from_numpy(x_valid).type(torch.FloatTensor)
    if model_type == 'regression':
        label_tensor_valid = torch.from_numpy(y_valid).type(torch.FloatTensor)
    elif model_type == 'classification':
        label_tensor_valid = torch.from_numpy(y_valid)
    dataset_valid = TensorDataset(input_tensor_valid, label_tensor_valid)

    # Load Train and Test data into the loader
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=dataset_valid, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader


def get_data_loaders_forClassifiers(Data,hyper_parameters):
    # Prepare data for data loader
    x_train = Data['train_features']
    y_train = Data['train_labels'].astype('int64')
    if min(y_train) == 1:
        y_train=y_train-1
    x_valid = Data['valid_features']
    y_valid = Data['valid_labels'].astype('int64')
    if min(y_valid) == 1:
        y_valid = y_valid-1
    x_test = Data['test_features']
    if Data['test_labels'] != None:
        y_test = Data['test_labels'].astype('int64')
    else:
        y_test = None
    if y_test != None:
        if min(y_test) == 1:
            y_test = y_test-1
    batch_size = hyper_parameters['batch_size']

    # Train Data
    input_tensor = torch.from_numpy(x_train).type(torch.FloatTensor)
    label_tensor = torch.from_numpy(y_train)
    dataset_train = TensorDataset(input_tensor, label_tensor)

    # Validation Data
    input_tensor_valid = torch.from_numpy(x_valid).type(torch.FloatTensor)
    label_tensor_valid = torch.from_numpy(y_valid)
    dataset_valid = TensorDataset(input_tensor_valid, label_tensor_valid)

    # Test Data
    if x_test != None:
        input_tensor_test = torch.from_numpy(x_test).type(torch.FloatTensor)
        label_tensor_test = torch.from_numpy(y_test)
        dataset_test = TensorDataset(input_tensor_test, label_tensor_test)

    # Load Train and Test data into the loader
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=dataset_valid, batch_size=batch_size, shuffle=False)
    if x_test != None:
        test_loader = DataLoader(dataset=dataset_test, batch_size=x_test.shape[0], shuffle=False)
    else:
        test_loader = None
        
    return train_loader, valid_loader, test_loader

def get_data_loaders_forClassifiers_test(Data,hyper_parameters):
    # Prepare data for data loader
    x_train = Data['train_features']
    y_train = Data['train_labels'].astype('int64')
    if min(y_train) == 1:
        y_train=y_train-1
    x_valid = Data['valid_features']
    y_valid = Data['valid_labels'].astype('int64')
    if min(y_valid) == 1:
        y_valid = y_valid-1
    x_test = Data['test_features']
    if Data['test_labels'] != None:
        y_test = Data['test_labels'].astype('int64')
    else:
        y_test = None
    if y_test != None:
        if min(y_test) == 1:
            y_test = y_test-1
    batch_size = hyper_parameters['batch_size']

    combined_data = np.concatenate((x_train,x_valid))
    combined_labels = np.concatenate((y_train,y_valid))

    
    full_tensor = torch.from_numpy(combined_data).type(torch.FloatTensor)
    label_tensor = torch.from_numpy(combined_labels)
    dataset_full = TensorDataset(full_tensor, label_tensor)

    
    # Load Train and Test data into the loader
    full_loader = DataLoader(dataset=dataset_full, batch_size=batch_size, shuffle=False)

    return full_loader

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



#Pre-training
class ConvNet_resting_data_mask(nn.Module):
    def __init__(self,drop_out_rate=0.5):
        super(ConvNet_resting_data_mask, self).__init__()
        self.drop_out_rate = drop_out_rate
        #CNN Block 1
        self.layer1 = Conv1dSame(246, 256, kernel_size=7, stride=1)
        self.layer2 = nn.ReLU()
        self.layer3 = Conv1dSame(256, 256, kernel_size=3, stride=1)
        self.layer4 = nn.ReLU()
        self.layer5 = nn.AvgPool1d(kernel_size=3, stride=1, padding=np.int(3 / 2))
        # CNN Block 2
        self.layer6 = Conv1dSame(256, 512, kernel_size=10, stride=1)
        self.layer7 = nn.ReLU()
        self.layer8 = Conv1dSame(512, 512, kernel_size=10, stride=1)
        self.layer9 = nn.ReLU()
        self.layer10 = nn.AvgPool1d(kernel_size=7, stride=1, padding=np.int(7 / 2))
        # CNN Block 3: Bring back to Channel Size
        self.layer11 = Conv1dSame(512, 246, kernel_size=12, stride=1)
        self.layer12 = nn.ReLU()
        self.layer13 = Conv1dSame(246, 246, kernel_size=12, stride=1)
        self.layer14 = nn.ReLU()
        self.layer15 = nn.AvgPool1d(kernel_size=7, stride=1, padding=np.int(7 / 2))
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
        #out_embedd = out
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

#Fine-Tuning
# Create the Classifier class with the Embedding Model
class CovnetClassifier_brainnectome_wEmbedder(nn.Module):
    def __init__(self,fname_model,USE_CUDA,freeze_embedder=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(CovnetClassifier_brainnectome_wEmbedder, self).__init__()
        # Instantiate Embedding model
        self.embedder = ConvNet_resting_data_mask()
        if USE_CUDA:

            self.embedder.load_state_dict(torch.load(fname_model))
        else:
            self.embedder.load_state_dict(torch.load(fname_model,map_location=torch.device('cpu')))

        # Instantiate an one-layer feed-forward classifier
        self.drop_out = nn.Dropout(p=0.95)
        self.classifier = nn.Linear(512, 2)
        
        cnt = 0
        if freeze_embedder:
            for name, param in self.embedder.named_parameters():
                if cnt < 6:
                    param.requires_grad = False
                    cnt += 1
                #print(name, param.requires_grad)
        # Intialize FC layer with weights and bias from Linear Probing / Frozen  Model
        #self.classifier.weight.data = torch.from_numpy(weights).type(torch.FloatTensor)
        #self.classifier.bias.data = torch.from_numpy(bias.reshape(-1,1)).type(torch.FloatTensor)
        #self.classifier = nn.Sequential(nn.Linear(512,2),nn.Sigmoid())
        # # Freeze the embedding model

    def forward(self, x):
        # Feed input to Embedder
        out = self.embedder(x)
        # 1D CNN Layer
        # out = self.cnn(out)
        out = out.mean(axis=2)
        #Dropout
        out = self.drop_out(out)
        # Feed input to classifier to compute logits
        out = self.classifier(out)
        return out



#Fine-Tuning : Training Method
def train_classifier_wEmbedding(train_loader,valid_loader,test_loader, hyper_parameters,fname_model, USE_CUDA=False):

    if not hyper_parameters['briannectome']:
        model = CovnetClassifier_working_memory_wEmbedder(hyper_parameters['fname_masked_model'])
    else:
        model = CovnetClassifier_brainnectome_wEmbedder(hyper_parameters['fname_masked_model'],USE_CUDA)

    #pdb.set_trace()
    if USE_CUDA:
        model.cuda()
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss()

    # optimizer = torch.optim.Adam(model.parameters(),
    #                                  lr=hyper_parameters['learning_rate'], weight_decay=0.5 * 1e-2)
    optimizer = torch.optim.Adam(model.parameters(),
                                    lr=hyper_parameters['learning_rate'], weight_decay=0.5 * 1e-2)
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
            #outputs_embedd = outputs_embedd.detach()
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
                    #outputs_embed = outputs_embed.detach()
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
            if (i + 1) % 1 == 0:
                print(
                    'Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.2f}%,Valid Loss: {:.4f}, Valid Accuracy: {:.2f} '
                    .format(epoch + 1, hyper_parameters['num_epochs'], i + 1, total_step, train_loss,
                            train_acc, total_valid_loss, 100.0 * val_acc))
    #Apply on the Test Data
    if not hyper_parameters['briannectome']:
        model = CovnetClassifier_working_memory_wEmbedder(hyper_parameters['fname_masked_model'])
    else:
        model = CovnetClassifier_brainnectome_wEmbedder(hyper_parameters['fname_masked_model'],USE_CUDA)
    if USE_CUDA:
        model.load_state_dict(torch.load(fname_model))
        model.cuda()
    else:
        model.load_state_dict(torch.load(fname_model,map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            if USE_CUDA:
                images = images.cuda()
                labels = labels.cuda()
            outputs = model(images)
            #output_embed = output_embed.detach()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels).item()
        # test_accuracy = (correct / total) * 100
        print('Validation Accuracy of the model: {} %'.format((correct / total) * 100))
        # valid_f1_score = f1_score(labels.cpu(), predicted.cpu(), average='macro')
        # print('Valid F1 score of the model: {} %'.format(valid_f1_score))
        valid_f1_score = None

        
    if test_loader != None:
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                if USE_CUDA:
                    images = images.cuda()
                    labels = labels.cuda()
                outputs = model(images)
                #ya = ya.detach()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels).item()
            test_accuracy = (correct / total) * 100
            print('Test Accuracy of the model: {} %'.format((correct / total) * 100))
    else:
        test_accuracy = None
    
    
    return model, train_loss_list, train_acc_list, val_loss_list, val_acc_list, test_accuracy,valid_f1_score



def test_model(x_valid,y_valid,hyper_parameters, fname_model):
    criterion = nn.CrossEntropyLoss()
    input_tensor_valid = torch.from_numpy(x_valid).type(torch.FloatTensor)
    label_tensor_valid = torch.from_numpy(y_valid)
    dataset_valid = TensorDataset(input_tensor_valid, label_tensor_valid)
    valid_loader = DataLoader(dataset=dataset_valid, batch_size=x_valid.shape[0], shuffle=False)
    model = CovnetClassifier_brainnectome_wEmbedder(hyper_parameters['fname_masked_model'],False)
    model.load_state_dict(torch.load(fname_model,map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels).item()
        test_accuracy = (correct / total) * 100
        test_f1_score = f1_score(labels, predicted, average='macro')
        
        print('Test Accuracy of the model: {} %'.format((correct / total) * 100))
        print('Test F1 score of the model: {} %'.format(100*test_f1_score))
        
    return test_accuracy, test_f1_score

def test_model_wids(x_valid,y_valid,ids,hyper_parameters, fname_model):
    criterion = nn.CrossEntropyLoss()
    input_tensor_valid = torch.from_numpy(x_valid).type(torch.FloatTensor)
    label_tensor_valid = torch.from_numpy(y_valid)
    dataset_valid = TensorDataset(input_tensor_valid, label_tensor_valid)
    valid_loader = DataLoader(dataset=dataset_valid, batch_size=x_valid.shape[0], shuffle=False)
    model = CovnetClassifier_brainnectome_wEmbedder(hyper_parameters['fname_masked_model'],False)
    model.load_state_dict(torch.load(fname_model,map_location=torch.device('cpu')))
    model.eval()
    id_table = pd.DataFrame()
    #print(ids)
    with torch.no_grad():
        i = 0
        correct = 0
        total = 0
        for images, labels in valid_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            #id_table.append({'id':ids[i],'correct':labels,'predicted':predicted})
            #id_table['id'] = ids[i]
            id_table['correct'] = labels
            id_table['predicted'] = predicted
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels).item()
            i += 1
        test_accuracy = (correct / total) * 100
        test_f1_score = f1_score(labels, predicted, average='macro')

        # print('Test Accuracy of the model: {} %'.format((correct / total) * 100))
        # print('Test F1 score of the model: {} %'.format(100*test_f1_score))
    id_table.index = ids
    return test_accuracy, test_f1_score, id_table

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

def load_finetune_dataset_wtrs(path):
    """A function to load numpy dictionary for finetune data given path

    Args:
        path (string): A string path to load the numpy dictionary

    Returns:
        Numpy matrices (train_input, test_input, train_target, test_target)
    """
    fp = open(path, "rb")
    data_dict = pickle.load(fp)
    fp.close()
    return data_dict["X_train"], data_dict["X_test"], data_dict["tr_train"], data_dict["Y_train"], data_dict["Y_test"], data_dict["tr_test"]

def load_nofold_finetune_dataset(path):
    fp = open(path,"rb")
    data_dict = pickle.load(fp)
    fp.close()
    return data_dict["X"],data_dict["Y"]



def get_layers(model):
    layers = []
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            layers += get_layers(module)
        elif isinstance(module, nn.ModuleList):
            for m in module:
                layers += get_layers(m)
        else:
            layers.append(module)
    return layers

def prepare_data_sliding_window(data, labels, window_size, step):
    """Generates a windowed version of input data

    Args:
        data (numpy matrix): Input data to be windowed
        labels (numpy array): Labels corresponding to each sample
        window_size (int): The size of the window
        step (int): The stride of the window

    Returns:
        (numpy matrix, numpy array): Windowed version of input data
    """
    Nsubjs, N, Nchannels = data.shape
    width = np.int(np.floor(window_size / 2.0))
    labels_window = list()
    data_window = list()
    for subj in tqdm(range(Nsubjs)):
        for k in range(width, N - width - 1, step):
            x = data[subj, k - width : k + width, :]
            x = np.expand_dims(x, axis=0)
            data_window.append(x)
            # window_data = np.concatenate((window_data, x))
            if labels is not None:
                labels_window.append(labels[subj])
    window_data = np.concatenate(data_window, axis=0)

    if labels is not None:
        return (window_data, np.asarray(labels_window, dtype=np.int64))
    else:
        return window_data

def get_features_labels(data,labels):

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
