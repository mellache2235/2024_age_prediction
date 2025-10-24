"""
evaluate_by_devbin.py
--------------------------------
• Iterates over every developmental-stage folder that already has fold_X.bin
• Loads the matching **hcp_dev_model** and **scaler** for each fold
• Generates predictions, applies bias-correction,
  and prints r, MAE, mean BAG before/after correction.
"""
import math, pickle, torch, numpy as np, pandas as pd, scipy.stats as st
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.interpolate import interp1d

# ----------------------------------------------------------------------
# Root folders ----------------------------------------------------------
DATA_ROOT = Path("/oak/stanford/groups/menon/projects/mellache/"
                 "2024_age_prediction/data")    # dev-bin folders live here

# where your best_outer_fold_X models are
MODEL_ROOT = Path("/oak/stanford/groups/menon/projects/mellache/"
                  "2024_age_prediction/scripts/train_regression_models/dev")

# where the *hcp_dev* training data (.bin) for scalers is
HCP_DEV_TPL = ("/oak/stanford/groups/menon/projects/mellache/"
               "2021_foundation_model/data/imaging/for_dnn/"
               "hcp_dev_age_five_fold/fold_{fold}.bin")

DEV_BINS = ["child_5_8", "late_child_8_11", "early_ado_11_14",
            "midlate_ado_14_18", "emerging_adult_18_21"]

# ----------------------------------------------------------------------
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

def reshapeData(data):
    no_subjs, no_ts, no_channels = data.shape
    # Reshape data to no_subjs, no_channels, no_ts
    data_reshape = np.empty((no_subjs, no_channels, no_ts))
    for subj in np.arange(no_subjs):
        x_subj = data[subj, :, :]
        x_subj = np.transpose(x_subj)
        data_reshape[subj, :, :] = x_subj
    return data_reshape

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


def load_scaler_for_fold(fold:int)->StandardScaler:
    """
    Re-compute the StandardScaler **exactly** as during training
    (fit on Y_train_dev of HCP dev set, then returned).
    """
    path_dev = HCP_DEV_TPL.format(fold=fold)
    _, _, Y_train_dev, _ = load_finetune_dataset(path_dev)

    sc = StandardScaler()
    sc.fit(Y_train_dev.reshape(-1, 1))
    return sc

def load_fold0_asd(bin_path:Path):
    """Return X, y (years) for all subjects in fold_0.bin"""
    X_tr, X_val, y_tr, y_val = load_finetune_dataset(bin_path)
    X = np.concatenate((X_tr, X_val))
    y = np.concatenate((y_tr, y_val))
    #X = adjust_timesteps_for_subjects(X)                   # same preproc as training
    X = torch.tensor(np.stack(X)).float()
    return X, y
'''
def ensemble_predict(bin_dir:Path, fold_bins:list):
    """Return actual ages, ensemble-mean predictions for this dev-bin folder."""
    all_pred, y_act = [], None

    for fb in fold_bins:
        fold = int(fb.stem.split('_')[-1])      # fold_0.bin → 0
        # --------------------------------------------------- load model
        model_path = MODEL_ROOT / f"best_outer_fold_{fold}_hcp_dev_model_2_27_24.pt"
        model = ConvNet()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()

        # --------------------------------------------------- load & scale target
        scaler = load_scaler_for_fold(fold)

        X_tr, X_val, Y_tr, Y_val = load_finetune_dataset(fb)
        X = np.concatenate((X_tr, X_val))
        Y = np.concatenate((Y_tr, Y_val))
        Y = scaler.transform(Y.reshape(-1, 1)).squeeze()      # <- scaled ages used in training
        Y_orig = scaler.inverse_transform(Y.reshape(-1, 1)).squeeze()

        # same interpolation you used in training
        interp_fun = interp1d(np.linspace(0, 1, X.shape[1]), X, axis=1)
        X2 = interp_fun(np.linspace(0, 1, math.floor(X.shape[1] * 2 / 0.8)))
        X2 = reshapeData(X2)
        X2 = torch.tensor(X2).float()

        with torch.no_grad():
            y_pred_scaled = model(X2).squeeze().cpu().numpy()

        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).squeeze()
        print("fold", fold,
          "| child sample - first 5 true vs pred:",
          list(zip(Y_orig[:5], y_pred[:5])))
        y_act = Y_orig                                       # same order across folds
        all_pred.append(y_pred)

    ens_pred = np.mean(np.stack(all_pred), axis=0)
    return y_act, ens_pred
'''

def ensemble_predict(X, fold_idxs):
    """Forward X through each outer-fold model & scaler, return ensemble mean."""
    preds = []
    for k in fold_idxs:
        # ---- model -------------------------------------------------
        mpath = MODEL_ROOT / f"best_outer_fold_{k}_hcp_dev_model_2_27_24.pt"
        model = ConvNet();  model.load_state_dict(torch.load(mpath, map_location="cpu"))
        model.eval()

        # ---- scaler ------------------------------------------------
        #scal_path = MODEL_ROOT / f"fold_{k}_scaler.pkl"
        #scaler    = pickle.load(open(scal_path, "rb"))
        scaler = load_scaler_for_fold(k)
        # ---- forward & inverse-transform ---------------------------
        interp_fun = interp1d(np.linspace(0, 1, X.shape[1]), X, axis=1)
        X2 = interp_fun(np.linspace(0, 1, math.floor(X.shape[1] * 2 / 0.8)))
        X2 = reshapeData(X2)
        X2 = torch.tensor(X2).float()
        with torch.no_grad():
            y_scaled = model(X2).squeeze().cpu().numpy()
        y_pred = scaler.inverse_transform(y_scaled.reshape(-1,1)).ravel()
        preds.append(y_pred)

    return np.mean(np.stack(preds), axis=0)   # ensemble mean

def bias_correct(y_true, y_pred):
    lin_params = np.load('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/asd_updated/lin_params_td_abide.npz')

    coefs = lin_params['coef']
    intercepts = lin_params['intercept']
 
    bag = y_pred - y_true
    coef, intercept = coefs[0], intercepts[0]
    bag_corr = bag - (coef * y_true + intercept)
    return y_true + bag_corr

# ----------------------------------------------------------------------
records = []
for dev in DEV_BINS:
    bin_path = DATA_ROOT / dev / "fold_0.bin"
    X_asd, y_asd = load_fold0_asd(bin_path)
    y_pred = ensemble_predict(X_asd, fold_idxs=range(5))

    # ---- metrics before correction
    r0,_ = st.pearsonr(y_asd, y_pred)
    mae0 = np.mean(np.abs(y_pred - y_asd))
    mbag0= np.mean(y_pred - y_asd)

    # ---- bias correction
    y_pred_corr = bias_correct(y_asd, y_pred)
    r1,_ = st.pearsonr(y_asd, y_pred_corr)
    mae1 = np.mean(np.abs(y_pred_corr - y_asd))
    mbag1= np.mean(y_pred_corr - y_asd)

    records.append([dev, len(y_asd),
                    r0, mae0, mbag0,
                    r1, mae1, mbag1])

# ----------------------------------------------------------------------
cols = ["bin","N","r","MAE","mean_BAG","r_corr","MAE_corr","mean_BAG_corr"]
df = pd.DataFrame(records, columns=cols)
print("\n===  Per-development-bin metrics  ===")
print(df.to_string(index=False))

