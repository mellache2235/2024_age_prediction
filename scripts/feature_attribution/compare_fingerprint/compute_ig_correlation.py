import numpy as np
import glob
import math
import random
import pandas as pd
from scipy.stats import spearmanr
import pdb
import pandas as pd

feature_dir = '/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/feature_attribution/'

abide_features = pd.read_csv(feature_dir + 'abide_asd_features_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS.csv')
adhd200_features = pd.read_csv(feature_dir + 'adhd200_adhd_features_all_sites_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS_single_model_predictions.csv')
cmihbn_features = pd.read_csv(feature_dir + 'cmihbn_adhd_weidong_cutoffs_features_all_sites_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS_single_model_predictions.csv')
stanford_features = pd.read_csv(feature_dir + 'stanford_asd_features_IG_convnet_regressor_trained_on_hcp_dev_top_regions_wIDS.csv')


abide_features = abide_features.drop(['Unnamed: 0','subject_id'],axis=1)
adhd200_features = adhd200_features.drop(['Unnamed: 0','subject_id'],axis=1)
cmihbn_features = cmihbn_features.drop(['Unnamed: 0'],axis=1)
stanford_features = stanford_features.drop(['Unnamed: 0','subject_id'],axis=1)

#print(stanford_features)
#print(abide_features)
#print(adhd200_features)
#print(cmihbn_features)

abide_features_mean = abide_features.mean(axis=0)
adhd200_features_mean = adhd200_features.mean(axis=0)
cmihbn_features_mean = cmihbn_features.mean(axis=0)
stanford_features_mean = stanford_features.mean(axis=0)

correlation, p_value = spearmanr(abide_features_mean, adhd200_features_mean)
print("Spearman correlation:", correlation)
print("p-value:", p_value)


