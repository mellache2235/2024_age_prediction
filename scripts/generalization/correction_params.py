import numpy as np
import pandas as pd
import math
import random
import pdb
import sklearn
from sklearn.linear_model import LinearRegression
import scipy
import pickle


actual_nki = np.load('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/nki_updated/actual_nki_ages.npz')
actual_nki_age = actual_nki['actual']

predicted_nki = np.load('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/nki_updated/predicted_nki_ages.npz')
predicted_nki_age = predicted_nki['predicted']

actual_cmi = np.load('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/cmihbn_updated/cmi_td_actual_ages.npz')
actual_cmi_age = actual_cmi['actual']

predicted_cmi = np.load('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/cmihbn_updated/cmi_td_predicted_ages.npz')
predicted_cmi_age = predicted_cmi['predicted']

actual_adhd200 = np.load('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/adhd200_updated/adhd200_td_actual_ages.npz')
actual_adhd200_age = actual_adhd200['actual']

predicted_adhd200 = np.load('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/adhd200_updated/adhd200_td_predicted_ages.npz')
predicted_adhd200_age = predicted_adhd200['predicted']

actual_ages = np.concatenate((actual_nki_age,actual_cmi_age,actual_adhd200_age))
predicted_ages = np.concatenate((predicted_nki_age,predicted_cmi_age,predicted_adhd200_age))

print(actual_ages)
print(predicted_ages)

print(actual_ages.shape)
print(predicted_ages.shape)

print(scipy.stats.pearsonr(actual_ages,predicted_ages))

BAG = (predicted_ages - actual_ages).reshape(-1,1)
lin_model = LinearRegression().fit(actual_ages.reshape(-1,1),BAG)

coef_list = []
intercept_list = []

coef_list.append(lin_model.coef_[0][0])
intercept_list.append(lin_model.intercept_[0])

print(coef_list)
print(intercept_list)

pdb.set_trace()
np.savez('lin_params_all_external_td',coef=coef_list,intercept=intercept_list)

