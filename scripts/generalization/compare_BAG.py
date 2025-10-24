import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, pearsonr
import pdb

adhd200_predicted = np.load('adhd200_updated/predicted_adhd200_ages_most_updated.npz')
predicted_ages = adhd200_predicted['predicted']
adhd200_actual = np.load('adhd200_updated/actual_adhd200_ages_most_updated.npz')
actual_ages = adhd200_actual['actual']
adhd200_bag = predicted_ages - actual_ages

cmi_predicted = np.load('cmihbn_updated/predicted_cmihbn_adhd_ages_most_updated.npz')
predicted_ages = cmi_predicted['predicted']
cmi_actual = np.load('cmihbn_updated/actual_cmihbn_adhd_ages_most_updated.npz')
actual_ages = cmi_actual['actual']
cmi_adhd_bag = predicted_ages - actual_ages
r,p = pearsonr(actual_ages,predicted_ages)

cmi_td_predicted = np.load('cmihbn_updated/predicted_cmihbn_td_ages.npz')
predicted_ages = cmi_td_predicted['predicted']
cmi_td_actual = np.load('cmihbn_updated/actual_cmihbn_td_ages.npz')
actual_ages = cmi_td_actual['actual']
cmi_td_bag = predicted_ages - actual_ages
r,p = pearsonr(actual_ages,predicted_ages)

adhd200_td_predicted = np.load('adhd200_updated/adhd200_td_predicted_ages.npz')
predicted_td_ages = adhd200_td_predicted['predicted']
adhd200_td_actual = np.load('adhd200_updated/adhd200_td_actual_ages.npz')
actual_td_ages = adhd200_td_actual['actual']
adhd200_td_bag = predicted_td_ages - actual_td_ages

abide_predicted = np.load('asd_updated/predicted_abide_asd_ages_most_updated.npz')
predicted_ages = abide_predicted['predicted']
abide_actual = np.load('asd_updated/actual_abide_asd_ages_most_updated.npz')
actual_ages = abide_actual['actual']
abide_bag = predicted_ages - actual_ages

abide_td_predicted = np.load('asd_updated/predicted_abide_td_ages_most_updated.npz')
predicted_ages = abide_td_predicted['predicted']
abide_td_actual = np.load('asd_updated/actual_abide_td_ages_most_updated.npz')
actual_ages = abide_td_actual['actual']
abide_td_bag = predicted_ages - actual_ages

stanford_predicted = np.load('asd_updated/predicted_stanford_asd_ages_most_updated.npz')
predicted_ages = stanford_predicted['predicted']
stanford_actual = np.load('asd_updated/actual_stanford_asd_ages_most_updated.npz')
actual_ages = stanford_actual['actual']
stanford_bag = predicted_ages - actual_ages

stanford_td_predicted = np.load('asd_updated/predicted_stanford_td_ages_most_updated.npz')
predicted_ages = stanford_td_predicted['predicted']
stanford_td_actual = np.load('asd_updated/actual_stanford_td_ages_most_updated.npz')
actual_ages = stanford_td_actual['actual']
stanford_td_bag = predicted_ages - actual_ages

'''
stanford_predicted = np.load('stanford_asd_predicted_ages.npz')
predicted_ages = stanford_predicted['predicted']
stanford_actual = np.load('stanford_asd_actual_ages.npz')
actual_ages = stanford_actual['actual']
stanford_bag = predicted_ages - actual_ages
'''

'''
r,p = pearsonr(actual_ages,predicted_ages)
print(r)
print(p)
pdb.set_trace()
'''

adhd_pooled_bag = np.concatenate((adhd200_bag,cmi_adhd_bag))
asd_pooled_bag = np.concatenate((abide_bag,stanford_bag))
#print(np.mean(cmi_td_bag))
#print(np.mean(cmi_adhd_bag))
#print(np.mean(cmi_adhd_bag))
#print(np.mean(cmi_td_bag))
print('ASD BAG')
print(np.mean(asd_pooled_bag))
print('ADHD BAG')
print(np.mean(adhd_pooled_bag))
#print(np.mean(stanford_bag))
#print(np.std(stanford_bag))
#print('TD BAG')
#print(np.mean(stanford_td_bag))
#print(np.std(stanford_td_bag))

print('Significant Difference?')
t_stat, p_val = ttest_ind(adhd_pooled_bag,asd_pooled_bag)
print(t_stat)
print(p_val)

