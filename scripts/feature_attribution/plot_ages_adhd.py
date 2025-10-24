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


fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5),constrained_layout=True)

predicted_adhd200 = np.load('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/feature_attribution/predicted_adhd200_HY_all_regions_dev_model.npz')
actual_adhd200 = np.load('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/feature_attribution/actual_adhd200_HY_all_regions_dev_model.npz')

predicted_ages = np.squeeze(predicted_adhd200['output'])
actual_ages = np.squeeze(actual_adhd200['output'])
print(predicted_ages)
print(actual_ages)

r,p = scipy.stats.spearmanr(actual_ages,predicted_ages)
print(r)
#print(mean_absolute_error(actual_ages,predicted_ages))

sns.regplot(x=actual_ages, y=predicted_ages, ci=None,
           scatter_kws={'color': 'blue', 's': 50},
           line_kws={'color': 'red'},ax=ax1)
ax1.text(0.95, 0.05, f"$r$ = {r:.2f}\np = {p:.3f}",
         transform=ax1.transAxes,  # Use axis coordinates (0 to 1)
         horizontalalignment='right',  # Align text to the right at x=0.95
         verticalalignment='bottom',   # Align text to the bottom at y=0.05
         fontsize=12)

ax1.spines[['right', 'top']].set_visible(False)
ax1.set_xlabel("Observed Hyperactivity",fontsize=16)
ax1.set_ylabel("Predicted Hyperactivity",fontsize=16)
ax1.set_title("ADHD200",fontsize=16)

predicted_cmi = np.load('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/feature_attribution/predicted_adhd_cmihbn_HY_all_regions_dev_model.npz')
actual_cmi = np.load('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/feature_attribution/actual_adhd_cmihbn_HY_all_regions_dev_model.npz')

predicted_ages = np.squeeze(predicted_cmi['output'])
actual_ages = np.squeeze(actual_cmi['output'])

#print(mean_absolute_error(actual_ages,predicted_ages))

r,p = scipy.stats.spearmanr(actual_ages,predicted_ages)
print(r)
#pdb.set_trace()

sns.regplot(x=actual_ages, y=predicted_ages, ci=None,
           scatter_kws={'color': 'blue', 's': 50},
           line_kws={'color': 'red'},ax=ax2)
ax2.text(0.95, 0.05, f"$r$ = {r:.2f}\np = {p:.3f}",
         transform=ax2.transAxes,  # Use axis coordinates (0 to 1)
         horizontalalignment='right',  # Align text to the right at x=0.95
         verticalalignment='bottom',   # Align text to the bottom at y=0.05
         fontsize=12)
ax2.spines[['right', 'top']].set_visible(False)
ax2.set_xlabel("Observed Hyperactivity",fontsize=16)
ax2.set_ylabel("Predicted Hyperactivity",fontsize=16)
ax2.set_title("CMI-HBN",fontsize=16)

plt.savefig('cmi_adhd200_adhd_feature_HY_scatter.png',format='png')
#plt.savefig('hcp_dev_nki_cmi_adhd200_td_brain_age_scatter.svg',format='svg')
