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

predicted_abide = np.load('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/feature_attribution/predicted_abide_asd_ados_total_all_regions_dev_model.npz')
actual_abide = np.load('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/feature_attribution/actual_abide_asd_ados_total_all_regions_dev_model.npz')

predicted_ages = np.squeeze(predicted_abide['output'])
actual_ages = np.squeeze(actual_abide['output'])
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
ax1.set_xlabel("Observed ADOS Total",fontsize=16)
ax1.set_ylabel("Predicted ADOS Total",fontsize=16)
ax1.set_title("ABIDE",fontsize=16)

predicted_stanford = np.load('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/feature_attribution/predicted_stanford_asd_SRS_all_regions_dev_model.npz')
actual_stanford = np.load('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/feature_attribution/actual_stanford_asd_SRS_total_all_regions_dev_model.npz')

predicted_ages = np.squeeze(predicted_stanford['output'])
actual_ages = np.squeeze(actual_stanford['output'])

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
ax2.set_xlabel("Observed SRS",fontsize=16)
ax2.set_ylabel("Predicted SRS",fontsize=16)
ax2.set_title("Stanford",fontsize=16)

plt.savefig('abide_stanford_asd_feature_clinical_scatter.png',format='png')
#plt.savefig('hcp_dev_nki_cmi_adhd200_td_brain_age_scatter.svg',format='svg')
