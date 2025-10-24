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
from scipy.stats import ttest_ind
import matplotlib.backends.backend_pdf as pdf
from matplotlib import font_manager

font_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/clustering_analysis/arial.ttf'
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()

predicted = np.load('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/asd_updated/predicted_abide_asd_ages_most_updated.npz')
actual = np.load('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/asd_updated/actual_abide_asd_ages_most_updated.npz')

predicted_ages = np.squeeze(predicted['predicted'])
actual_ages = np.squeeze(actual['actual'])
print(np.mean(predicted_ages-actual_ages))
#pdb.set_trace()

r,p = scipy.stats.pearsonr(x=actual_ages,y=predicted_ages)
mae = np.mean(np.abs(predicted_ages-actual_ages))

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5),constrained_layout=True,dpi=300)
r_squared = r ** 2

# Conditional formatting for p-value:
if p < 0.001:
    p_text = r"$\mathit{P} < 0.001$"
else:
    p_text = rf"$\mathit{{P}} = {p:.3f}$"

sns.set_style("white")
sns.regplot(x=actual_ages, y=predicted_ages, ci=None,
           scatter_kws={'color': 'navy', 'alpha' : 0.6,'s': 40,'edgecolor': 'w', 'linewidth':0.5},
           line_kws={'color': 'red', 'linewidth' : 2},ax=ax1)

lims = [actual_ages.min()-1, actual_ages.max()+2]
#ax1.plot(lims, lims, linestyle='--', color='gray', linewidth=1.2, label='Identity line')
ax1.set_xlim(5,22)
ax1.set_ylim(5,22)
ax1.set_xticks(np.arange(5, 23, 2))
ax1.set_yticks(np.arange(5, 23, 2))

# Final annotation with proper italicization and upright formatting:
ax1.text(0.95, 0.05,
        f"$\mathit{{R}}^2$ = {r_squared:.3f}\n"
        f"{p_text}\n"
        f"$\mathrm{{MAE}} = {mae:.2f}\;\mathrm{{years}}$",
        transform=ax1.transAxes,
        horizontalalignment='right',
        verticalalignment='bottom',
        fontsize=11)

ax1.set_xlabel("Chronological Age",fontsize=15,labelpad=10)
ax1.set_ylabel("Brain Age",fontsize=15,labelpad=10)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')
ax1.tick_params(axis='both', which='major', length=6, width=1)
for spine in ['bottom', 'left']:
    ax1.spines[spine].set_linewidth(1.5)
ax1.spines[['right', 'top']].set_visible(False)
ax1.set_title("ABIDE",fontsize=15,pad=10)
#plt.tight_layout(pad=1.2)
#plt.savefig('dev_brain_age_scatter_high_quality.png',format='png')


##### ADHD-200

predicted_adhd200 = np.load('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/asd_updated/predicted_stanford_asd_ages_most_updated.npz')
actual_adhd200 = np.load('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/asd_updated/actual_stanford_asd_ages_most_updated.npz')

predicted_ages = np.squeeze(predicted_adhd200['predicted'])
actual_ages = np.squeeze(actual_adhd200['actual'])
print(np.mean(predicted_ages-actual_ages))
pdb.set_trace()

r,p = scipy.stats.pearsonr(x=actual_ages,y=predicted_ages)
r_squared = r ** 2
mae = np.mean(np.abs(predicted_ages-actual_ages))
# Conditional formatting for p-value:
if p < 0.001:
    p_text = r"$\mathit{P} < 0.001$"
else:
    p_text = rf"$\mathit{{P}} = {p:.3f}$"

sns.set_style("white")
sns.regplot(x=actual_ages, y=predicted_ages, ci=None,
           scatter_kws={'color': 'navy', 'alpha' : 0.6,'s': 40,'edgecolor': 'w', 'linewidth':0.5},
           line_kws={'color': 'red', 'linewidth' : 2},ax=ax2)

lims = [actual_ages.min()-1, actual_ages.max()+2]
#ax2.plot(lims, lims, linestyle='--', color='gray', linewidth=1.2, label='Identity line')
ax2.set_xlim(5,22)
ax2.set_ylim(5,22)
ax2.set_xticks(np.arange(5, 23, 2))
ax2.set_yticks(np.arange(5, 23, 2))

# Final annotation with proper italicization and upright formatting:
ax2.text(0.95, 0.05,
        f"$\mathit{{R}}^2$ = {r_squared:.3f}\n"
        f"{p_text}\n"
        f"$\mathrm{{MAE}} = {mae:.2f}\;\mathrm{{years}}$",
        transform=ax2.transAxes,
        horizontalalignment='right',
        verticalalignment='bottom',
        fontsize=11)

ax2.set_xlabel("Chronological Age",fontsize=15,labelpad=10)
ax2.set_ylabel("Brain Age",fontsize=15,labelpad=10)
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')
ax2.tick_params(axis='both', which='major', length=6, width=1)
for spine in ['bottom', 'left']:
    ax2.spines[spine].set_linewidth(1.5)
ax2.spines[['right', 'top']].set_visible(False)
ax2.set_title("Stanford",fontsize=15,pad=10)

plt.savefig('asd_brain_age_prediction_figure.png',format='png')
pdf.FigureCanvas(fig).print_pdf('asd_brain_age_prediction_figure.ai')