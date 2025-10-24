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
from sklearn.linear_model import LinearRegression
import matplotlib.backends.backend_pdf as pdf
from matplotlib import font_manager

font_path = '/oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/clustering_analysis/arial.ttf'
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()

predicted = np.load('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/plotting/predicted_hcp_dev_ages_most_updated.npz')
actual = np.load('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/plotting/actual_hcp_dev_ages_most_updated.npz')

predicted_ages = np.squeeze(predicted['predicted'])
actual_ages = np.squeeze(actual['actual'])

#BAG = (predicted_ages - actual_ages).reshape(-1,1)

#lin_model = LinearRegression().fit(actual_ages.reshape(-1,1),BAG)

#Offset = lin_model.coef_[0][0] * actual_ages + lin_model.intercept_[0]

#predicted_ages = predicted_ages - Offset

r,p = scipy.stats.pearsonr(x=actual_ages,y=predicted_ages)
mae = np.mean(np.abs(predicted_ages-actual_ages))

fig,axes = plt.subplots(2,2,figsize=(10,10),constrained_layout=True,dpi=300)
r_squared = r ** 2

# Conditional formatting for p-value:
if p < 0.001:
    p_text = r"$\mathit{P} < 0.001$"
else:
    p_text = rf"$\mathit{{P}} = {p:.3f}$"

sns.set_style("white")
sns.regplot(x=actual_ages, y=predicted_ages, ci=None,
           scatter_kws={'color': 'navy', 'alpha' : 0.6,'s': 40,'edgecolor': 'w', 'linewidth':0.5},
           line_kws={'color': 'red', 'linewidth' : 2},ax=axes[0,0])

lims = [actual_ages.min()-1, actual_ages.max()+2]
#axes[0,0].plot(lims, lims, linestyle='--', color='gray', linewidth=1.2, label='Identity line')
axes[0,0].set_xlim(5,23)
axes[0,0].set_ylim(5,23)
axes[0,0].set_xticks(np.arange(5, 24, 2))
axes[0,0].set_yticks(np.arange(5, 24, 2))

# Final annotation with proper italicization and upright formatting:
axes[0,0].text(0.95, 0.05,
        f"$\mathit{{R}}^2$ = {r_squared:.3f}\n"
        f"{p_text}\n"
        f"$\mathrm{{MAE}} = {mae:.2f}\;\mathrm{{years}}$",
        transform=axes[0,0].transAxes,
        horizontalalignment='right',
        verticalalignment='bottom',
        fontsize=11)

axes[0,0].set_xlabel("Chronological Age",fontsize=15,labelpad=10)
axes[0,0].set_ylabel("Brain Age",fontsize=15,labelpad=10)
axes[0,0].xaxis.set_ticks_position('bottom')
axes[0,0].yaxis.set_ticks_position('left')
axes[0,0].tick_params(axis='both', which='major', length=6, width=1)
for spine in ['bottom', 'left']:
    axes[0,0].spines[spine].set_linewidth(1.5)
axes[0,0].spines[['right', 'top']].set_visible(False)
axes[0,0].set_title("HCP-Development",fontsize=15,pad=10)
#plt.tight_layout(pad=1.2)
#plt.savefig('dev_brain_age_scatter_high_quality.png',format='png')


##### NKI_RS

predicted_nki = np.load('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/nki_updated/predicted_nki_ages.npz')
actual_nki = np.load('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/nki_updated/actual_nki_ages.npz')

predicted_ages = np.squeeze(predicted_nki['predicted'])
actual_ages = np.squeeze(actual_nki['actual'])


BAG = (predicted_ages - actual_ages).reshape(-1,1)

lin_model = LinearRegression().fit(actual_ages.reshape(-1,1),BAG)

Offset = lin_model.coef_[0][0] * actual_ages + lin_model.intercept_[0]

predicted_ages = predicted_ages - Offset

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
           line_kws={'color': 'red', 'linewidth' : 2},ax=axes[0,1])

lims = [actual_ages.min()-1, actual_ages.max()+2]
#axes[0,1].plot(lims, lims, linestyle='--', color='gray', linewidth=1.2, label='Identity line')
axes[0,1].set_xlim(5,23)
axes[0,1].set_ylim(5,23)
axes[0,1].set_xticks(np.arange(5, 24, 2))
axes[0,1].set_yticks(np.arange(5, 24, 2))

# Final annotation with proper italicization and upright formatting:
axes[0,1].text(0.95, 0.05,
        f"$\mathit{{R}}^2$ = {r_squared:.3f}\n"
        f"{p_text}\n"
        f"$\mathrm{{MAE}} = {mae:.2f}\;\mathrm{{years}}$",
        transform=axes[0,1].transAxes,
        horizontalalignment='right',
        verticalalignment='bottom',
        fontsize=11)

axes[0,1].set_xlabel("Chronological Age",fontsize=15,labelpad=10)
axes[0,1].set_ylabel("Brain Age",fontsize=15,labelpad=10)
axes[0,1].xaxis.set_ticks_position('bottom')
axes[0,1].yaxis.set_ticks_position('left')
axes[0,1].tick_params(axis='both', which='major', length=6, width=1)
for spine in ['bottom', 'left']:
    axes[0,1].spines[spine].set_linewidth(1.5)
axes[0,1].spines[['right', 'top']].set_visible(False)
axes[0,1].set_title("NKI-RS",fontsize=15,pad=10)

### CMI-HBN
predicted_cmi = np.load('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/cmihbn_updated/predicted_cmihbn_td_ages.npz')
actual_cmi = np.load('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/cmihbn_updated/actual_cmihbn_td_ages.npz')

predicted_ages = np.squeeze(predicted_cmi['predicted'])
actual_ages = np.squeeze(actual_cmi['actual'])


BAG = (predicted_ages - actual_ages).reshape(-1,1)

lin_model = LinearRegression().fit(actual_ages.reshape(-1,1),BAG)

Offset = lin_model.coef_[0][0] * actual_ages + lin_model.intercept_[0]

predicted_ages = predicted_ages - Offset

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
           line_kws={'color': 'red', 'linewidth' : 2},ax=axes[1,0])

lims = [actual_ages.min()-1, actual_ages.max()+2]
#axes[1,0].plot(lims, lims, linestyle='--', color='gray', linewidth=1.2, label='Identity line')
axes[1,0].set_xlim(5,23)
axes[1,0].set_ylim(5,23)
axes[1,0].set_xticks(np.arange(5, 24, 2))
axes[1,0].set_yticks(np.arange(5, 24, 2))

# Final annotation with proper italicization and upright formatting:
axes[1,0].text(0.95, 0.05,
        f"$\mathit{{R}}^2$ = {r_squared:.3f}\n"
        f"{p_text}\n"
        f"$\mathrm{{MAE}} = {mae:.2f}\;\mathrm{{years}}$",
        transform=axes[1,0].transAxes,
        horizontalalignment='right',
        verticalalignment='bottom',
        fontsize=11)

axes[1,0].set_xlabel("Chronological Age",fontsize=15,labelpad=10)
axes[1,0].set_ylabel("Brain Age",fontsize=15,labelpad=10)
axes[1,0].xaxis.set_ticks_position('bottom')
axes[1,0].yaxis.set_ticks_position('left')
axes[1,0].tick_params(axis='both', which='major', length=6, width=1)
for spine in ['bottom', 'left']:
    axes[1,0].spines[spine].set_linewidth(1.5)
axes[1,0].spines[['right', 'top']].set_visible(False)
axes[1,0].set_title("CMI-HBN",fontsize=15,pad=10)

### ADHD-200
predicted_adhd200 = np.load('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/adhd200_updated/predicted_adhd200_td_ages.npz')
actual_adhd200 = np.load('/oak/stanford/groups/menon/projects/mellache/2024_age_prediction/scripts/generalization/adhd200_updated/actual_adhd200_td_ages.npz')

predicted_ages = np.squeeze(predicted_adhd200['predicted'])
actual_ages = np.squeeze(actual_adhd200['actual'])


BAG = (predicted_ages - actual_ages).reshape(-1,1)

lin_model = LinearRegression().fit(actual_ages.reshape(-1,1),BAG)

Offset = lin_model.coef_[0][0] * actual_ages + lin_model.intercept_[0]

predicted_ages = predicted_ages - Offset

r,p = scipy.stats.pearsonr(x=actual_ages,y=predicted_ages)
r_squared = r ** 2
mae = np.mean(np.abs(predicted_ages-actual_ages))

if p < 0.001:
    p_text = r"$\mathit{P} < 0.001$"
else:
    p_text = rf"$\mathit{{P}} = {p:.3f}$"

sns.set_style("white")
sns.regplot(x=actual_ages, y=predicted_ages, ci=None,
           scatter_kws={'color': 'navy', 'alpha' : 0.6,'s': 40,'edgecolor': 'w', 'linewidth':0.5},
           line_kws={'color': 'red', 'linewidth' : 2},ax=axes[1,1])

#lims = [actual_ages.min()-1, actual_ages.max()+2]
#axes[1,1].plot(lims, lims, linestyle='--', color='gray', linewidth=1.2, label='Identity line')
axes[1,1].set_xlim(5,23)
axes[1,1].set_ylim(5,23)
axes[1,1].set_xticks(np.arange(5, 24, 2))
axes[1,1].set_yticks(np.arange(5, 24, 2))

# Final annotation with proper italicization and upright formatting:
axes[1,1].text(0.95, 0.05,
        f"$\mathit{{R}}^2$ = {r_squared:.3f}\n"
        f"{p_text}\n"
        f"$\mathrm{{MAE}} = {mae:.2f}\;\mathrm{{years}}$",
        transform=axes[1,1].transAxes,
        horizontalalignment='right',
        verticalalignment='bottom',
        fontsize=11)

axes[1,1].set_xlabel("Chronological Age",fontsize=15,labelpad=10)
axes[1,1].set_ylabel("Brain Age",fontsize=15,labelpad=10)
axes[1,1].xaxis.set_ticks_position('bottom')
axes[1,1].yaxis.set_ticks_position('left')
axes[1,1].tick_params(axis='both', which='major', length=6, width=1)
for spine in ['bottom', 'left']:
    axes[1,1].spines[spine].set_linewidth(1.5)
axes[1,1].spines[['right', 'top']].set_visible(False)
axes[1,1].set_title("ADHD-200",fontsize=15,pad=10)

plt.show()
plt.savefig('td_brain_age_prediction_figure.png',format='png')
#plt.savefig('hcp_dev_nki_cmi_adhd200_td_brain_age_scatter.svg',format='svg')
pdf.FigureCanvas(fig).print_pdf('td_brain_age_prediction_figure.ai')
