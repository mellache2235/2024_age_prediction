import pandas as pd
import numpy as np
import random
import math
import umap
import pdb
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

mean_dev = np.load('features_hcp_dev_hcp_dev_model.npz')
dev_features = mean_dev['features']
mean_dev = dev_features.mean(axis=0)
feature_importance = np.abs(mean_dev)
top_k_percent = 0.2
num_top_features = int(len(feature_importance) * top_k_percent)
top_feature_indices = np.argsort(feature_importance)[-num_top_features:]

adhd_ig = np.load('features_adhd200_adhd_hcp_dev_model.npz')
adhd_features = adhd_ig['features']

td_ig = np.load('features_adhd200_td_hcp_dev_model.npz')
td_features = td_ig['features']

deviation_td = td_features[:,top_feature_indices] - mean_dev[top_feature_indices]
deviation_adhd = adhd_features[:,top_feature_indices] - mean_dev[top_feature_indices]

deviations = np.vstack([deviation_td, deviation_adhd])
labels = np.array(["TD"] * deviation_td.shape[0] + ["ADHD"] * deviation_adhd.shape[0])
print(labels.shape)
pdb.set_trace()

reducer = umap.UMAP(random_state=42,n_components=2)

deviations = deviations.reshape((deviations.shape[0],-1))

embeddings = reducer.fit_transform(deviations)

fig,ax1 = plt.subplots()

sns.scatterplot(embeddings[:,0],embeddings[:,1],hue=labels,palette=['deepskyblue','salmon'],ax=ax1)
ax1.xaxis.set_major_locator(ticker.NullLocator())
ax1.xaxis.set_minor_locator(ticker.NullLocator())
ax1.yaxis.set_major_locator(ticker.NullLocator())
ax1.yaxis.set_minor_locator(ticker.NullLocator())

ax1.set_xlabel('UMAP-1',fontsize=16,fontweight='bold')
ax1.set_ylabel('UMAP-2',fontsize=16,fontweight='bold')

L = ax1.legend(loc="lower right")
L.get_texts()[0].set_text('TD')
L.get_texts()[1].set_text('ADHD')
L.get_frame().set_visible(False)
ax1.spines[['right', 'top']].set_visible(False)
ax1.set_title("Brain Fingerprints",fontsize=16)

plt.savefig('td_adhd_top20_fingerprint_umap.png')

