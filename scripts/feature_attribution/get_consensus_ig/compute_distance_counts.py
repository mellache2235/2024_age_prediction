import pandas as pd
import numpy as np
import math
import random
import pdb
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

feature_map_nki = pd.read_csv('consensus_features_abide_asd_top20.csv')
feature_map_dev = pd.read_csv('consensus_features_adhd200_top20.csv')

feature_map_nki = feature_map_nki.loc[:,['RegionID','Count']]
feature_map_dev = feature_map_dev.loc[:,['RegionID','Count']]

df_combined = pd.merge(feature_map_nki, feature_map_dev, on='RegionID', how='outer', suffixes=('_Map1', '_Map2'))

df_combined = df_combined.dropna()
df_combined = df_combined.sort_values('RegionID')

print(df_combined)

similarity = cosine_similarity([np.asarray(df_combined['Count_Map1'])], [np.asarray(df_combined['Count_Map2'])])[0, 0]
print(f"Cosine Similarity: {similarity:.4f}")

print(similarity)

