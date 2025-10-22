# This file has no purpose in the main flow of the app
# it is simply here to take the clustered output and produce a csv which I can use
# to add to the SQLite database for further analysis.


import os
import sys
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
sys.path.append("../data_sets/")
from rfm import rfm_data 

rfm = rfm_data()

# --- Remove outliers
z_score_df = rfm[['total_spend', 'visits', 'time_from_last_visit']]
z_score = np.abs(stats.zscore(z_score_df))
remove_outliers = (z_score < 2).all(axis=1)
rfm_outliers_removed = rfm[remove_outliers].drop_duplicates()

# --- Scale numeric features only (exclude prsn_id)
features = rfm_outliers_removed[['total_spend', 'visits', 'time_from_last_visit']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# --- Run KMeans
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
rfm_outliers_removed['cluster'] = kmeans.fit_predict(scaled_features)

# --- Keep only ID + cluster assignments
clustered_output = rfm_outliers_removed[['prsn_id', 'cluster']]

# --- Save to CSV
clustered_output.to_csv("./clustered_data.csv", index=False)


