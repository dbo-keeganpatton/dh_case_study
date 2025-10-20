import sys
import pandas as pd
import streamlit as st
sys.path.append("../data_sets/")
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from rfm_outlier_removed import clean_rfm_data_outliers



def create_cluster_line_plot():

    data = clean_rfm_data_outliers()
    scalar = StandardScaler().fit(data.values)
    features = scalar.transform(data.values)
    scaled_features = pd.DataFrame(
        features,
        columns=['total_spend', 'visits', 'time_from_last_visit']
    )


    kmeans = KMeans(n_clusters=3, init='k-means++')
    kmeans.fit(scaled_features)
    s_score = silhouette_score(scaled_features, kmeans.labels_, metric='euclidean')

    return st.metric(label='Silhouette Score', value=s_score)


