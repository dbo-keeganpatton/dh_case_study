import sys
import pandas as pd
import streamlit as st
sys.path.append("../data_sets/")
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from rfm_outlier_removed import clean_rfm_data_outliers



def create_kmeans_bar(col):

    data = clean_rfm_data_outliers()
    scalar = StandardScaler().fit(data.values)
    features = scalar.transform(data.values)
    scaled_features = pd.DataFrame(
        features,
        columns=['total_spend', 'visits', 'time_from_last_visit']
    )

    kmeans = KMeans(n_clusters=3, init='k-means++')
    kmeans.fit(scaled_features)
    pred = kmeans.predict(scaled_features)
    frame = pd.DataFrame(data)
    frame['cluster'] = pred

    plot_cluster_df = frame.groupby(['cluster'], as_index=False).mean()

    return st.bar_chart(
        data=plot_cluster_df,
        x='cluster',
        y=col,
        color='cluster'
    )



