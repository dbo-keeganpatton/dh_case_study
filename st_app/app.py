import sys
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
sys.path.append("./components/visuals/")
sys.path.append("./components/data_sets/")
from bx_plots import create_box_plot
st.set_page_config(layout="wide")


#########################################
#        s1. Data Exploration           #
########################################
st.markdown('''
# :blue[Data Exploration]
##### We start by identifying data features that can be used to distinguish customer Loyalty.
1. The :blue[**Frequency**] of visits
2. How :blue[**Recently**] each customer visited
3. The amount that each customer has :blue[**Spent**].
''')

st.header('Box Plot')
field_list = ['total_spend', 'visits', 'time_from_last_visit']
with st.container():
    cols = st.columns(3)
    for item, col in zip(field_list, cols):
        with col:
           create_box_plot(item)


###############################
#       Remove Outliers       #
##############################
z_score_df = rfm[['total_spend', 'visits', 'time_from_last_visit']]
z_score = np.abs(stats.zscore(z_score_df))
remove_outliers = (z_score < 2).all(axis=1) 
rfm_outliers_removed = z_score_df[remove_outliers]


##############################
#       Normalize Data       #
##############################
st.subheader('Standardized Dataset')
rfm_outliers_removed.drop_duplicates()
scalar = StandardScaler().fit(rfm_outliers_removed.values)
features = scalar.transform(rfm_outliers_removed.values)
scaled_features = pd.DataFrame(features, columns=['total_spend', 'visits', 'time_from_last_visit'])
st.write(scaled_features)


#############################
#  Determine # of Clusters  #
#############################
st.write('Inflection Point indicated at the elbow which lies at 3 clusters')
SSE = []
for cluster in range(1, 10):
    kmeans = KMeans(n_clusters=cluster, init='k-means++')
    kmeans.fit(scaled_features)
    SSE.append(kmeans.inertia_)

frame = pd.DataFrame({'Cluster': range(1, 10), 'SSE': SSE})
st.line_chart(frame, x='Cluster', y='SSE')


#############################
#   Create Kmeans Cluster   #
#############################
kmeans = KMeans(n_clusters=3, init='k-means++')
kmeans.fit(scaled_features)
s_score = silhouette_score(scaled_features, kmeans.labels_, metric='euclidean')
st.metric(label='Silhouette Score', value=s_score)

pred = kmeans.predict(scaled_features)
frame = pd.DataFrame(rfm_outliers_removed)
frame['cluster'] = pred


st.subheader("""We observe that customers in Cluster 0 have a high spend, frequently visit, and have patronized our store recently.
""")
plot_cluster_df = frame.groupby(['cluster'], as_index=False).mean()
with st.container():
    cols = st.columns(3)
    for item, col in zip(field_list, cols):
        with col:
            st.bar_chart(data=plot_cluster_df, x='cluster', y=item, color='cluster')
st.divider()


