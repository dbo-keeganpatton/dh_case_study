import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
conn = st.connection('case_study_db', type='sql')


st.set_page_config(layout="wide")
# Just output the consolidated data
st.subheader("Data Shape")
st.write(conn.query("select * from transactions").head())


# The next sections will look at how recent we saw a visit from each customer
# How often we see them return, and their overall spend value for creating
# groupings
rfm_df = conn.query(
    """
    select distinct 
    prsn_id,
    item_qty,
    transaction_code,
    net_spend_amt,
    date_id
    from transactions
    """)

################################
#           Recency            #
###############################
st.subheader("Customer Recency")
rfm_df['visit_rank'] = rfm_df.sort_values(['prsn_id', 'date_id']).groupby(['prsn_id'])['date_id'].rank(method='min').astype(int)
customer_last_visit_df = rfm_df[rfm_df['visit_rank']==1]
customer_last_visit_df['date_id'] = pd.to_datetime(customer_last_visit_df['date_id'])
customer_last_visit_df['time_from_last_visit'] = (
    customer_last_visit_df['date_id'] - min(customer_last_visit_df['date_id']) 
).dt.days
st.write(customer_last_visit_df.head())
st.divider()


################################
#           Frequency          #
###############################
st.subheader("Customer Freqency")
customer_frequency_df = rfm_df.groupby('prsn_id')['transaction_code'].nunique().reset_index()
customer_frequency_df.columns = ['prsn_id', 'visits']
st.write(customer_frequency_df.head())
st.divider()


################################
#           Spend             #
###############################
st.subheader("Customer Spend")
rfm_df['net_spend_amt'] = rfm_df['net_spend_amt'].astype(float)
customer_spend_df = rfm_df.groupby('prsn_id')['net_spend_amt'].sum().reset_index()
customer_spend_df.columns = ['prsn_id', 'total_spend']
st.write(customer_spend_df.head())
st.divider()


################################
#           RFM DF             #
###############################
st.subheader("Customer RFM Profile")
rf = customer_last_visit_df.merge(customer_frequency_df, on='prsn_id')
rfm = rf.merge(customer_spend_df, on='prsn_id')
rfm = rfm[['prsn_id', 'total_spend', 'visits', 'time_from_last_visit']]
st.write(rfm.head())
st.divider()


###############################
#     Distribution Viz        #
##############################
st.subheader("Data Distribution")
field_list = ['total_spend', 'visits', 'time_from_last_visit']
with st.container():
    cols = st.columns(3)
    for item, col in zip(field_list, cols):
        with col:
            fig, ax = plt.subplots()
            ax.boxplot(rfm[item], vert=False) 
            ax.set_title(item)
            st.pyplot(fig)   
st.divider()


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
st.write(frame)

plot_cluster_df = frame.groupby(['cluster'], as_index=False).mean()
with st.container():
    cols = st.columns(3)
    for item, col in zip(field_list, cols):
        with col:
            st.bar_chart(data=plot_cluster_df, x='cluster', y=item)
st.divider()


