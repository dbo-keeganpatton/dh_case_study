import sys
import numpy as np
import pandas as pd
import streamlit as st
sys.path.append("./components/visuals/")
sys.path.append("./components/data_sets/")
from bx_plots import create_box_plot
from hg_plots import create_histogram
from cln_bx_plot import create_clean_box_plot
from cluster_line import create_cluster_line_plot 
from cluster_bars import create_kmeans_bar
st.set_page_config(layout="wide")



#########################################
#            Reusable Vars              #
#########################################
feature_list = ['total_spend', 'visits', 'time_from_last_visit']



#########################################
#        s1. Data Exploration           #
#########################################
st.markdown('''# :blue[Data Exploration & Feature Selection]''')
with st.container(border=True):
    st.markdown(
        '''##### We start by identifying data features that can be used to distinguish customer Loyalty, and viewing their distribution in general.
        ''')
    st.markdown(
        '''
        1. :blue[**Recency**] of last visit for each customer.
        2. :blue[**Frequency**] of visits.
        3. :blue[**Total Spent**] by each customer.
        '''
    )
    st.markdown('''##### Overall, we observe that this data is heavily :blue[left skewed], indicating not only an abnormal distribution, but also the presence of several outliers that will affect our clustering further on.''')

    with st.container():
        cols = st.columns(3)
        for item, col in zip(feature_list, cols):
            with col:
               create_histogram(item)
                
st.divider()



#########################################
#        s2. Outlier Detection          #
########################################
st.markdown("# :blue[Outlier Distribution]")
with st.container(border=True):
    st.markdown('''
    ##### A classic box plot can help us confirm the presence of numerous outliers. From this we can see that each feature will need to be sanitized of outliers.
    ''')

    with st.container():
        cols = st.columns(3)
        for item, col in zip(feature_list, cols):
            with col:
               create_box_plot(item)

st.divider()



#########################################
#         s3. Outlier Removal           #
#########################################
st.markdown('''# :blue[Outlier Removal]''')
with st.container(border=True):

    st.markdown('''
    ##### We can determine the threshold by which we determine a value to be an outlier by calculating each value's :blue[Z-Score], which calculates the number of standard deviations each value is from the average of the dataset.
    ''')
    st.markdown('''
    ##### For our test here, we will remove all data that is more than a single standard deviation from our mean values.
    ''')

    st.code(
        body='''
            import numpy as np
            from scipy import stats

            z_score_df = rfm[['total_spend', 'visits', 'time_from_last_visit']]
            z_score = np.abs(stats.zscore(z_score_df))
            remove_outliers = (z_score < 2).all(axis=1) 
            rfm_outliers_removed = z_score_df[remove_outliers]''',
        language='python'
    )

    st.markdown('''##### We can see that that this has dramatically widened the :blue[IQR] for our plot, improving our dataset's quality.''')
    with st.container():
        cols = st.columns(3)
        for item, col in zip(feature_list, cols):
            with col:
               create_clean_box_plot(item)

st.divider()



#########################################
#         s4. Cluster Testing           #
#########################################
st.markdown('''# :blue[Cluster Testing]''')
with st.container(border=True):
    
    st.markdown('''##### We will use the elbow method to test our optimal cluster number, by running the K-Means algorith on each number of clusters from 1 to 10. The inflection point of this series, or "Elbow", indicates the best value as it will provide the greatest number of segments, while minimizing the number of errors.''')

    st.markdown('''##### The resulting plot below shows that :blue[3 Clusters] will be the best path forward.''')
    create_cluster_line_plot()



#########################################
#      s5. Validate Cluster Choice      #
#########################################
st.markdown('''# :blue[Cluster Validation - Silhouette Score]''')
with st.container(border=True):
    st.markdown('''
            ##### Deriving our silhouette score for the chosen number of clusters arrives at a value of :blue[0.53] which indicates a good amount of separation between our cluster edges.
            ''')


    st.code(
        body='''
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler

        data = clean_rfm_data_outliers()
        scalar = StandardScaler().fit(data.values)
        features = scalar.transform(data.values)
        scaled_features = pd.DataFrame(
            features,
            columns=['total_spend', 'visits', 'time_from_last_visit']
        )

        kmeans = KMeans(n_clusters=3, init='k-means++')
        kmeans.fit(scaled_features)
        s_score = round(
            silhouette_score(scaled_features, kmeans.labels_, metric='euclidean'),
            2
        )''',
        language='python'
    )








st.subheader("""We observe that customers in Cluster 0 have a high spend, frequently visit, and have patronized our store recently.
""")
with st.container():
    cols = st.columns(3)
    for item, col in zip(feature_list, cols):
        with col:
            create_kmeans_bar(item)
st.divider()


