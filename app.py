import sys
import numpy as np
import pandas as pd
import streamlit as st
sys.path.append("./components/")
sys.path.append("./components/visuals/")
sys.path.append("./components/data_sets/")
from bx_plots import create_box_plot
from hg_plots import create_histogram
from cln_bx_plot import create_clean_box_plot
from cluster_line import create_cluster_line_plot 
from cluster_bars import create_kmeans_bar
from s_score import silhouette_score_text 
from cluster2_top_items import top_products 
from cluster2_top_weekdays import top_weekdays
st.set_page_config(
    layout="wide",
    page_title="DH Case Study",
    page_icon="🐳"
)


st.title("Customer Loyalty Segmentation Lab")
st.link_button(label="View Project Code Repository", url="https://github.com/dbo-keeganpatton/dh_case_study")
#########################################
#            Reusable Vars              #
#########################################
feature_list = ['total_spend', 'visits', 'time_from_last_visit']

#########################################
#                Prompt                 #
#########################################
with st.container(border=True):
    st.markdown('''
    * Develop a customer-level segmentation classifying each customer into n number of segments dependent on loyalty.
        1. What metrics were chosen to determine loyalty? 
        2. How did you choose the number of customer segments?
        3. In what ways could this segmentation be used to influence business strategy?
    ''')

#########################################
#        s1. Data Exploration           #
#########################################
st.markdown('''## :blue[Data Exploration & Feature Selection]''')
with st.container(border=True):
    st.markdown('''We start by identifying potential features that can be used to distinguish customer Loyalty, and viewing their distribution in general across the consolidated dataset. A series of histograms show that this data is heavily :blue[left skewed], indicating not only an abnormal distribution, but also the presence of several outliers that will affect our clustering further on. These will need to be identified and removed for our clustering later on''')
    st.markdown(
        '''
        1. :blue[**Frequency**] of visits.
        2. :blue[**Total Spent**] by each customer.
        3. :blue[**Recency**] of last visit for each customer.
        '''
    )

    with st.container():
        cols = st.columns(3)
        for item, col in zip(feature_list, cols):
            with col:
               create_histogram(item)
    st.divider()

    st.markdown('''We can further visulize the presence of outliers in the data with a series of box plots. From this it is obvious that each of our features will need some cleaning before feeding to our model.''')
    with st.container():
        cols = st.columns(3)
        for item, col in zip(feature_list, cols):
            with col:
               create_box_plot(item)

                

#########################################
#         s2. Outlier Removal           #
#########################################
st.markdown('''## :blue[Outlier Removal]''')
with st.container(border=True):

    st.markdown('''We know that outliers exist, but we need to determine the threshold by which we will classify a value as an outlier an remove them. This can be achieved by deriving each value's :blue[Z-Score]. Which calculates the distance from the dataset's mean measured in 'n' standard deviations.''')
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

    st.markdown('''Using the code above, we will remove all data that is :blue[more than a single standard deviation] from our mean values.''')
    st.divider()
    
    st.markdown('''Removing these outliers has an effect of widening the :blue[IQR] for our 3 features, improving our data quality, and better positioning our data to fit to our model.''')
    with st.container():
        cols = st.columns(3)
        for item, col in zip(feature_list, cols):
            with col:
               create_clean_box_plot(item)



#########################################
#         s3. Cluster Testing           #
#########################################
st.markdown('''## :blue[Model Selection & Cluster Testing]''')
with st.container(border=True):
    
    st.markdown('''For our excersice, we will be using a K-Means algorithm as our model by which to segment our customers. The first step is testing and selecting the number of clusters that will be optimal to lead us to a meaningful series of groups. We can test this by processing our data through K-Means several times with each iteration applying a different cluster number, and testing the :blue[Within-cluster sum of squares] for each iteration. This is known as the :blue[Elbow Method], and merely shows the point at which we are able to have the maximum number of clusters, while still retaining defined edges and separation for each one.''')
    create_cluster_line_plot()
    st.markdown('''The resulting plot above shows that :blue[3 Clusters] provides us with the most optimal number of clusters, while minimizing our :blue[WCSS]. This is indicated by the "elbow" of the chart where the line starts to bend further towards the y-orgin.''')
    st.divider()

    st.markdown(f"""We can further validate our decision by calculating the :blue[silhouette score] for the chosen number of clusters. This is a value between -1 and 1 that measures cluster separation and compactness. Sci-Kit learn provides native support for calculating this seen in the code excerpt below""")


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

    st.markdown(f"""## Silhouette Score [:blue[{silhouette_score_text()}]]""")
    st.markdown('''Our resulting silhouette score is close to +1 and indicates solid cluster cohesion allowing us to proceed with creating and plotting our model.''')


#########################################
#      s4. Create and Plot Clusters     #
#########################################
st.markdown('''## :blue[K-Means Cluster Plot]''')
with st.container(border=True):

    st.markdown(f"Plotting our 3 clusters for each feature clearly shows that one of our clusters is highly aligned to patterns in each feature that would correlate with customer loyalty.")

    cols = st.columns(3)
    for item, col in zip(feature_list, cols):
        with col:
            create_kmeans_bar(item)
    


st.markdown('''## :blue[Potential Strategy]''')
with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Products')
        top_products()
    with col2:
        st.subheader('Days')
        top_weekdays()
