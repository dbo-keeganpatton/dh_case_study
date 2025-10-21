import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
sys.path.append("./data_sets/")
from rfm_outlier_removed import clean_rfm_data_outliers


def silhouette_score_text():

    rfm_outliers_removed = clean_rfm_data_outliers()
    scalar = StandardScaler().fit(rfm_outliers_removed.values)
    features = scalar.transform(rfm_outliers_removed.values)
    scaled_features = pd.DataFrame(features, columns=['total_spend', 'visits', 'time_from_last_visit'])


    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
    kmeans.fit(scaled_features)
    return  round(silhouette_score(scaled_features, kmeans.labels_, metric='euclidean'),2)

