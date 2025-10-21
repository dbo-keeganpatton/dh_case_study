from rfm import rfm_data 
from scipy import stats
import streamlit as st
import numpy as np


def clean_rfm_data_outliers():

    rfm = rfm_data()
    z_score_df = rfm[['total_spend', 'visits', 'time_from_last_visit']]
    z_score = np.abs(stats.zscore(z_score_df))
    remove_outliers = (z_score < 2).all(axis=1) 
    rfm_outliers_removed = z_score_df[remove_outliers]
    rfm_outliers_removed = rfm_outliers_removed.drop_duplicates()

    return rfm_outliers_removed
    

