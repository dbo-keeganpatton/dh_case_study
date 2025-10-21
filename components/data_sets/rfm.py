import streamlit as st
from recency import recency_data
from frequency import frequency_data
from spend import spend_data


def rfm_data():

    spend_df = spend_data()
    recency_df = recency_data()
    frequency_df = frequency_data()

    rf = recency_df.merge(frequency_df, on='prsn_id')
    rfm = rf.merge(spend_df, on='prsn_id')
    rfm = rfm[['prsn_id', 'total_spend', 'visits', 'time_from_last_visit']]

    return rfm


