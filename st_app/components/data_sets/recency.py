import sys
import pandas as pd
import streamlit as st
from model_data import get_data
conn = st.connection('case_study_db', type='sql')


@st.cache_data
def recency_data():

    data = get_data()
    data['visit_rank'] = data.sort_values(['prsn_id', 'date_id']).groupby(['prsn_id'])['date_id'].rank(
        method='min'
    ).astype(int)

    customer_last_visit_df = data[data['visit_rank']==1]
    customer_last_visit_df['date_id'] = pd.to_datetime(customer_last_visit_df['date_id'])
    customer_last_visit_df['time_from_last_visit'] = (
        customer_last_visit_df['date_id'] - min(customer_last_visit_df['date_id']) 
    ).dt.days

    return customer_last_visit_df

