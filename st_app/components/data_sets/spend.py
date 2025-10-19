import sys
import streamlit as st
from model_data import get_data
conn = st.connection('case_study_db', type='sql')

@st.cache_data
def spend_data():

    data = get_data()
    data['net_spend_amt'] = data['net_spend_amt'].astype(float)
    customer_spend_df = data.groupby('prsn_id')['net_spend_amt'].sum().reset_index()
    customer_spend_df.columns = ['prsn_id', 'total_spend']

    return customer_spend_df
