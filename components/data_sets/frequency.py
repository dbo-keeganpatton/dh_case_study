import streamlit as st
from model_data import get_data
conn = st.connection('case_study_db', type='sql')


@st.cache_data
def frequency_data():
    
    data = get_data()
    customer_frequency_df = data.groupby('prsn_id')['transaction_code'].nunique().reset_index()
    customer_frequency_df.columns = ['prsn_id', 'visits']

    return customer_frequency_df






