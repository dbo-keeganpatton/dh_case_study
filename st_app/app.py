import sys
import os
import pandas as pd
import streamlit as st
sys.path.append('./query_apis')
st.set_page_config(layout="wide")
conn = st.connection('case_study_db', type='sql')


# Just output the consolidated data
st.subheader("Data Shape")
st.write(conn.query("select * from transactions").head())


# The next sections will look at how recent we saw a visit from each customer
# How often we see them return, and their overall spend value for creating
# groupings
rfm_df = conn.query(
    """
    select 
    prsn_id,
    item_qty,
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
st.write(customer_last_visit_df)


################################
#           Frequency          #
###############################
st.subheader("Customer Freqency")
customer_frequency_df = rfm_df.groupby('prsn_id')['date_id'].count().reset_index()
customer_frequency_df.columns = ['prsn_id', 'visits']
st.write(customer_frequency_df)


################################
#           Spend             #
###############################
st.subheader("Customer Spend")
rfm_df['net_spend_amt'] = rfm_df['net_spend_amt'].astype(float)
customer_spend_df = rfm_df.groupby('prsn_id')['net_spend_amt'].sum().reset_index()
customer_spend_df.columns = ['prsn_id', 'total_spend']
st.write(customer_spend_df)


################################
#           RFM DF             #
###############################
rf = customer_last_visit_df.merge(customer_frequency_df, on='prsn_id')
rfm = rf.merge(customer_spend_df, on='prsn_id')
rfm = rfm[['prsn_id', 'total_spend', 'visits', 'time_from_last_visit']]
st.write(rfm)
