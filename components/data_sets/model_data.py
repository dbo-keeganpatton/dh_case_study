import streamlit as st

@st.cache_data
def get_data():
    conn = st.connection('case_study_db', type='sql')
    return conn.query("""
        select distinct
        prsn_id,
        item_qty,
        transaction_code,
        net_spend_amt,
        date_id
        from transactions
        """
    )
