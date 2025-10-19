import streamlit as st

@st.cache_data
def get_weekday_groups():
    conn = st.connection('case_study_db', type='sql')
    return conn.query("""
        with
            cust as 
            (
                select prsn_id, 
                count(distinct transaction_code) as txns
                from transactions
                group by prsn_id
                having txns > 25
            ),

            filtered_txn as 
            (
                select
                *
                from transactions
                inner join cust on cust.prsn_id = transactions.prsn_id
            )
            
            select
            count(distinct prsn_id) as custs,
            count(distinct transaction_code) as txns,
            count(distinct transaction_code) / count(distinct prsn_id) as mean_daily_txn,
            week_day
            from filtered_txn
            group by week_day
            """
    )


