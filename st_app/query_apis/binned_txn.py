import streamlit as st

@st.cache_data
def binned_txn_query():
    conn = st.connection('case_study_db', type='sql')
    return conn.query("""
        with
            agg as
            (

                select 
                prsn_id,
                count(distinct transaction_code) as txns,
                sum(net_spend_amt) as net_spend
                from transactions 
                group by prsn_id
            ),

            add_bin as
            (
                select 
                prsn_id,
                txns,
                net_spend,
                case
                    when txns > 0 and txns <= 25 then 1
                    when txns > 25 and txns <= 50 then 2 
                    when txns > 50 and txns <= 75 then 3 
                    when txns > 75 and txns <= 100 then 4
                    when txns > 100 then 5 
                    else null
                end as txns_bins_sort,
                case
                    when txns > 0 and txns <= 25 then '1-25'
                    when txns > 25 and txns <= 50 then '26-50'
                    when txns > 50 and txns <= 75 then '51-75'
                    when txns > 75 and txns <= 100 then '76-100'
                    when txns > 100 then '101+'
                    else null
                end as txns_bins
                from agg
            )

        select 
        txns_bins,
        txns_bins_sort,
        count(distinct prsn_id) as cust_cnt,
        sum(net_spend) / sum(txns) as mean_net_spend
        from add_bin 
        group by txns_bins 
        """
        )

