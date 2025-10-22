import streamlit as st

def top_products():
    conn = st.connection('case_study_db', type='sql')
    query = conn.query("""
            with
                combine_data as
                (
                    select
                    txn.*,
                    clst.cluster
                    from transactions as txn
                    join cust_cluster as clst using (prsn_id)
                    where true
                    and clst.cluster = 2
                )

            select 
            count(*) as item_cnt,
            prod_desc
            from combine_data
            group by 2
            order by 1 desc
            limit 10
            """
        )
    return st.dataframe(query)
