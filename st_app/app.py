import streamlit as st
import pandas as pd
from pandasql import sqldf
st.set_page_config(layout="wide")

@st.cache_data
def get_analysis_data(path):
    data=pd.read_csv(path)
    return data

app_data = get_analysis_data("../files/analysis_dataset_cleaned.csv")


##########################
#   Histogram Sections   #
##########################
hist_bins = [0, 25, 50, 75, 100, 125, 150, 175, 200]
labels = ["0-25", "26-50", "51-75", "76-100", "101-125", "126-150", "151-175", "176+"]
transaction_counts = app_data.groupby('prsn_id')['transaction_code'].count()
hist_data = pd.cut(transaction_counts, labels=labels, bins=hist_bins,right=True)
hist_counts = hist_data.value_counts().sort_index()
hist_df = hist_counts.reset_index()
hist_df.columns = ['Range', 'Txn Count']
st.subheader("Distribution of Customer Transaction Counts")
st.bar_chart(
    data=hist_df,
    x="Range",
    y="Txn Count",
    x_label="",
    y_label=""

)

test = sqldf("""
with
    agg as
    (

        select 
        prsn_id,
        count(distinct transaction_code) as txns,
        sum(net_spend_amt) as net_spend
        from app_data
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

st.write(test)

st.header("Raw Data")
st.data_editor(app_data)



