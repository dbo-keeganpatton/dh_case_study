import os
import pandas as pd
pd.set_option('display.max_columns', None)
# The purpose of this file is to join all datasets into a single
# consumable dataset that can be used for analysis


print("Starting Process")
card_dim = "../files/card_dim.csv"
prod_dim = "../files/prod_dim.csv"
store_dim = "../files/store_dim.csv"
transaction_dim = "../files/transaction_item.csv"


# prsn_id indicates a unique customer ID
card_dim_df = pd.read_csv(card_dim)
# prod_id is UUID for prod_dim
prod_dim_df = pd.read_csv(prod_dim)
# store_id is UUID for store_dim
store_dim_df = pd.read_csv(store_dim)
# contains card_id, store_id, and prod_id
# This dataset contains 367,752 records
# and 28,032 unique transactions
transaction_dim_df = pd.read_csv(transaction_dim)


# Consolidate all attributes into our core fact table,
# I just think that is easier and tend to enjoy OBT style schemas
# when possible.
add_prsn_id_to_txn_df = pd.merge(transaction_dim_df, card_dim_df, how="left", on="card_id")
add_prod_dm_to_txn_df = pd.merge(add_prsn_id_to_txn_df, prod_dim_df, how="left", on="prod_id")
consolidated_txn_df = pd.merge(add_prod_dm_to_txn_df, store_dim_df, how="left", on="store_id")


# Output for further analysis
consolidated_txn_df.to_csv("../files/consolidated_txn_data.csv")
if os.path.exists("../files/consolidated_txn_data.csv"):
    print(f"""
        consolidated_txn_data.csv created with: 
        {consolidated_txn_df.shape[0]} rows,
        {consolidated_txn_df.shape[1]} columns
        """
    )
else:
    "Error creating file"
