import os
import pandas as pd
pd.set_option('display.max_columns', None)


# This file evaluates some general data quality from 
# our consolidated dataset, mainly the presence of null
# or duplicate values
consolidated_data = "../files/consolidated_txn_data.csv"
cons_df = pd.read_csv(consolidated_data)


# It appears that a small number of products exist within
# the transactions dataset, which have no presence in 
# our product dim data.
print(f"""Count of Null Product ID Values: {cons_df[cons_df.isnull().any(axis=1)].prod_id.nunique()} in Txn Data.""")


# Testing some example product ID's to ensure this is not
# a formatting issue on either side.
missing_prod_id_list =  cons_df[cons_df.isnull().any(axis=1)].prod_id.unique()


# bring in Product Dim dataset for testing
prod_dim = pd.read_csv("../files/prod_dim.csv")
null_prod_desc_count = 0 
for item in missing_prod_id_list:
    prod_dim[prod_dim.prod_id==item].prod_desc.nunique()
    null_prod_desc_count +=1
    
print(f"{null_prod_desc_count} null product descriptions in product dimension.")
    

# It appears that the null product ID's present in our core transaction dataset do indeed
# exist in the product data, however their attributes are null values, hence they will have no
# relevance in subsequent analysis. 
# I will filter out these transactions to improve data quality.
remove_nulls_from_consolidated_data = cons_df.dropna() 
print(f"""Removed {len(cons_df) - len(remove_nulls_from_consolidated_data)} rows in Transaction dataset due to null products.""")
remove_nulls_from_consolidated_data.to_csv("../files/analysis_dataset_cleaned.csv")
if os.path.exists("../files/analysis_dataset_cleaned.csv"):
    print("Created analysis_dataset_cleaned.csv dataset for use")
else:
    print("error creating dataset")
