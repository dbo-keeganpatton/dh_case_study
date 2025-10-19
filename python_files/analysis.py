import pandas as pd
pd.set_option('display.max_columns', None)


# Some dtype QOL changes
df = pd.read_csv("../files/analysis_dataset_cleaned.csv")
df['date_id'] = pd.to_datetime(df.date_id, errors='coerce').dt.date

print(f"Data range: {df.date_id.min()} to {df.date_id.max()}")
print(df.columns)
print("Top 10 Customers by Number of Visits")
print(df.groupby(by=['prsn_id'])['transaction_code'].count().sort_values(ascending=False).head(10))



