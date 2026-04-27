# Final Project Preprocessing
# Sachin Patel
# U01124876

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# for same results
seed = 42
np.random.seed(seed)

# load data
df = pd.read_csv("data/DataCoSupplyChainDataset.csv", encoding='latin-1')

print("Data loaded")
print("Rows:", df.shape[0])
print("Cols:", df.shape[1])
print("Shape =", df.shape)

# remove unnecessary columns from dataset
dataleak_col = ['Delivery Status',
                'Days for shipping (real)',
                'Order Status']

private_col = ['Customer Email',
               'Customer Password',
               'Customer Fname',
               'Customer Lname',
               'Customer Street']

id_col = ['Customer Id',
          'Order Id',
          'Order Item Id',
          'Order Customer Id',
          'Product Card Id',
          'Order Item Cardprod Id',
          'Category Id',
          'Department Id',
          'Product Category Id']

missing_col = ['Product Description',
               'Order Zipcode']

extra_col = ['Product Image',
             'Product Status',
             'Customer Zipcode',
             'Product Name',
             'Customer City',
             'Order City',
             'Customer State',
             'Order State',
             'Order Country',
             'Customer Country']

cols_to_drop = dataleak_col + private_col + id_col + missing_col + extra_col

df = df.drop(columns=cols_to_drop)

print("Dropped cols:", len(cols_to_drop))
print("Dataset size after dropping cols:- ", df.shape)

# make some new features from dates
df['order date (DateOrders)'] = pd.to_datetime(df['order date (DateOrders)'])
df['shipping date (DateOrders)'] = pd.to_datetime(df['shipping date (DateOrders)'])

# order date features
df['ord_hour'] = df['order date (DateOrders)'].dt.hour
df['ord_day'] = df['order date (DateOrders)'].dt.dayofweek
df['ord_month'] = df['order date (DateOrders)'].dt.month
df['ord_year'] = df['order date (DateOrders)'].dt.year
# weekend
df['weekend_flag'] = df['ord_day'].isin([5,6]).astype(int)

# shipping gap
df['ship_gap'] = (
    df['shipping date (DateOrders)'] -
    df['order date (DateOrders)']
).dt.days

df = df.drop(columns=['order date (DateOrders)',
                      'shipping date (DateOrders)'])

print("Added date-based new features")
print("Dataset size after adding features :-", df.shape)

# take only 50,000 row for sample data... 

df_sample, _ = train_test_split(
    df,
    train_size=50000,
    stratify=df['Late_delivery_risk'],
    random_state=seed
)

df_sample = df_sample.reset_index(drop=True)

late_pct = df_sample['Late_delivery_risk'].mean() * 100

print("Sample size:", df_sample.shape)
print("Late %:", round(late_pct,1))
print("On-time %:", round(100-late_pct,1))


# save cleaned file
out_file = 'data/dataco_cleaned.csv'
df_sample.to_csv(out_file, index=False)

print("Saved file:", out_file)

print("Done preprocessing")
print("Final cleaned dataset size :-", df_sample.shape)