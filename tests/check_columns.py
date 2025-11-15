import pandas as pd

df = pd.read_parquet('data/raw/train/train/datetime=2025-10-01-00-00/', engine='pyarrow')

print('Critical target columns:')
targets = ['impression_id', 'user_id', 'row_id', 'datetime', 'buyer_d1', 'buyer_d7', 
           'buyer_d14', 'buyer_d28', 'iap_revenue_d1', 'iap_revenue_d7', 
           'iap_revenue_d14', 'iap_revenue_d28']
for col in targets:
    print(f'  {col}: {col in df.columns}')

print('\nAll columns with "session" in name:')
session_cols = [c for c in df.columns if 'session' in c.lower()]
print(session_cols)

print('\navg_daily_sessions sample values:')
print(df['avg_daily_sessions'].head(10))
print(f'\nType: {type(df["avg_daily_sessions"].iloc[2])}')
