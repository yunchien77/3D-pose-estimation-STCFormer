import pandas as pd

# 讀取TXT檔案
file_path = 'code/002L_1_5.txt'
df = pd.read_csv(file_path)

# 只保留 'Time' 和 'Linear Transformer Gonio G' 這兩個欄位
filtered_df = df[['Time', 'Linear Transformer Gonio G']]

# 將結果保存到一個新的TXT檔案
filtered_df.to_csv('code/filtered_data.txt', index=False, sep=',')
