import os
import pandas as pd
import numpy as np
import csv

folder_path = r'D:/output/002/002L_hb_1/002L_hb_1_5'
file_elec = r'elec.txt'

# 文件的完整路徑
dir_elec = os.path.join(folder_path, file_elec)

# 讀取 elec.txt 文件，假設它是以空格或逗號分隔的表格數據
elec_data = pd.read_csv(dir_elec, sep=',')  # 修改 `sep` 根據你的文件格式

print(elec_data.columns)

# 檢查文件夾路徑是否包含 'L'，然後選擇不同的計算方式
if 'L' in folder_path:
    elec = np.array(elec_data['Linear Transformer Gonio G'])
else:
    elec = np.array(180 - elec_data['Linear Transformer Gonio G'])

# 確保第一個值為 180，並調整後續值
first_value = elec[0]
offset = 180 - first_value  # 計算偏移量

# 將所有值根據偏移量進行調整
elec_adjusted = elec + offset

# 保存結果為 CSV 文件
csv_filename = 'elec.csv'
output_dir = os.path.join(folder_path, csv_filename)

with open(output_dir, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['elec_angles'])  # 寫入標題行

    for angle in elec_adjusted:
        csv_writer.writerow([angle])  # 寫入每個角度
