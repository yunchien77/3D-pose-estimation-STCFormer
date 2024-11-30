import pandas as pd
import numpy as np
import os
from openpyxl import load_workbook

def interpolate_sheet(df):
    # 獲取原始數據的長度
    original_length = len(df)
    
    # 計算新的數據長度（保持原有的時間長度，但每秒256幀）
    new_length = original_length * 256 // 30
    
    # 創建新的時間索引
    old_index = np.arange(original_length)
    new_index = np.linspace(0, original_length - 1, new_length)
    
    # 對每列進行插值
    new_data = {}
    for column in df.columns:
        new_data[column] = np.interp(new_index, old_index, df[column])
    
    # 創建新的DataFrame
    return pd.DataFrame(new_data)

def process_excel_file(input_file, output_file):
    # 讀取Excel文件中的所有工作表
    xlsx = pd.ExcelFile(input_file)
    sheet_names = xlsx.sheet_names
    
    # 創建一個ExcelWriter對象來寫入輸出文件
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for sheet_name in sheet_names:
            print(f"Processing sheet: {sheet_name}")
            # 讀取工作表
            df = pd.read_excel(input_file, sheet_name=sheet_name)
            
            # 對工作表進行插值處理
            new_df = interpolate_sheet(df)
            
            # 將處理後的數據寫入新的Excel文件的相應工作表
            new_df.to_excel(writer, sheet_name=f"{sheet_name}_256fps", index=False)
    
    print(f"Interpolation completed. New Excel file saved to {output_file}")

# 使用函數
input_file = 'code/3dData.xlsx'  # 輸入的Excel文件
output_file = 'code/3dData_256.xlsx'  # 輸出的Excel文件

# 確保輸出文件夾存在
os.makedirs(os.path.dirname(output_file), exist_ok=True)

process_excel_file(input_file, output_file)