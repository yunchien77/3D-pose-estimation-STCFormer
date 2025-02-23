import pandas as pd
import numpy as np

def interpolate_data(input_file, output_file):
    # 讀取CSV文件
    df = pd.read_csv(input_file)
    
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
    new_df = pd.DataFrame(new_data)
    
    # 保存新的CSV文件
    new_df.to_csv(output_file, index=False)
    
    print(f"Interpolation completed. New CSV file saved to {output_file}")
    return new_df

# 使用函數
input_file = 'E:/output/021/021R_sb_2/021R_sb_2_7/output_3D/Angles_3D.csv'  # 請確保這是您的輸入文件名
output_file = 'E:/output/021/021R_sb_2/021R_sb_2_7/output_3D/Angles_3D_256fps.csv'
interpolated_df = interpolate_data(input_file, output_file)