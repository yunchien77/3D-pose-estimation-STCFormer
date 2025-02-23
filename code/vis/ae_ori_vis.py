import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import scipy.signal
from openpyxl import load_workbook
import tkinter as tk
from tkinter import messagebox

def export_to_excel(folder_name, excel_path='good_data.xlsx'):
    """將好的數據路徑輸出到Excel檔案"""
    if not os.path.exists(excel_path):
        df = pd.DataFrame(columns=['Folder_Name'])
        df.to_excel(excel_path, index=False)
    
    # 讀取現有的Excel檔案
    df = pd.read_excel(excel_path)
    
    # 添加新的資料夾名稱
    new_row = pd.DataFrame({'Folder_Name': [folder_name]})
    df = pd.concat([df, new_row], ignore_index=True)
    
    # 保存更新後的檔案
    df.to_excel(excel_path, index=False)
    print(f"已將 {folder_name} 添加到 {excel_path}")

def create_overlay_plot(folder_path):
    """创建疊圖並保存"""
    print("分析資料夾：", folder_path)
    
    # 獲取資料夾名稱
    folder_name = os.path.basename(folder_path)
    
    # 讀取數據
    file_hpe = r'ae_half.xlsx'
    dir_hpe = os.path.join(folder_path, file_hpe)
    df_hpe = pd.read_excel(dir_hpe)
    
    # 獲取兩個信號
    processed_signal = np.array(df_hpe['Processed_Output'])
    original_signal = np.array(df_hpe['Original_Full_Signal'])
    time = np.arange(len(processed_signal))
    time_zeroed = time - time[0]
    
    # 創建圖表
    plt.figure(figsize=(12, 8))
    
    # 繪製兩個信號
    plt.plot(time_zeroed, processed_signal, 'b-', label='Processed Output', linewidth=2)
    plt.plot(time_zeroed, original_signal, 'r-', label='Original Signal', linewidth=2, alpha=0.7)
    
    # 添加圖表元素
    plt.suptitle(f'Current Data: {folder_name}', fontsize=14)
    plt.title('Signal Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (degrees)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 顯示圖表
    plt.show()
    
    # 詢問使用者這是否是好的疊凸效果
    root = tk.Tk()
    root.withdraw()  # 隱藏主窗口
    is_good = messagebox.askyesno("確認", "這是好的疊圖效果嗎？")
    
    if is_good:
        # 如果是好的效果，保存圖片並記錄到Excel
        output_path = os.path.join(folder_path, 'overlay.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"圖片已保存至: {output_path}")
        export_to_excel(folder_name)
    
    plt.close()
    return is_good

def process_folders(base_folder):
    """處理所有子資料夾"""
    processed_count = 0
    good_count = 0
    
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        if os.path.isdir(folder_path):
            for fname in os.listdir(folder_path):
                fname_path = os.path.join(folder_path, fname)
                if os.path.isdir(fname_path) and os.path.exists(os.path.join(fname_path, 'ae_half.xlsx')):
                    try:
                        processed_count += 1
                        print(f"正在處理 {fname} ({processed_count})")
                        if create_overlay_plot(fname_path):
                            good_count += 1
                        print(f"成功處理 {fname}")
                    except Exception as e:
                        print(f"處理 {fname} 時發生錯誤: {str(e)}")
    
    print(f"\n處理完成！")
    print(f"總共處理了 {processed_count} 筆數據")
    print(f"其中 {good_count} 筆為好的疊圖效果")

if __name__ == "__main__":
    base_folder = r"D:/para/010"  # 請根據實際路徑調整
    process_folders(base_folder)