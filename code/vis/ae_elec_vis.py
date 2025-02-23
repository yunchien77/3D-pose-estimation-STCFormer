import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import scipy.signal
import tkinter as tk
from tkinter import messagebox

def export_to_excel(folder_name, excel_path='bad_data.xlsx'):
    """將好的數據路徑輸出到Excel檔案"""
    if not os.path.exists(excel_path):
        df = pd.DataFrame(columns=['Folder_Name'])
        df.to_excel(excel_path, index=False)
    
    df = pd.read_excel(excel_path)
    new_row = pd.DataFrame({'Folder_Name': [folder_name]})
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_excel(excel_path, index=False)
    print(f"已將 {folder_name} 添加到 {excel_path}")

def create_overlay_plot(folder_path):
    """創建疊圖並顯示"""
    # 獲取資料夾名稱
    folder_name = os.path.basename(folder_path)
    
    try:
        # 讀取 elec.txt 數據
        elec_file = 'elec.txt'
        elec_path = os.path.join(folder_path, elec_file)
        df_elec = pd.read_csv(elec_path, sep=",")
        if 'R' in elec_path: 
            elec = np.array(180 - df_elec['Linear Transformer Gonio G'])
        else:
            elec = np.array(df_elec['Linear Transformer Gonio G'])
        
        # 將第一個值調整到180，然後後面的值根據差值做修改
        adjustment = 180 - elec[0]
        elec = elec + adjustment
        
        # 讀取 ae_half.xlsx 中的 Processed_Output 數據
        excel_file = 'ae_half.xlsx'
        excel_path = os.path.join(folder_path, excel_file)
        df_processed = pd.read_excel(excel_path)
        processed_output = np.array(df_processed['Processed_Output'])
        original_signal = np.array(df_processed['Original_Full_Signal'])

        adj = processed_output[0] - elec[0]
        elec += adj
        
        # 創建時間軸
        time_elec = np.arange(len(elec))
        time_processed = np.arange(len(processed_output))
        
        # 創建圖表
        plt.figure(figsize=(12, 6))
        
        # 繪製兩個信號
        plt.plot(time_elec, elec, 'g-', label='Elec Signal', linewidth=2)
        plt.plot(time_processed, processed_output, 'b-', label='Processed Output', linewidth=2, alpha=0.7)
        plt.plot(time_processed, original_signal, 'r-', label='Original Signal', linewidth=2, alpha=0.7)
        
        # 添加圖表元素
        plt.suptitle(f'Current Data: {folder_name}', fontsize=14)
        plt.title('Signal Comparison')
        plt.xlabel('Time Steps')
        plt.ylabel('Angle (degrees)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 顯示圖表
        plt.show()
        
        # 詢問使用者這是否是好的疊圖效果
        root = tk.Tk()
        root.withdraw()
        is_good = messagebox.askyesno("確認", "這是好的疊圖效果嗎？")
        
        if not is_good:
            # 保存圖片並記錄到Excel
            # output_path = os.path.join(folder_path, 'overlay.png')
            # plt.savefig(output_path, dpi=300, bbox_inches='tight')
            # print(f"圖片已保存至: {output_path}")
            export_to_excel(folder_name)
        
        plt.close()
        return is_good
        
    except Exception as e:
        print(f"處理 {folder_name} 時發生錯誤: {str(e)}")
        return False

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
    base_folder = r"D:/para/007"  # 請根據實際路徑調整
    process_folders(base_folder)