import pandas as pd

def compare_excel_columns(file1_path, file2_path):
    """
    比較兩個 Excel 檔案的欄位差異
    
    Parameters:
    file1_path (str): 第一個 Excel 檔案的路徑
    file2_path (str): 第二個 Excel 檔案的路徑
    
    Returns:
    tuple: (只在檔案1中的欄位, 只在檔案2中的欄位)
    """
    try:
        # 讀取兩個 Excel 檔案
        df1 = pd.read_excel(file1_path)
        df2 = pd.read_excel(file2_path)
        
        # 獲取兩個檔案的欄位名稱
        columns1 = set(df1.columns)
        columns2 = set(df2.columns)
        
        # 找出差異
        only_in_file1 = columns1 - columns2
        only_in_file2 = columns2 - columns1
        
        # 排序結果以便閱讀
        only_in_file1 = sorted(list(only_in_file1))
        only_in_file2 = sorted(list(only_in_file2))
        
        # 顯示結果
        print("\n只在檔案1中的欄位:")
        if only_in_file1:
            for col in only_in_file1:
                print(f"- {col}")
        else:
            print("無")
            
        print("\n只在檔案2中的欄位:")
        if only_in_file2:
            for col in only_in_file2:
                print(f"- {col}")
        else:
            print("無")
            
        return only_in_file1, only_in_file2
        
    except Exception as e:
        print(f"發生錯誤: {str(e)}")
        return None, None

# 使用範例
if __name__ == "__main__":
    # 替換成你的實際檔案路徑
    file1_path = r"C:\Users\cherr\anaconda3\envs\STCformer\STCFormer-main\analysis_results_hpe_v2.xlsx"
    file2_path = r"C:\Users\cherr\anaconda3\envs\STCformer\STCFormer-main\analysis_results_hpe_v3.xlsx"
    
    compare_excel_columns(file1_path, file2_path)