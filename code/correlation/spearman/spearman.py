import pandas as pd
from scipy import stats
import re

def load_and_process_data(parameter_file, fmas_file):
    """
    讀取並處理參數文件和 FMAS 評分文件
    
    Args:
        parameter_file: 包含 p1 參數的 Excel 文件
        fmas_file: 包含 FMAS 評分的 Excel 文件
    
    Returns:
        配對好的 p1 值和 FMAS 評分列表
    """
    # 讀取 FMAS 數據
    fmas_df = pd.read_excel(fmas_file)
    # 將 subject 轉為字符串並補零至三位數
    fmas_df['Subject'] = fmas_df['Subject'].apply(lambda x: f"{x:03d}")
    
    # 讀取參數數據，設置第一行為列名
    params_df = pd.read_excel(parameter_file)
    
    # 將第一列設為索引
    params_df.set_index(params_df.columns[0], inplace=True)
    
    # 提取需要的數據
    para_values = []
    fmas_values = []
    
    # 遍歷每一列（每個受試者的數據）
    for column in params_df.columns:
        # 只處理包含 's' 的數據（排除 'h' 的數據）
        if '_s' in column.lower():
            # 提取受試者編號
            subject_num = column[:3]
            
            try:
                # 獲取對應的 p1 值
                para_value = float(params_df.loc['Decay rate (b)', column])
                
                # 查找對應的 FMAS 評分
                fmas_score = fmas_df.loc[fmas_df['Subject'] == subject_num, 'fmas'].values
                
                if len(fmas_score) > 0:
                    # 處理可能的 '1+' 情況
                    fmas_score = fmas_score[0]
                    if isinstance(fmas_score, str) and '+' in fmas_score:
                        fmas_score = float(fmas_score.replace('+', '')) + 0.5
                    else:
                        fmas_score = float(fmas_score)
                    
                    para_values.append(para_value)
                    fmas_values.append(fmas_score)
                    print(f"Successfully processed: Subject {subject_num}, p1={para_value}, FMAS={fmas_score}")
                
            except Exception as e:
                print(f"Error processing subject {subject_num}: {str(e)}")
                continue
    
    if not para_values or not fmas_values:
        raise ValueError("No valid data pairs were found. Please check your input files.")
    
    return para_values, fmas_values

def perform_correlation_analysis(para_values, fmas_values):
    """
    進行 Spearman 相關性分析
    
    Args:
        para_values: p1 參數值列表
        fmas_values: FMAS 評分列表
    
    Returns:
        相關係數和 p 值
    """
    correlation, p_value = stats.spearmanr(para_values, fmas_values)
    return correlation, p_value

def analyze_data(parameter_file, fmas_file):
    """
    主要分析函數
    """
    # 載入和處理數據
    para_values, fmas_values = load_and_process_data(parameter_file, fmas_file)
    
    # 進行相關性分析
    correlation, p_value = perform_correlation_analysis(para_values, fmas_values)
    
    # 輸出結果
    print(f"\nAnalysis Results:")
    print(f"Spearman 相關係數: {correlation:.20f}")
    print(f"P 值: {p_value:.20f}")
    
    # 創建數據框以展示配對數據
    results_df = pd.DataFrame({
        'P1 值': para_values,
        'FMAS 評分': fmas_values
    })
    
    return results_df, correlation, p_value

# 使用示例
if __name__ == "__main__":
    try:
        parameter_file = r"C:\Users\cherr\anaconda3\envs\STCformer\STCFormer-main\exponential_results_elec.xlsx"
        fmas_file = r"C:\Users\cherr\anaconda3\envs\STCformer\STCFormer-main\code\mas_table.xlsx"
        
        results_df, correlation, p_value = analyze_data(parameter_file, fmas_file)
        print("\nDetailed Results:")
        print(results_df)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")