import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict

def parse_title(title):
    """解析數據標題，返回解析後的組件"""
    try:
        subject_id = title[:3]  # 前三碼（受試者ID）
        leg_type = 's' if 's' in title.lower() else None  # 只關注痙攣腳的數據
        angle = 'a' if 'a' in title.lower() else 'b'  # 角度 (a=45度, b=90度)
        
        # 解析姿勢編號
        parts = title.split('_')
        pose_num = parts[2] if len(parts) > 2 else None  # 取得姿勢編號
        
        if leg_type and pose_num:
            return {
                'subject_id': subject_id,
                'leg_type': leg_type,
                'angle': angle,
                'pose': pose_num,
                'full_title': title
            }
        return None
    except Exception:
        return None

def remove_outliers_and_average(values):
    """處理單組數據的離群值並計算平均值"""
    if len(values) <= 1:
        return values, [], np.mean(values)
    
    # 使用四分位距法(IQR)檢測離群值
    Q1, Q3 = np.percentile(values, [25, 75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # 識別非離群值
    valid_values = [v for v in values if lower_bound <= v <= upper_bound]
    outliers = [v for v in values if v < lower_bound or v > upper_bound]
    
    # 如果所有值都被標記為離群值，保留原始數據
    if len(valid_values) == 0:
        return values, [], np.mean(values)
    
    return valid_values, outliers, np.mean(valid_values)

def group_and_process_measurements(params_df):
    """將相同受試者的多次測量分組並處理離群值，按角度和姿勢分開處理"""
    # 使用巢狀的defaultdict來組織數據
    measurements = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    titles = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # 收集所有痙攣側的測量值
    for column in params_df.columns:
        parsed = parse_title(column)
        if parsed and parsed['leg_type'] == 's':
            try:
                value = float(params_df.loc['Decay rate (b)', column])  # 修改為 'p1'
                if not pd.isna(value):
                    measurements[parsed['subject_id']][parsed['angle']][parsed['pose']].append(value)
                    titles[parsed['subject_id']][parsed['angle']][parsed['pose']].append(parsed['full_title'])
            except Exception:
                continue
    
    # 處理每個受試者在每種情境下的數據
    processed_data = {}
    for subject_id in measurements:
        processed_data[subject_id] = {}
        for angle in measurements[subject_id]:
            processed_data[subject_id][angle] = {}
            for pose in measurements[subject_id][angle]:
                values = measurements[subject_id][angle][pose]
                valid_values, outliers, mean = remove_outliers_and_average(values)
                processed_data[subject_id][angle][pose] = {
                    'mean': mean,
                    'valid_values': valid_values,
                    'outliers': outliers,
                    'titles': titles[subject_id][angle][pose]
                }
    
    return processed_data

def load_and_process_data(parameter_file, fmas_file):
    """讀取並處理參數文件和 FMAS 評分文件"""
    # 讀取 FMAS 數據
    fmas_df = pd.read_excel(fmas_file)
    fmas_df['Subject'] = fmas_df['Subject'].apply(lambda x: f"{x:03d}")
    
    # 讀取參數數據
    params_df = pd.read_excel(parameter_file)
    params_df.set_index(params_df.columns[0], inplace=True)
    
    # 處理參數數據（包含離群值移除）
    processed_params = group_and_process_measurements(params_df)
    
    detailed_results = []
    all_para_values = []
    all_fmas_values = []
    
    # 配對數據
    for subject_id, angles in processed_params.items():
        fmas_score = fmas_df.loc[fmas_df['Subject'] == subject_id, 'fmas'].values
        
        if len(fmas_score) > 0:
            # 處理 FMAS 評分
            fmas_score = fmas_score[0]
            if isinstance(fmas_score, str) and '+' in fmas_score:
                fmas_score = float(fmas_score.replace('+', '')) + 0.5
            else:
                fmas_score = float(fmas_score)
            
            # 處理每種情境
            for angle in angles:
                for pose in angles[angle]:
                    data = angles[angle][pose]
                    
                    # 加入整體分析數據
                    all_para_values.append(data['mean'])
                    all_fmas_values.append(fmas_score)
                    
                    detailed_results.append({
                        'subject_id': subject_id,
                        'angle': angle,
                        'pose': pose,
                        'p1_mean': data['mean'],
                        'fmas_score': fmas_score,
                        'valid_values': data['valid_values'],
                        'outliers': data['outliers'],
                        'titles': data['titles']
                    })
                    
                    print(f"Processed: Subject {subject_id}, {angle}_{pose}, "
                          f"mean p1={data['mean']:.4f}, FMAS={fmas_score}")
                    if data['outliers']:
                        print(f"  Outliers removed: {data['outliers']}")
    
    return all_para_values, all_fmas_values, detailed_results

def perform_correlation_analysis(para_values, fmas_values):
    """進行 Kendall 相關性分析"""
    correlation, p_value = stats.kendalltau(para_values, fmas_values)
    return correlation, p_value

def export_results_to_excel(correlation, p_value, sample_size, detailed_results, output_file):
    """將分析結果匯出到Excel"""
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        # 1. 統計摘要
        summary_data = {
            '統計項目': ['Kendall tau相關係數', 'p值', '總樣本數'],
            '數值': [correlation, p_value, sample_size]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='統計摘要', index=False)
        
        # 2. 詳細結果
        detailed_data = []
        for result in detailed_results:
            detailed_data.append({
                '受試者ID': result['subject_id'],
                '角度': '45度' if result['angle'] == 'a' else '90度',
                '姿勢': result['pose'],
                'P1平均值': result['p1_mean'],
                'FMAS評分': result['fmas_score'],
                '有效數據數量': len(result['valid_values']),
                '離群值數量': len(result['outliers']),
                '數據標題': ', '.join(result['titles'])
            })
        pd.DataFrame(detailed_data).to_excel(writer, sheet_name='詳細配對結果', index=False)
        
        # 3. 原始數據（包含離群值標記）
        raw_data = []
        for result in detailed_results:
            for value in result['valid_values']:
                raw_data.append({
                    '受試者ID': result['subject_id'],
                    '角度': '45度' if result['angle'] == 'a' else '90度',
                    '姿勢': result['pose'],
                    'P1值': value,
                    'FMAS評分': result['fmas_score'],
                    '是否離群值': '否'
                })
            for value in result['outliers']:
                raw_data.append({
                    '受試者ID': result['subject_id'],
                    '角度': '45度' if result['angle'] == 'a' else '90度',
                    '姿勢': result['pose'],
                    'P1值': value,
                    'FMAS評分': result['fmas_score'],
                    '是否離群值': '是'
                })
        pd.DataFrame(raw_data).to_excel(writer, sheet_name='原始數據', index=False)

def analyze_data(parameter_file, fmas_file, output_file):
    """主要分析函數"""
    # 載入和處理數據
    para_values, fmas_values, detailed_results = load_and_process_data(parameter_file, fmas_file)
    
    # 進行相關性分析
    correlation, p_value = perform_correlation_analysis(para_values, fmas_values)
    
    # 輸出結果
    print(f"\nAnalysis Results:")
    print(f"Kendall tau 相關係數: {correlation:.20f}")
    print(f"P 值: {p_value:.20f}")
    print(f"總樣本數: {len(para_values)}")
    
    # 匯出結果
    export_results_to_excel(correlation, p_value, len(para_values), detailed_results, output_file)
    print(f"\n詳細結果已匯出到: {output_file}")
    
    return correlation, p_value, detailed_results

if __name__ == "__main__":
    try:
        parameter_file = r"C:\Users\cherr\anaconda3\envs\STCformer\STCFormer-main\exponential_results_3d.xlsx"
        fmas_file = r"C:\Users\cherr\anaconda3\envs\STCformer\STCFormer-main\code\mas_table.xlsx"
        output_file = "kendall_analysis_results.xlsx"
        
        correlation, p_value, detailed_results = analyze_data(
            parameter_file, fmas_file, output_file)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")