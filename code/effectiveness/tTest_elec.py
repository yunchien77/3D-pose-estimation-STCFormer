import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def load_excel_file(file_path):
    """載入Excel文件並轉換為DataFrame"""
    return pd.read_excel(file_path)

def parse_title(title):
    """解析數據標題，返回解析後的組件"""
    try:
        subject_id = title[:3]  # 前三碼（受試者ID）
        side = title[3:4]  # L或R
        leg_type = 'h' if 'h' in title else 's'  # 健康腳或痙攣腳
        
        parts = title.split('_')
        pose_num = parts[2]  # 取得_1或_2
        
        # 建立基本識別碼（不含角度和重複測量次數）
        base_id = f"{subject_id}_{side}_{leg_type}_{pose_num}"
        
        return {
            'subject_id': subject_id,
            'side': side,
            'leg_type': leg_type,
            'pose': pose_num,
            'base_id': base_id,
            'full_title': title
        }
    except Exception:
        return None

def group_measurements(df):
    """將相同情況的多次測量分組"""
    columns = df.columns[1:]  # 假設第一列是參數名稱
    measurements = defaultdict(list)
    titles = defaultdict(list)
    
    # 將所有測量按基本識別碼分組
    for col in columns:
        parsed = parse_title(col)
        if parsed:
            p1_value = df.loc[df.iloc[:, 0] == 'Decay rate (b)', col].values
            # p1_value = df.loc[df.iloc[:, 0] == 'p1', col].values
            if len(p1_value) > 0 and not pd.isna(p1_value[0]):
                measurements[parsed['base_id']].append(float(p1_value[0]))
                titles[parsed['base_id']].append(parsed['full_title'])
    
    return measurements, titles

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

def find_matching_pairs(measurements, titles):
    """找到配對的測量組"""
    pairs = []
    processed = set()
    
    for base_id1, values1 in measurements.items():
        if base_id1 in processed:
            continue
            
        parsed1 = parse_title(titles[base_id1][0])
        
        # 尋找匹配的另一組測量
        for base_id2, values2 in measurements.items():
            if base_id2 in processed:
                continue
                
            parsed2 = parse_title(titles[base_id2][0])
            
            # 檢查是否配對（相同受試者、不同腳型h/s、相同姿勢）
            if (parsed1['subject_id'] == parsed2['subject_id'] and
                parsed1['leg_type'] != parsed2['leg_type'] and
                parsed1['pose'] == parsed2['pose']):
                
                valid_values1, outliers1, mean1 = remove_outliers_and_average(values1)
                valid_values2, outliers2, mean2 = remove_outliers_and_average(values2)
                
                # 確定健康側和患側
                if parsed1['leg_type'] == 'h':
                    healthy_data = {
                        'mean': mean1,
                        'values': valid_values1,
                        'outliers': outliers1,
                        'titles': titles[base_id1]
                    }
                    spastic_data = {
                        'mean': mean2,
                        'values': valid_values2,
                        'outliers': outliers2,
                        'titles': titles[base_id2]
                    }
                else:
                    healthy_data = {
                        'mean': mean2,
                        'values': valid_values2,
                        'outliers': outliers2,
                        'titles': titles[base_id2]
                    }
                    spastic_data = {
                        'mean': mean1,
                        'values': valid_values1,
                        'outliers': outliers1,
                        'titles': titles[base_id1]
                    }
                
                pairs.append({
                    'subject_id': parsed1['subject_id'],
                    'pose': parsed1['pose'],
                    'healthy': healthy_data,
                    'spastic': spastic_data
                })
                
                processed.add(base_id1)
                processed.add(base_id2)
                break
    
    return pairs

def perform_analysis(df):
    """執行完整的分析流程"""
    # 將測量分組
    measurements, titles = group_measurements(df)
    
    # 找到配對並處理數據
    pairs = find_matching_pairs(measurements, titles)
    
    if len(pairs) > 0:
        # 進行配對t檢定
        healthy_means = [p['healthy']['mean'] for p in pairs]
        spastic_means = [p['spastic']['mean'] for p in pairs]
        t_stat, p_value = stats.ttest_rel(healthy_means, spastic_means)
        
        return {
            'detailed_results': pairs,
            't_statistic': t_stat,
            'p_value': p_value
        }
    
    return None

def export_results_to_excel(results, output_path):
    """將分析結果匯出到Excel"""
    if not results:
        return False
        
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        # 1. 統計摘要
        summary_data = {
            '統計項目': ['t統計量', 'p值'],
            '數值': [results['t_statistic'], results['p_value']]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='統計摘要', index=False)
        
        # 2. 詳細結果
        detailed_data = []
        for pair in results['detailed_results']:
            detailed_data.append({
                '受試者ID': pair['subject_id'],
                '姿勢': pair['pose'],
                '健康側平均值': pair['healthy']['mean'],
                '痙攣側平均值': pair['spastic']['mean'],
                '差異': pair['healthy']['mean'] - pair['spastic']['mean'],
                '健康側數據數量': len(pair['healthy']['values']),
                '痙攣側數據數量': len(pair['spastic']['values']),
                '健康側離群值數量': len(pair['healthy']['outliers']),
                '痙攣側離群值數量': len(pair['spastic']['outliers']),
                '健康側標題': ', '.join(pair['healthy']['titles']),
                '痙攣側標題': ', '.join(pair['spastic']['titles'])
            })
        pd.DataFrame(detailed_data).to_excel(writer, sheet_name='詳細配對結果', index=False)
        
        # 3. 原始數據（包含離群值標記）
        raw_data = []
        for pair in results['detailed_results']:
            # 健康側數據
            for value in pair['healthy']['values']:
                raw_data.append({
                    '受試者ID': pair['subject_id'],
                    '姿勢': pair['pose'],
                    '類型': '健康側',
                    '數值': value,
                    '是否離群值': '否'
                })
            for value in pair['healthy']['outliers']:
                raw_data.append({
                    '受試者ID': pair['subject_id'],
                    '姿勢': pair['pose'],
                    '類型': '健康側',
                    '數值': value,
                    '是否離群值': '是'
                })
            # 痙攣側數據
            for value in pair['spastic']['values']:
                raw_data.append({
                    '受試者ID': pair['subject_id'],
                    '姿勢': pair['pose'],
                    '類型': '痙攣側',
                    '數值': value,
                    '是否離群值': '否'
                })
            for value in pair['spastic']['outliers']:
                raw_data.append({
                    '受試者ID': pair['subject_id'],
                    '姿勢': pair['pose'],
                    '類型': '痙攣側',
                    '數值': value,
                    '是否離群值': '是'
                })
        
        pd.DataFrame(raw_data).to_excel(writer, sheet_name='原始數據', index=False)
        
        return True

def main():
    # input_file = Path("../../analysis_results.xlsx")
    input_file = Path("../../exponential_results_elec.xlsx")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(f"analysis_results_detailed_{timestamp}.xlsx")
    
    df = load_excel_file(input_file)
    results = perform_analysis(df)
    
    if results:
        print("\n=== 配對t檢定結果 ===")
        print(f"t統計量: {results['t_statistic']:.20f}")
        print(f"p值: {results['p_value']:.20f}")

        if export_results_to_excel(results, output_file):
            print(f"\n詳細結果已匯出到: {output_file}")
        else:
            print("\n匯出結果時發生錯誤")
    else:
        print("\n無法執行分析。")

if __name__ == "__main__":
    main()