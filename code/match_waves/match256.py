import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from dtaidistance import dtw
import pandas as pd

def load_csv_data(file_path, column_name):
    df = pd.read_csv(file_path)
    return df[column_name].values

def load_txt_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()[1:]  # 跳過標題行
    data = [float(line.split(',')[1]) for line in lines]
    return np.array(data)

def normalize_sequence(seq):
    """对序列进行Z-score标准化"""
    return zscore(seq)

def find_best_match_dtw(long_seq, short_seq, window=None):
    """使用DTW找到最佳匹配位置"""
    n, m = len(long_seq), len(short_seq)
    min_dist = float('inf')
    best_start = 0
    
    for i in range(0, n - m + 1, 10):  # 使用步長10来加速計算
        dist = dtw.distance(long_seq[i:i+m], short_seq, window=window)
        if dist < min_dist:
            min_dist = dist
            best_start = i
    
    # 在最佳匹配周圍进行精细搜索
    for i in range(max(0, best_start-10), min(n-m+1, best_start+11)):
        dist = dtw.distance(long_seq[i:i+m], short_seq, window=window)
        if dist < min_dist:
            min_dist = dist
            best_start = i
    
    return best_start

def main():
    # 加載數據
    file1_path = 'code/Angles_3D_256fps.csv'  # 新生成的256fps数据文件
    file2_path = 'code/P21 R-2.txt'  # 原始TXT数据文件
    
    # 根據檔案名稱判斷應該使用的列名
    if 'R' in file2_path:
        column_name = 'right_angles'
    elif 'L' in file2_path:
        column_name = 'left_angles'
    else:
        column_name = 'unknown_column'  # 若無法判斷，則設定為未知列名

    print(f"Selected column name: {column_name}")

    seq1 = load_csv_data(file1_path, column_name)
    seq2 = load_txt_data(file2_path)
    
    if 'R' in file2_path:
        seq2 = 180 - seq2

    # 確保seq1是較長的序列
    if len(seq2) > len(seq1):
        seq1, seq2 = seq2, seq1

    # 標準化序列
    seq1_norm = normalize_sequence(seq1)
    seq2_norm = normalize_sequence(seq2)

    # 使用DTW找到最佳匹配位置
    dtw_window = 10  # DTW窗口大小
    best_start = find_best_match_dtw(seq1_norm, seq2_norm, window=dtw_window)
    best_end = best_start + len(seq2)

    print(f"最佳匹配起始位置：第 {best_start} 帧")
    print(f"最佳匹配结束位置：第 {best_end} 帧")


    # 繪圖
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    # 繪製整體匹配結果
    ax1.plot(seq1, label='long sequence', alpha=0.7)
    ax1.plot(range(best_start, best_end), seq1[best_start:best_end], label='matching area', linewidth=2)
    ax1.plot(range(best_start, best_end), seq2, label='short sequence', linestyle='--')
    ax1.legend()
    ax1.set_title('Sequence matching results')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Angle (degrees)')

    # 繪製匹配區域的放大圖
    ax2.plot(seq1[best_start:best_end], label='Long sequence matching part')
    ax2.plot(seq2, label='short sequence')
    ax2.legend()
    ax2.set_title('Enlarged view of matching area')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Angle (degrees)')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()