import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from scipy.fftpack import fft, fftfreq
import scipy.signal
from scipy.signal import find_peaks

# 1. 數據讀取
folder_path = r'E:\output\002\002R_sb_1\002R_sb_1_3'
file_elec = r'elec.txt'
dir_elec = os.path.join(folder_path, file_elec)

df_elec = pd.read_csv(dir_elec, sep=",")
time = np.array(df_elec['Time'])
if 'L' in dir_elec:
    elec = np.array(df_elec['Linear Transformer Gonio G'])   #L
else: 
    elec = np.array(180 - df_elec['Linear Transformer Gonio G'])   #R

# 調整角度值
adjustment = 180 - elec[0]  
elec = elec + adjustment    

# 2. 數據平滑處理
elec = scipy.signal.savgol_filter(elec, 13, 3, mode='nearest')

# Reset time to start from zero
time_zeroed = time - time[0]

# 3. 找出局部極值
max_index = argrelextrema(elec, np.greater)
min_index = argrelextrema(elec, np.less)

# 4. 找出平穩階段
def find_stable_regions(data, window_size=10, threshold=0.5):
    """
    找出數據中的平穩區域
    window_size: 滑動窗口大小
    threshold: 判定為平穩的變化閾值
    """
    variations = []
    means = []
    
    # 計算每個窗口的變化程度
    for i in range(len(data) - window_size + 1):
        window = data[i:i+window_size]
        variation = np.std(window)  # 使用標準差來衡量變化程度
        mean = np.mean(window)
        variations.append(variation)
        means.append(mean)
    
    # 找出變化小於閾值的區域
    stable_regions = []
    current_region = []
    
    for i, var in enumerate(variations):
        if var < threshold:
            current_region.append(means[i])
        elif current_region:
            stable_regions.append(np.mean(current_region))
            current_region = []
    
    if current_region:
        stable_regions.append(np.mean(current_region))
    
    return np.array(stable_regions)

# 過濾平穩區域
def filter_stable_regions(time, elec, begin_angle, rest_angle, tolerance=2):
    """
    只過濾序列開頭和結尾的平穩區段
    
    Parameters:
    - time: 時間序列
    - elec: 角度序列
    - begin_angle: 起始角度
    - rest_angle: 靜息角度
    - tolerance: 判定為平穩區域的閾值
    
    Returns:
    - filtered_time: 過濾後的時間序列
    - filtered_elec: 過濾後的角度序列
    """
    # 找出序列開頭的平穩區段結束點
    start_idx = 0
    for i in range(len(elec)):
        if abs(elec[i] - begin_angle) > tolerance:
            start_idx = i
            break
    
    # 從後往前找出序列結尾的平穩區段起始點
    end_idx = len(elec) - 1
    for i in range(len(elec)-1, -1, -1):
        if abs(elec[i] - rest_angle) > tolerance:
            end_idx = i
            break
    
    # 過濾數據
    filtered_time = time[start_idx:end_idx+1]
    filtered_elec = elec[start_idx:end_idx+1]
    
    return filtered_time, filtered_elec

# 使用極小值點來找平穩區域
stable_regions = find_stable_regions(elec[min_index])
stable_regions = np.sort(stable_regions)  # 排序穩定區域的值

# 如果找到至少兩個平穩區域
if len(stable_regions) >= 2:
    rest_angle = stable_regions[0]  # 最小的平穩值為休息角度
    begin_angle = stable_regions[-1]  # 最大的平穩值為起始角度
else:
    # 如果沒有找到足夠的平穩區域，使用最大和最小值
    rest_angle = np.min(elec[min_index])
    begin_angle = np.max(elec[min_index])

print(f"找到的平穩區域值: {stable_regions}")
print(f'\n(起始角度: {begin_angle}, 靜息角度: {rest_angle})')

# 過濾平穩區域
filtered_time, filtered_elec = filter_stable_regions(time_zeroed, elec, begin_angle, rest_angle)

# 在過濾後的數據上找極值點
max_index_filtered = argrelextrema(filtered_elec, np.greater)[0]
min_index_filtered = argrelextrema(filtered_elec, np.less)[0]

rest_angle = 121.7485166
begin_angle = 180.1

# 5. 計算相關參數
threshold = rest_angle
a = filtered_elec[max_index_filtered[1]]
b = filtered_elec[min_index_filtered[0]]
c = filtered_elec[min_index_filtered[1]]


print(f"a={a}, b={b}, c={c}")

a0 = begin_angle - threshold
a1 = begin_angle - b
a2 = a - b
a3 = a - threshold
a4 = a - c
print(f"A0={a0}, A1={a1}, A2={a2}, A3={a3}, A4={a4}")

num_waves = len(max_index_filtered)
print(f"Number of waves: {num_waves}")

p1 = a1 / (1.6 * a0)
p2 = num_waves
p4 = a3
p5 = a4 / (1.6 * a3)
print(f"p1={p1}, p4={p4}, p5={p5}")

# 繪圖
plt.figure(figsize=(12, 8))

# 繪製原始信號
plt.plot(time_zeroed, elec, 'black', label='Original Signal')

# 繪製過濾後的信號
# plt.plot(filtered_time, filtered_elec, 'black', label='Filtered Signal')
plt.scatter(filtered_time[max_index_filtered], filtered_elec[max_index_filtered], 
           c='red', s=30, label='Local Max')
plt.scatter(filtered_time[min_index_filtered], filtered_elec[min_index_filtered], 
           c='blue', s=30, label='Local Min')

# 添加平穩區域的標記
plt.axhline(y=rest_angle, color='#F5D5B1', linestyle='--', label='Rest Angle')
plt.axhline(y=begin_angle, color='#F4C2B1', linestyle='--', label='Begin Angle')

# 添加關鍵點
plt.axhline(y=a, color='#BCF5D4', linestyle='-.', label='a')
plt.axhline(y=b, color='#A5D7CE', linestyle='-.', label='b')
plt.axhline(y=c, color='#96BBD6', linestyle='-.', label='c')

# 添加計算結果
bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="white", alpha=0.8)
result_text = (f"A0={a0:.3f}\nA1={a1:.3f}\nA2={a2:.3f}\nA3={a3:.3f}\nA4={a4:.3f}")
plt.text(0.05, 0.22, result_text, transform=plt.gca().transAxes, fontsize=12, 
         verticalalignment='top', bbox=bbox_props)

result_text = (f"p1={p1:.3f}\np4={p4:.3f}\np5={p5:.3f}")
plt.text(0.27, 0.22, result_text, transform=plt.gca().transAxes, fontsize=12, 
         verticalalignment='top', bbox=bbox_props)

plt.legend(loc='upper right', fontsize='medium')
plt.title('Angle Variation Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Angle (degrees)')
plt.ylim(60, 190)
plt.grid(True, alpha=0.3)

# 保存圖片
output_path = os.path.join(folder_path, 'para.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')

print(f"圖片已保存至: {output_path}")

plt.show()