import numpy as np
from scipy.stats import wilcoxon

# 假設這是你的成對數據
data1 = np.array([1.102, 1.169])
data2 = np.array([0.392, 0.4])

# 檢查數據類型
if not all(isinstance(x, (int, float)) for x in data1) or not all(isinstance(x, (int, float)) for x in data2):
    raise ValueError("所有數據必須是數值型。")

# 檢查數據大小是否相同
if len(data1) != len(data2):
    raise ValueError("兩組數據必須具有相同的大小。")
    
# 使用wilcoxon檢驗
stat, p_value = wilcoxon(data1, data2)

print(f'統計值: {stat}, p值: {p_value}')

# 判斷p值是否小於顯著性水平（例如0.05）
alpha = 0.05
if p_value < alpha:
    print("拒絕虛無假設，兩組數據有顯著差異。")
else:
    print("無法拒絕虛無假設，兩組數據無顯著差異。")
