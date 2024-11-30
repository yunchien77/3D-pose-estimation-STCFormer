import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import scipy.signal

folder_path = r'E:/video/20240815_020/txt'
file_elec = r'P20 L-1.txt'

dir_elec = os.path.join(folder_path, file_elec)
df_elec = pd.read_csv(dir_elec, sep=",")

#elec = np.array(180 - df_elec['Linear Transformer Gonio G'])
elec = np.array(180-df_elec['Linear Transformer Gonio G'])

plt.figure(figsize=(12, 4))
plt.plot(elec, label='Left Foot Angle', color='green')

plt.xlabel('Time Step')
plt.ylabel('Angle (degrees)')
plt.title('Angle Variation Over Time for Left and Right Foot')
plt.legend()


# fig, axes = plt.subplots(1, 2, figsize=(20, 6))

# # 左腳
# axes[0].plot(elec, label='Electrogoniometer Angle', color='green')
# axes[0].set_xlabel('Time Step')
# axes[0].set_ylabel('Angle (degrees)')
# axes[0].set_title('Left Angle Variation Over Times')
# axes[0].legend()
# #axes[0].set_ylim(60, 180)  # 設定 y 軸範圍


# # 右腳
# axes[1].plot(elec, label='Electrogoniometer Angle', color='green')
# axes[1].set_xlabel('Time Step')
# axes[1].set_ylabel('Angle (degrees)')
# axes[1].set_title('Right Angle Variation Over Times')
# axes[1].legend()
# #axes[1].set_ylim(60, 180)  # 設定 y 軸範圍

# # 調整子圖之間的間距
# plt.tight_layout()

# 顯示圖表
plt.show()