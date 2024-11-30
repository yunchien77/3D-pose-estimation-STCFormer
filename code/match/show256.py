import pandas as pd
import matplotlib.pyplot as plt

def plot_angle_variation(input_file, input_file2):
    # 讀取CSV文件
    df = pd.read_excel(input_file)
    df2 = pd.read_excel(input_file2)
    # 繪製角度變化圖
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # # 左腳
    # axes[0].plot(df['left_angles'], label='3D_Angles', color='orange')
    # axes[0].set_xlabel('Time Step')
    # axes[0].set_ylabel('Angle (degrees)')
    # axes[0].set_title('Left Angle Variation Over Time')
    # axes[0].legend()
    
    # # 右腳
    # axes[1].plot(df['right_angles'], label='3D_Angles')
    # axes[1].set_xlabel('Time Step')
    # axes[1].set_ylabel('Angle (degrees)')
    # axes[1].set_title('Right Angle Variation Over Time')
    # axes[1].legend()

    # 左腳
    axes[0].plot(df.iloc[:, 154], label='3D_Angles', color='orange')
    axes[0].plot(df2.iloc[:, 154], label='elec_Angles', color='blue')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Angle (degrees)')
    axes[0].set_title('Left Angle Variation Over Time')
    axes[0].legend()
    axes[0].grid(True)
    
    # 右腳
    axes[1].plot(df.iloc[:, 155], label='3D_Angles', color='orange')
    axes[1].plot(df2.iloc[:, 155], label='elec_Angles', color='blue')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Angle (degrees)')
    axes[1].set_title('Right Angle Variation Over Time')
    axes[1].legend()
    axes[1].grid(True)
    
    # 調整子圖之間的間距
    plt.tight_layout()
    
    # 設定 y 軸範圍
    # axes[0].set_ylim(60, 180)
    # axes[1].set_ylim(60, 180)
    
    # 顯示圖表
    plt.show()

# 使用函數
input_file = 'code/3dData_256.xlsx'  # 使用插值後的文件
input_file2 = 'code/elecData.xlsx'
plot_angle_variation(input_file, input_file2)