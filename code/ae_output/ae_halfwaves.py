import numpy as np
import scipy
from scipy.signal import savgol_filter
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.interpolate import interp1d
from keras.saving import register_keras_serializable
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# @register_keras_serializable()
# def mse(y_true, y_pred):
#     return tf.reduce_mean(tf.square(y_true - y_pred))

@register_keras_serializable()
def custom_peak_loss(y_true, y_pred):
    # MSE損失
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # 計算一階導數
    dy_true = y_true[:, 1:] - y_true[:, :-1]
    dy_pred = y_pred[:, 1:] - y_pred[:, :-1]
    
    # 導數的MSE損失（保持波形特徵）
    gradient_loss = tf.reduce_mean(tf.square(dy_true - dy_pred))
    
    # 計算二階導數（捕捉曲率變化）
    d2y_true = dy_true[:, 1:] - dy_true[:, :-1]
    d2y_pred = dy_pred[:, 1:] - dy_pred[:, :-1]
    curvature_loss = tf.reduce_mean(tf.square(d2y_true - d2y_pred))
    
    # 組合損失
    total_loss = mse_loss + 0.3 * gradient_loss + 0.1 * curvature_loss
    return total_loss

def find_angle_csv_files(base_path):
    """遞迴尋找所有子資料夾中的 Angles_3D_256fps.csv 檔案"""
    angle_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file == 'Angles_3D_256fps.csv':
                angle_files.append(os.path.join(root, file))
    return angle_files

def process_angle_data(file_path):
    """根據檔案路徑選擇正確的角度欄位"""
    df = pd.read_csv(file_path)
    
    if 'L' in file_path:
        angle_column = 'left_angles'
    elif 'R' in file_path:
        angle_column = 'right_angles'
    else:
        raise ValueError("無法判斷角度欄位")
    
    angles = df[angle_column].to_numpy()
    adjustment = 180 - angles[0]
    adjusted_angles = angles + adjustment
    
    return adjusted_angles.reshape(-1, 1).T

class SignalProcessor:
    def __init__(self, input_file_path, model):
        self.input_file_path = input_file_path
        self.model = model
        self.output_dir = os.path.dirname(input_file_path)
        self.output_file_name = os.path.basename(self.output_dir)
        self.input_data = process_angle_data(input_file_path)
        self.full_original_data = self.input_data.copy()
        self.filter_data = savgol_filter(self.full_original_data, 101, 3, mode='nearest')
        self.start_idx = 0
        self.end_idx = None
        self.fig = None
        self.ax = None
        self.confirmed = False
        
    def detect_initial_indices(self):
        """使用原有的get_first_two_waves函數檢測初始end_idx"""
        self.end_idx = get_first_two_waves(self.filter_data)
        print(f"自動檢測到的end_idx: {self.end_idx}")
        
    def plot_for_confirmation(self):
        """繪製信號並等待使用者確認"""
        self.fig, self.ax = plt.subplots(figsize=(15, 6))
        
        # 繪製原始數據和過濾後的數據
        self.ax.plot(self.full_original_data[0], label='Original Data', alpha=0.5, color='gray')
        self.ax.plot(self.filter_data[0], label='Filtered Data', alpha=0.7, color='green')
        
        # 標記start_idx和end_idx位置
        if self.end_idx is not None:
            self.ax.axvline(x=self.start_idx, color='blue', linestyle='--', label='Start Index')
            self.ax.axvline(x=self.end_idx, color='red', linestyle='--', label='End Index')
        
        self.ax.set_title('Please confirm the starting point and ending point location\nClick "Confirm" to confirm, or click on the image to select a new location\n(First click on the starting point, then click on the ending point)')
        self.ax.set_xlabel('Time Steps')
        self.ax.set_ylabel('Value')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        
        # 添加確認按鈕
        confirm_ax = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.confirm_button = Button(confirm_ax, 'Confirm')
        self.confirm_button.on_clicked(self.confirm)
        
        # 連接滑鼠點擊事件
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        
        plt.show()
        
    def onclick(self, event):
        """處理滑鼠點擊事件"""
        if event.inaxes != self.ax or self.confirmed:
            return
        
        if event.button == 1:  # 左鍵點擊
            if self.start_idx == 0:  # 首先設置start_idx
                self.start_idx = int(event.xdata)
                self.ax.axvline(x=self.start_idx, color='blue', linestyle='--', label='New Start')
            else:  # 然後設置end_idx
                self.end_idx = int(event.xdata)
                self.ax.clear()
                self.plot_current_state()
            plt.draw()
            
    def plot_current_state(self):
        """更新圖表顯示當前狀態"""
        self.ax.plot(self.full_original_data[0], label='Original Data', alpha=0.5, color='gray')
        self.ax.plot(self.filter_data[0], label='Filtered Data', alpha=0.7, color='green')
        self.ax.axvline(x=self.start_idx, color='blue', linestyle='--', label='Start Index')
        if self.end_idx is not None:
            self.ax.axvline(x=self.end_idx, color='red', linestyle='--', label='End Index')
        self.ax.set_title('Please confirm the start and end point locations\nClick "Confirm" to confirm, or click on the image to select a new location')
        self.ax.set_xlabel('Time Steps')
        self.ax.set_ylabel('Value')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
            
    def confirm(self, event):
        """確認選擇的位置"""
        self.confirmed = True
        plt.close()
        
    def process_data(self):
        """在確認位置後處理數據"""
        if not self.confirmed:
            print("位置未確認，取消處理")
            return
        
        # 確保輸出目錄存在
        os.makedirs(self.output_dir, exist_ok=True)
        output_file_path = os.path.join(self.output_dir, 'ae_half.xlsx')
        output_fig_path = os.path.join(self.output_dir, 'processed_output.png')
        
        # 截取指定範圍的數據
        input_data = self.input_data[:, self.start_idx:self.end_idx]
        
        # 插值到1000個點
        interpolated_data = interpolate_data(input_data, target_length=1000)
        
        # 標準化數據
        normalized_data, mean, std = normalize_data(interpolated_data)
        
        # 重塑數據以符合模型輸入要求
        model_input = normalized_data.reshape(normalized_data.shape[0], normalized_data.shape[1], 1)
        
        # 預測
        predictions = self.model.predict(model_input)
        
        # 反標準化預測結果
        denormalized_predictions = denormalize_data(predictions.squeeze(), mean, std)
        
        # 將預測結果從1000點插值回原始截取範圍的長度
        processed_part = interpolate_data(
            denormalized_predictions.reshape(1, -1), 
            target_length=self.end_idx - self.start_idx
        )[0]
        
        # 準備完整的數據
        processed_data = np.concatenate([
            self.full_original_data[0, :self.start_idx], 
            processed_part, 
            self.full_original_data[0, self.end_idx:]
        ])
        
        # 儲存結果
        output_df = pd.DataFrame({
            'Processed_Output': processed_data,
            'Original_Full_Signal': self.full_original_data[0]
        })
        output_df.to_excel(output_file_path, index=False)
        print(f"結果已儲存至 {output_file_path}")
        
        # 視覺化結果
        self.plot_final_result(processed_part, output_fig_path)
        
    def plot_final_result(self, processed_part, output_fig_path):
        """繪製最終結果"""
        plt.figure(figsize=(15, 4))
        
        x_full = np.arange(self.full_original_data.shape[1])
        x_pred = np.arange(self.start_idx, self.end_idx)
        
        plt.plot(x_full, self.full_original_data[0], label='Original Data', alpha=0.5, color='gray')
        plt.plot(x_full, self.filter_data[0], label='Filtered Data', alpha=0.7, color='green')
        plt.plot(x_full[self.start_idx:self.end_idx],
                self.full_original_data[0, self.start_idx:self.end_idx],
                label='Selected Range',
                color='blue',
                alpha=0.7)
        plt.plot(x_pred, processed_part,
                label='Processed Output',
                color='red',
                alpha=0.7)
        
        plt.axvline(x=self.start_idx, color='blue', linestyle='--', alpha=0.5)
        plt.axvline(x=self.end_idx, color='red', linestyle='--', alpha=0.5)
        
        plt.title(f'Data Processing Result for {self.output_file_name}')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_fig_path)
        plt.close()

def get_first_two_waves(data, window_size=30, valley_threshold=170, prominence_threshold=3, stable_window=30, stable_threshold=170, min_peak_distance=10):
    """
    改進的波形檢測函數，提高波形檢測的魯棒性，並印出詳細的波形檢測資訊
    """
    from scipy.signal import find_peaks, peak_prominences
    import numpy as np

    data = data.squeeze()
    baseline = detect_baseline(data)
    print(f"基準線檢測結果: {baseline}")

    # 找到第一個穩定的區段起始點
    start_index = 0
    for i in range(len(data) - stable_window):
        if np.all(data[i:i+stable_window] < stable_threshold):
            start_index = i
            break

    print(f"找到的穩定起始點: {start_index}")

    # 在穩定起始點之後進行波谷檢測
    subset_data = data[start_index:]
    
    # 嘗試多種策略檢測波形
    strategies = [
        ('策略1：低要求的波谷檢測', 
         lambda d, inv_d: find_peaks(inv_d, distance=min_peak_distance, prominence=0.1)),
        
        ('策略2：寬鬆顯著性', 
         lambda d, inv_d: find_peaks(inv_d, distance=window_size, prominence=0.5)),
        
        ('策略3：基於數據絕對值變化', 
         lambda d, inv_d: find_peaks(np.abs(np.diff(d)), distance=min_peak_distance))
    ]

    valleys = None
    valley_prominences = None

    for strategy_name, strategy in strategies:
        # 反轉信號找峰值（波谷）
        inverted_data = -subset_data
        
        try:
            valleys, _ = strategy(subset_data, inverted_data)
            valley_prominences = peak_prominences(inverted_data, valleys, wlen=200)[0]
            
            # 詳細印出每個策略的波形檢測結果
            print(f"\n{strategy_name} 波形檢測結果:")
            print(f"找到的波谷索引: {valleys}")
            print(f"對應波谷的顯著性: {valley_prominences}")
            print(f"波谷數量: {len(valleys)}")
            
            # 印出每個波谷的詳細資料
            for i, (valley_idx, prominence) in enumerate(zip(valleys, valley_prominences), 1):
                print(f"波谷 {i}:")
                print(f"  索引: {valley_idx}")
                print(f"  在子集數據中的值: {subset_data[valley_idx]}")
                print(f"  顯著性: {prominence}")
            
            # 如果成功找到足夠的波谷，則跳出循環
            if len(valleys) >= 2:
                break
        except Exception as e:
            print(f"{strategy_name} 策略檢測失敗: {e}")
            continue

    # 如果仍然找不到波谷
    if valleys is None or len(valleys) < 2:
        print("\n無法檢測到足夠的波谷，使用替代策略")
        
        # 替代策略：使用波形的顯著變化點
        diff_data = np.diff(subset_data)
        valleys = np.where(np.abs(diff_data) > np.std(diff_data) * 2)[0]
        
        if len(valleys) < 2:
            print("無法檢測波形，返回一半數據")
            return (start_index + len(subset_data))//2

    # 篩選波谷
    significant_valleys = [
        valley for valley, prominence in zip(valleys, valley_prominences) 
        if prominence > prominence_threshold
    ]

    filtered_valleys = [
        valley for valley in significant_valleys 
        if subset_data[valley] < valley_threshold
    ]
    
    print("\n篩選後的波谷:")
    for i, valley in enumerate(filtered_valleys, 1):
        print(f"第 {i} 個顯著波谷:")
        print(f"  索引: {valley}")
        print(f"  在子集數據中的值: {subset_data[valley]}")

    # 如果找到至少兩個波谷
    if len(filtered_valleys) >= 2:
        # 找第二個波谷之後的第三個波谷（第二波的結束）
        third_valley_candidates = [
            valley for valley in significant_valleys 
            if valley > filtered_valleys[1]
        ]
        
        # 如果找到第三個波谷
        if third_valley_candidates:
            final_valley_index = start_index + third_valley_candidates[0]
            print(f"\n最終選定的波形結束索引: {final_valley_index}")
            return final_valley_index
        
        # 若找不到第三個波谷，回傳從穩定起始點開始的數據長度
        final_length = start_index + len(subset_data)
        print(f"\n未找到第三個波谷，回傳預設長度: {final_length}")
        return final_length
    
    print(f"\n只檢測到 {len(filtered_valleys)} 個顯著波谷")
    return start_index + len(subset_data)

def detect_baseline(signal, window_size=20, high_value_threshold=170, tolerance=30):
    # 原有的 detect_baseline 函數實作
    local_var = np.array([np.std(signal[max(0, i - 5):min(len(signal), i + 6)])
                          for i in range(len(signal))])

    stable_regions = local_var < 0.5
    valid_baselines = []

    for i in range(len(signal) - window_size):
        if np.all(stable_regions[i:i + window_size]):
            current_level = np.mean(signal[i:i + window_size])

            if current_level < (high_value_threshold - tolerance):
                valid_baselines.append(current_level)

    if valid_baselines:
        return np.median(valid_baselines)

    low_values = signal[signal < (high_value_threshold - tolerance)]
    if len(low_values) > 0:
        return np.median(low_values)

    return signal[len(signal) - 3]

def interpolate_data(data, target_length=1000, kind='linear'):
    """將數據插值到指定長度"""
    def interpolate_row(row):
        x_old = np.arange(len(row))
        f = interp1d(x_old, row, kind=kind, fill_value="extrapolate")
        x_new = np.linspace(0, len(row)-1, target_length)
        return f(x_new)

    return np.apply_along_axis(interpolate_row, axis=1, arr=data)

def normalize_data(data):
    """標準化數據"""
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    return (data - mean) / (std + 1e-8), mean, std

def denormalize_data(normalized_data, mean, std):
    """反標準化數據"""
    return normalized_data * (std + 1e-8) + mean

def main():
    base_folder = r'D:/para/021'  # 請根據實際路徑調整
    model_path = '..//ae_model/autoencoder_v3.h5'
    
    angle_files = find_angle_csv_files(base_folder)
    
    if not angle_files:
        print("未找到 Angles_3D_256fps.csv 檔案")
        return
    
    # model = load_model(model_path, custom_objects={'mse': mse})
    model = load_model(model_path, custom_objects={'custom_peak_loss': custom_peak_loss})
    print("模型載入成功")
    
    for file_path in angle_files:
        try:
            processor = SignalProcessor(file_path, model)
            processor.detect_initial_indices()
            processor.plot_for_confirmation()
            if processor.confirmed:
                processor.process_data()
        except Exception as e:
            print(f"處理 {file_path} 時發生錯誤: {e}")

if __name__ == "__main__":
    main()