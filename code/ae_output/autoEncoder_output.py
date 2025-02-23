import numpy as np
import scipy
from scipy.signal import savgol_filter
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.interpolate import interp1d
from keras.saving import register_keras_serializable

@register_keras_serializable()
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def detect_baseline(signal, window_size=20, high_value_threshold=170, tolerance=30):
    # 計算信號的局部變異度
    local_var = np.array([np.std(signal[max(0, i - 5):min(len(signal), i + 6)])
                          for i in range(len(signal))])

    # 找出穩定段（局部變異度小的區域）
    stable_regions = local_var < 0.5

    # 在穩定區域中尋找基準值
    valid_baselines = []

    for i in range(len(signal) - window_size):
        # 檢查是否為穩定區域
        if np.all(stable_regions[i:i + window_size]):
            current_level = np.mean(signal[i:i + window_size])

            # 檢查是否不在高角度範圍內
            if current_level < (high_value_threshold - tolerance):
                valid_baselines.append(current_level)

    if valid_baselines:
        # 返回所有有效基準值的中位數
        return np.median(valid_baselines)

    # 如果沒有找到合適的基準值，使用非高角度區域的中位數作為備選
    low_values = signal[signal < (high_value_threshold - tolerance)]
    if len(low_values) > 0:
        return np.median(low_values)

    # 如果上述方法都失敗，返回信號末端的值
    return signal[len(signal) - 3]

def get_first_two_waves(data, window_size=10, valley_threshold=150):
    """
    找到數據中完整的波 [第一波谷 -> 第二波谷 -> 第三波谷] 並返回第二波的結束位置
    新增局部最小值判斷，必須前後 window_size 個點都比當前點大，且第一個波谷必須小於 valley_threshold。
    """
    data = data.squeeze()
    baseline = detect_baseline(data)
    print(f"Baseline detected: {baseline}")

    # 初始化
    current_wave = {"first_start": None, "second_start": None, "third_start": None}
    looking_for = "valley"  # 初始狀態為尋找波谷
    valley_count = 0  # 用來追蹤已經找到的波谷數量

    for i in range(window_size, len(data) - window_size):
        if looking_for == "valley":
            # 檢查當前點是否為局部最小值
            if all(data[i] < data[i - j] and data[i] < data[i + j] for j in range(1, window_size + 1)):
                # 確保第一個波谷的數值小於 valley_threshold
                if valley_count == 0 and data[i] < valley_threshold:
                    current_wave["first_start"] = i
                    valley_count += 1

                # 第二個波谷（無需數值小於 valley_threshold 的限制）
                elif valley_count == 1:
                    current_wave["second_start"] = i
                    valley_count += 1

                # 第三個波谷，作為第二波的結束點
                elif valley_count == 2:
                    current_wave["third_start"] = i
                    valley_count += 1
                    return current_wave["third_start"]

    # 如果未找到三個波谷，返回數據長度
    print(f"Only {valley_count} valley(s) detected.")
    return len(data)

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
    # 設定檔案路徑
    input_file_path = 'test.csv'
    model_path = 'autoencoder_model.h5'
    output_file_path = 'processed_output.xlsx'

    # 載入模型
    model = load_model(model_path, custom_objects={'mse': mse})
    print("模型載入成功")

    # 讀取輸入數據
    input_data = pd.read_csv(input_file_path).to_numpy().T
    print(f"原始輸入數據形狀: {input_data.shape}")

    # 保存完整原始數據
    full_original_data = input_data.copy()

    # 使用Savitzky-Golay濾波器
    filter_data = savgol_filter(full_original_data, 57, 3, mode='nearest')

    # 找出第二波結束的索引
    start_idx = 0
    end_idx = get_first_two_waves(filter_data)
    print(f"end_idx: {end_idx}")

    # 截取指定範圍的數據
    if end_idx is None:
        end_idx = input_data.shape[1]
    input_data = input_data[:, start_idx:end_idx]
    print(f"截取後數據形狀: {input_data.shape}")

    # 插值到1000個點
    interpolated_data = interpolate_data(input_data, target_length=1000)
    print(f"插值後數據形狀: {interpolated_data.shape}")

    # 標準化數據
    normalized_data, mean, std = normalize_data(interpolated_data)
    print(f"標準化後數據形狀: {normalized_data.shape}")

    # 重塑數據以符合模型輸入要求
    model_input = normalized_data.reshape(normalized_data.shape[0], normalized_data.shape[1], 1)
    print(f"模型輸入數據形狀: {model_input.shape}")

    # 預測
    predictions = model.predict(model_input)
    print(f"預測結果形狀: {predictions.shape}")

    # 反標準化預測結果
    denormalized_predictions = denormalize_data(predictions.squeeze(), mean, std)
    
    # 儲存結果
    output_df = pd.DataFrame(denormalized_predictions.T)
    output_df.to_excel(output_file_path, index=False)
    print(f"結果已儲存至 {output_file_path}")

    # 視覺化結果（可選）
    import matplotlib.pyplot as plt

    # 決定要顯示的樣本數
    n_samples = min(3, denormalized_predictions.shape[0])

    plt.figure(figsize=(15, 4*n_samples))

    # 為原始數據和預測結果創建x軸
    x_full = np.arange(full_original_data.shape[1])
    x_pred = np.linspace(start_idx, end_idx, denormalized_predictions.shape[1])

    for i in range(n_samples):
        plt.subplot(n_samples, 1, i+1)

        # 繪製完整原始數據
        plt.plot(x_full, full_original_data[i], label='Original Data', alpha=0.5, color='gray')

        # 突出顯示處理的部分
        plt.plot(x_full[start_idx:end_idx],
                full_original_data[i, start_idx:end_idx],
                label='Selected Range',
                color='blue',
                alpha=0.7)

        # 繪製預測結果
        plt.plot(x_pred, denormalized_predictions[i],
                label='Processed Output',
                color='red',
                alpha=0.7)

        # 添加垂直線標示處理範圍
        plt.axvline(x=start_idx, color='green', linestyle='--', alpha=0.5)
        plt.axvline(x=end_idx, color='green', linestyle='--', alpha=0.5)

        plt.title(f'Sample {i+1}')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()