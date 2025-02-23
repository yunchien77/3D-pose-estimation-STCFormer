import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.interpolate import interp1d
from keras.saving import register_keras_serializable
import os
import matplotlib.pyplot as plt

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
    """Recursively find all Angles_3D_256fps.csv files"""
    angle_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file == 'Angles_3D_256fps.csv':
                angle_files.append(os.path.join(root, file))
    return angle_files

def process_angle_data(file_path):
    """Process angle data based on file path"""
    df = pd.read_csv(file_path)
    
    # Choose column based on path
    if 'L' in file_path:
        angle_column = 'left_angles'
    elif 'R' in file_path:
        angle_column = 'right_angles'
    else:
        raise ValueError("Cannot determine angle column")
    
    angles = df[angle_column].to_numpy()
    
    # Adjust first value to 180
    adjustment = 180 - angles[0]
    adjusted_angles = angles + adjustment
    
    return adjusted_angles.reshape(-1, 1).T

def process_single_file(input_file_path, model, target_length=1000):
    """Process a single file through the autoencoder"""
    print(f"Processing file: {input_file_path}")

    # Generate output paths
    output_dir = os.path.dirname(input_file_path)
    output_file_name = os.path.basename(output_dir)
    output_file_path = os.path.join(output_dir, 'ae_fullwaves_1.xlsx')
    output_fig_path = os.path.join(output_dir, 'ae_fullwaves_1.png')

    # Read and process input data
    input_data = process_angle_data(input_file_path)
    print(f"Original input data shape: {input_data.shape}")

    # Save full original data
    full_original_data = input_data.copy()

    # Interpolate to target length
    interpolated_data = interpolate_data(input_data, target_length=target_length)
    print(f"Interpolated data shape: {interpolated_data.shape}")

    # Normalize data
    normalized_data, mean, std = normalize_data(interpolated_data)
    print(f"Normalized data shape: {normalized_data.shape}")

    # Reshape for model input
    model_input = normalized_data.reshape(normalized_data.shape[0], normalized_data.shape[1], 1)
    print(f"Model input shape: {model_input.shape}")

    # Get predictions
    predictions = model.predict(model_input)
    print(f"Prediction shape: {predictions.shape}")

    # Denormalize predictions
    denormalized_predictions = denormalize_data(predictions.squeeze(), mean, std)
    
    # Interpolate back to original length
    processed_data = interpolate_data(
        denormalized_predictions.reshape(1, -1), 
        target_length=full_original_data.shape[1]
    )[0]

    # Save results
    output_df = pd.DataFrame({
        'Processed_Output': processed_data,
        'Original_Signal': full_original_data[0]
    })
    output_df.to_excel(output_file_path, index=False)
    print(f"Results saved to {output_file_path}")

    # Visualize results
    plt.figure(figsize=(15, 4))
    
    x_range = np.arange(len(processed_data))
    
    # Plot original and processed data
    plt.plot(x_range, full_original_data[0], label='Original Data', alpha=0.5, color='gray')
    plt.plot(x_range, processed_data, label='Processed Output', color='red', alpha=0.7)

    plt.title(f'Data Processing for {output_file_name}')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_fig_path)
    plt.close()

def interpolate_data(data, target_length=1000, kind='linear'):
    """Interpolate data to specified length"""
    def interpolate_row(row):
        x_old = np.arange(len(row))
        f = interp1d(x_old, row, kind=kind, fill_value="extrapolate")
        x_new = np.linspace(0, len(row)-1, target_length)
        return f(x_new)

    return np.apply_along_axis(interpolate_row, axis=1, arr=data)

def normalize_data(data):
    """Normalize data"""
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    return (data - mean) / (std + 1e-8), mean, std

def denormalize_data(normalized_data, mean, std):
    """Denormalize data"""
    return normalized_data * (std + 1e-8) + mean

def main():
    # Set base folder path
    base_folder = r'D:/para/001'  # Adjust path as needed
    model_path = 'ae_model/improved_autoencoder_1.h5'

    # Find all Angles_3D_256fps.csv files
    angle_files = find_angle_csv_files(base_folder)
    
    if not angle_files:
        print("No Angles_3D_256fps.csv files found")
        return

    # Load model
    # model = load_model(model_path, custom_objects={'mse': mse})

    model = load_model(model_path, custom_objects={'custom_peak_loss': custom_peak_loss})
    print("Model loaded successfully")

    # Process each file
    for file_path in angle_files:
        try:
            process_single_file(file_path, model)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    main()