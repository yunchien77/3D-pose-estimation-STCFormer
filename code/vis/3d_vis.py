import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import os 


current_directory = os.getcwd()
print("Current Working Directory:", current_directory)

#file_path = 'demo/output/011R_sb_2_5/output_3D/output_keypoints_3d.npz'
file_path = 'D:/output/021/021R_sb_2_7/output_3D/output_keypoints_3d.npz'

f = np.load(file_path)
print('Array Name: ', f.files)  # Array Name:  ['reconstruction']

# Access the 'reconstruction' array
reconstruction = f['reconstruction']

# Print the shape of the 'reconstruction' array
print('Shape of reconstruction array:', reconstruction.shape)

# Print the shape of the first element in 'reconstruction'
print('Shape of the first element in reconstruction:', reconstruction[0].shape)

# Print the first element to understand its structure
print('First element in reconstruction:', reconstruction[0])

angle_right = []
angle_left = []

for data in f['reconstruction'][:-1]:
    # 右腳關節點
    x2_r, y2_r, z2_r = data[1]
    x3_r, y3_r, z3_r = data[2]
    x4_r, y4_r, z4_r = data[3]

    # 左腳關節點
    x5_l, y5_l, z5_l = data[4]
    x6_l, y6_l, z6_l = data[5]
    x7_l, y7_l, z7_l = data[6]

    # 身體中線三點成平面
    x1_m, y1_m, z1_m = data[0]
    x8_m, y8_m, z8_m = data[7]
    x9_m, y9_m, z9_m = data[8]

    vec_1 = np.array([x1_m, y1_m, z1_m])
    vec_8 = np.array([x8_m, y8_m, z8_m])
    vec_9 = np.array([x9_m, y9_m, z9_m])

    mu = vec_1 - vec_8
    mv = vec_9 - vec_8

    # 中線平面的法向量
    n = np.cross(mu, mv)

    vec_2 = np.array([x2_r, y2_r, z2_r])
    vec_3 = np.array([x3_r, y3_r, z3_r])
    vec_4 = np.array([x4_r, y4_r, z4_r])

    vec_5 = np.array([x5_l, y5_l, z5_l])
    vec_6 = np.array([x6_l, y6_l, z6_l])
    vec_7 = np.array([x7_l, y7_l, z7_l])

    ru = vec_2 - vec_3
    rv = vec_4 - vec_3
    lu = vec_5 - vec_6
    lv = vec_7 - vec_6

    # 右腳 ru、rv 的水平分量
    norm_ru = np.linalg.norm(ru)
    norm_rv = np.linalg.norm(rv)
    proj_u2n_r = (np.dot(ru, n) * n) / norm_ru
    ru_h = ru - proj_u2n_r
    proj_v2n_r = (np.dot(rv, n) * n) / norm_rv
    rv_h = rv - proj_v2n_r

    dot_product_r = np.dot(ru_h, rv_h)
    norm_ru_h = np.linalg.norm(ru_h)
    norm_rv_h = np.linalg.norm(rv_h)

    cos_theta_r = dot_product_r / (norm_ru_h * norm_rv_h)
    angle_rad_r = np.arccos(cos_theta_r)
    angle_deg_r = np.degrees(angle_rad_r)

    # 左腳 lu、lv 的水平分量
    norm_lu = np.linalg.norm(lu)
    norm_lv = np.linalg.norm(lv)
    proj_u2n_l = (np.dot(lu, n) * n) / norm_lu
    lu_h = lu - proj_u2n_l
    proj_v2n_l = (np.dot(lv, n) * n) / norm_lv
    lv_h = lv - proj_v2n_l

    dot_product_l = np.dot(lu_h, lv_h)
    norm_lu_h = np.linalg.norm(lu_h)
    norm_lv_h = np.linalg.norm(lv_h)

    cos_theta_l = dot_product_l / (norm_lu_h * norm_lv_h)
    angle_rad_l = np.arccos(cos_theta_l)
    angle_deg_l = np.degrees(angle_rad_l)

    angle_right.append(angle_deg_r)
    angle_left.append(angle_deg_l)

# 儲存角度數據
csv_filename = 'Angles_3D.csv'
output_dir = os.path.join(file_path, "..", csv_filename)

with open(output_dir, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['left_angles', 'right_angles'])  # 寫入標題行

    for row in zip(angle_left, angle_right):
      csv_writer.writerow(row)

# 繪製角度變化圖
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 左腳
axes[0].plot(angle_left, label='3D_Angles', color='orange')
axes[0].set_xlabel('Time Step')
axes[0].set_ylabel('Angle (degrees)')
axes[0].set_title('Left Angle Variation Over Time')
axes[0].legend()

# 右腳
axes[1].plot(angle_right, label='3D_Angles')
axes[1].set_xlabel('Time Step')
axes[1].set_ylabel('Angle (degrees)')
axes[1].set_title('Right Angle Variation Over Time')
axes[1].legend()

# 調整子圖之間的間距
plt.tight_layout()
# 設定 y 軸範圍
plt.ylim(60, 180) 


# 保存圖片
plot_filename = '3d.png'
output_dir = os.path.dirname(os.path.dirname(file_path))
print(f"Output Directory: {output_dir}")
plot_path = os.path.join(output_dir, plot_filename)
plt.savefig(plot_path)

print(f"Plot saved as: {plot_path}")

# 顯示圖表
plt.show()


