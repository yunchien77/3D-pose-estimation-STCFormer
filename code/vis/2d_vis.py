import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import os 

current_directory = os.getcwd()
print("Current Working Directory:", current_directory)

file_path = 'demo/output/011R_sb_2_5/input_2D/keypoints.npz'
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


angles_r = []
angles_l = []

# n = a['reconstruction'][0].shape[0]

for data in f['reconstruction'][0][:-1]:
    x2 = data[1][0]
    y2 = data[1][1]

    x3 = data[2][0]
    y3 = data[2][1]

    x4 = data[3][0]
    y4 = data[3][1]

    vec2 = np.array([x2, y2])
    vec3 = np.array([x3, y3])
    vec4 = np.array([x4, y4])

    u_r = vec2 - vec3
    v_r = vec4 - vec3

    inner_product_r = np.dot(u_r, v_r)
    length_product_r = (np.sqrt(pow(u_r[0], 2) + pow(u_r[1], 2))) * (np.sqrt(pow(v_r[0], 2) + pow(v_r[1], 2)))

    if inner_product_r != 0 and length_product_r != 0:
          cos_angle_r = inner_product_r / length_product_r
          pi_angle_r = np.arccos(cos_angle_r)
          angle_r = np.degrees(pi_angle_r)

    angles_r.append(angle_r)
    print(f'angle_r : {angle_r}')

    ##############################################

    x5 = data[4][0]
    y5 = data[4][1]

    x6 = data[5][0]
    y6 = data[5][1]

    x7 = data[6][0]
    y7 = data[6][1]

    vec5 = np.array([x5, y5])
    vec6 = np.array([x6, y6])
    vec7 = np.array([x7, y7])

    u_l = vec5 - vec6
    v_l = vec7 - vec6

    inner_product_l = np.dot(u_l, v_l)
    length_product_l = (np.sqrt(pow(u_l[0], 2) + pow(u_l[1], 2))) * (np.sqrt(pow(v_l[0], 2) + pow(v_l[1], 2)))

    if inner_product_l != 0 and length_product_l != 0:
          cos_angle_l = inner_product_l / length_product_l
          pi_angle_l = np.arccos(cos_angle_l)
          angle_l = np.degrees(pi_angle_l)

    angles_l.append(angle_l)
    print(f'angle_l : {angle_l}')

##################################################### 
csv_filename = 'Angles_2D.csv'
output_dir = os.path.join(file_path, "..", csv_filename)

with open(output_dir, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['left_angles', 'right_angles'])  # 寫入標題行

    for row in zip(angles_l, angles_r):
      csv_writer.writerow(row)
#####################################################

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 左腳
axes[0].plot(angles_l, label='2D_Angles', color='orange')
axes[0].set_xlabel('Time Step')
axes[0].set_ylabel('Angle (degrees)')
axes[0].set_title('Left Angle Variation Over Times')
axes[0].legend()


# 右腳
axes[1].plot(angles_r, label='2D_Angles')
axes[1].set_xlabel('Time Step')
axes[1].set_ylabel('Angle (degrees)')
axes[1].set_title('Right Angle Variation Over Times')
axes[1].legend()

# 調整子圖之間的間距
plt.tight_layout()
# 設定 y 軸範圍
plt.ylim(60, 180) 

# 顯示圖表
plt.show()