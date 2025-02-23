import os
import shutil

def move_files(source_base, target_base):
    """
    將源資料夾中的elec.txt檔案及其對應資料夾移動到目標資料夾
    
    Parameters:
    - source_base: 源資料夾路徑
    - target_base: 目標資料夾路徑
    """
    # 確保目標基礎資料夾存在
    if not os.path.exists(target_base):
        os.makedirs(target_base)
    
    # 遍歷源資料夾中的所有檔案和資料夾
    for item in os.listdir(source_base):
        source_path = os.path.join(source_base, item)

        for fname in os.listdir(source_path):
            fname_path = os.path.join(source_path, fname)

            fname_path = os.path.join(fname_path, 'output_3D')
            print(fname_path)

            # 檢查是否為資料夾
            if os.path.isdir(fname_path):
                elec_file = os.path.join(fname_path, 'Angles_3D_256fps.csv')
                
                # 如果資料夾中有指定檔案
                if os.path.exists(elec_file):
                    # 在目標路徑創建對應的資料夾
                    target_dir = os.path.join(target_base, item)
                    target_dir = os.path.join(target_dir, fname)
                    if not os.path.exists(target_dir):
                        os.makedirs(target_dir)
                    
                    # 複製elec.txt到新位置
                    target_file = os.path.join(target_dir, 'Angles_3D_256fps.csv')
                    shutil.copy2(elec_file, target_file)
                    print(f"已移動: {item}/Angles_3D_256fps.csv")

if __name__ == "__main__":
    source_base = r"E:/output/021"
    target_base = r"D:/para/021"
    
    move_files(source_base, target_base)
    print("檔案移動完成!")