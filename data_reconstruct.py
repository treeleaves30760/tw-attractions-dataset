import os
import glob
from pathlib import Path

def rename_files_in_directory(base_dir):
    # 確保 base_dir 存在
    if not os.path.exists(base_dir):
        print(f"目錄 {base_dir} 不存在")
        return
    
    # 遍歷所有子目錄
    for spot_dir in os.listdir(base_dir):
        spot_path = os.path.join(base_dir, spot_dir)
        
        # 確保這是一個目錄
        if not os.path.isdir(spot_path):
            continue
            
        # 獲取該目錄下所有的 JSON 檔案
        json_files = glob.glob(os.path.join(spot_path, "*.json"))
        json_files.sort()  # 排序檔案列表
        
        # 為每個檔案重新命名
        for index, old_file in enumerate(json_files, start=1):
            # 建立新的檔案名稱
            new_name = f"{spot_dir}-{str(index).zfill(3)}.json"
            new_path = os.path.join(spot_path, new_name)
            
            try:
                # 如果目標檔案已存在，先刪除
                if os.path.exists(new_path):
                    os.remove(new_path)
                    
                # 重新命名檔案
                os.rename(old_file, new_path)
                print(f"已重新命名: {os.path.basename(old_file)} -> {new_name}")
            except Exception as e:
                print(f"重新命名 {old_file} 時發生錯誤: {str(e)}")

# 使用方式
base_directory = "datasets"  # 請根據實際路徑修改
rename_files_in_directory(base_directory)