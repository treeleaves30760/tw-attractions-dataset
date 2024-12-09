import os
import json
import shutil
from tqdm import tqdm
import logging

# 設定logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def process_json_files():
    # 設定目錄路徑
    source_dataset_dir = "/media/Pluto/stanley_hsu/TW_attraction/datasets"
    filtered_images_dir = "/media/Pluto/stanley_hsu/TW_attraction/Small_Filter_Images"
    target_dataset_dir = "/media/Pluto/stanley_hsu/TW_attraction/Small_Filter_Image_Dataset"
    
    # 確保目標目錄存在
    os.makedirs(target_dataset_dir, exist_ok=True)
    
    # 獲取所有過濾後的圖片路徑
    filtered_images = set()
    for root, _, files in os.walk(filtered_images_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # 獲取相對路徑
                rel_path = os.path.relpath(os.path.join(root, file), filtered_images_dir)
                filtered_images.add(rel_path)
    
    logging.info(f"找到 {len(filtered_images)} 張過濾後的圖片")
    
    # 計數器
    total_json = 0
    copied_json = 0
    
    # 遍歷所有json檔案
    for root, _, files in os.walk(source_dataset_dir):
        for file in files:
            if file.endswith('.json'):
                total_json += 1
                json_path = os.path.join(root, file)
                
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 檢查image_path是否存在於過濾後的圖片中
                    landmark_name = data.get('landmark_name', '')
                    image_path = data.get('image_path', '')
                    
                    if landmark_name and image_path:
                        full_image_path = os.path.join(landmark_name, image_path)
                        
                        if full_image_path in filtered_images:
                            # 創建目標目錄（保持相同的目錄結構）
                            rel_path = os.path.relpath(json_path, source_dataset_dir)
                            target_path = os.path.join(target_dataset_dir, rel_path)
                            os.makedirs(os.path.dirname(target_path), exist_ok=True)
                            
                            # 複製json檔案
                            shutil.copy2(json_path, target_path)
                            copied_json += 1
                            
                            if copied_json % 100 == 0:
                                logging.info(f"已處理 {copied_json} 個符合條件的JSON檔案")
                    
                except Exception as e:
                    logging.error(f"處理JSON檔案時發生錯誤 {json_path}: {str(e)}")
    
    logging.info(f"處理完成！總共掃描了 {total_json} 個JSON檔案")
    logging.info(f"複製了 {copied_json} 個符合條件的JSON檔案")
    logging.info(f"篩選率: {(copied_json/total_json)*100:.2f}%")

if __name__ == "__main__":
    process_json_files()