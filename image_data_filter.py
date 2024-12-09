import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from clip import clip
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import logging

# 設定logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        # 遍歷所有子目錄收集圖片路徑
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                for img_name in os.listdir(subdir_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(subdir_path, img_name))
        
        logging.info(f"找到 {len(self.image_paths)} 張圖片")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, img_path
        except Exception as e:
            logging.error(f"讀取圖片失敗 {img_path}: {str(e)}")
            return None

def main():
    # 設定來源和目標目錄
    source_dir = "/media/Pluto/stanley_hsu/TW_attraction/images/TW_Attractions"
    target_dir = "/media/Pluto/stanley_hsu/TW_attraction/Small_Filter_Images"
    
    # 確保目標目錄存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 載入CLIP模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"使用設備: {device}")
    
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # 準備文本提示
    text_inputs = torch.cat([
        clip.tokenize("a photo of a face"),
        clip.tokenize("a photo containing human face"),
        clip.tokenize("a portrait photo"),
        clip.tokenize("a selfie"),
        clip.tokenize("a person"),
        clip.tokenize("people in the photo")
    ]).to(device)
    
    # 創建數據集和數據載入器
    dataset = ImageDataset(
        source_dir,
        transform=preprocess
    )
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4)
    
    threshold = 0.4
    
    copied_count = 0
    total_processed = 0
    
    logging.info("開始處理圖片...")
    
    with torch.no_grad():
        for images, image_paths in tqdm(dataloader):
            # 將圖片移到GPU（如果可用）
            images = images.to(device)
            
            # 獲取圖片特徵
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # 獲取文本特徵
            text_features = model.encode_text(text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 計算相似度
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            max_similarity = similarity.max(dim=-1)[0]
            
            # 處理每張圖片
            for idx, (sim, img_path) in enumerate(zip(max_similarity, image_paths)):
                total_processed += 1
                
                # 記錄相似度分數
                logging.debug(f"圖片 {img_path} 的人臉相似度: {sim.item():.4f}")
                
                if sim.item() < threshold:  # 如果相似度低於閾值，表示沒有明顯人臉
                    # 保持原始目錄結構
                    relative_path = os.path.relpath(img_path, source_dir)
                    target_path = os.path.join(target_dir, relative_path)
                    
                    # 確保目標目錄存在
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    
                    try:
                        # 複製圖片
                        shutil.copy2(img_path, target_path)
                        copied_count += 1
                    except Exception as e:
                        logging.error(f"複製圖片失敗 {img_path}: {str(e)}")

    logging.info(f"處理完成！總共處理 {total_processed} 張圖片，複製了 {copied_count} 張圖片")
    logging.info(f"篩選率: {(copied_count/total_processed)*100:.2f}%")

if __name__ == "__main__":
    main()