import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TWAttractionDataset(Dataset):
    """
    Custom dataset for Taiwan Attractions with conversation and image data
    """
    def __init__(
        self,
        root_dir: str,
        tw_list_path: str,
        image_size: Tuple[int, int] = (224, 224),
        max_tokens: int = 512
    ):
        """
        Initialize the dataset
        
        Args:
            root_dir (str): Root directory containing attraction folders
            tw_list_path (str): Path to TW_List.json
            image_size (tuple): Target image size for resizing
            max_tokens (int): Maximum number of tokens for text
        """
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise ValueError(f"Root directory does not exist: {root_dir}")
            
        self.image_size = image_size
        self.max_tokens = max_tokens
        
        # Load TW_List.json
        if not Path(tw_list_path).exists():
            raise ValueError(f"TW_List.json does not exist: {tw_list_path}")
            
        try:
            with open(tw_list_path, 'r', encoding='utf-8') as f:
                tw_list_data = json.load(f)
                if 'TW_Attractions' not in tw_list_data:
                    raise ValueError("TW_List.json does not contain 'TW_Attractions' key")
                self.tw_list = tw_list_data['TW_Attractions']
                logger.info(f"Loaded {len(self.tw_list)} attractions from TW_List.json")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in {tw_list_path}")
        
        # Initialize data samples
        self.samples = self._load_all_samples()
        if not self.samples:
            raise ValueError("No valid samples found in the dataset")
        logger.info(f"Loaded {len(self.samples)} total samples")
        
        # Setup image transforms
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def _load_all_samples(self) -> List[Dict]:
        """
        Load all samples from the dataset
        """
        samples = []
        
        for attraction in self.tw_list:
                
            attraction_dir = self.root_dir / attraction
            if not attraction_dir.exists():
                logger.warning(f"Directory not found for attraction: {attraction}")
                continue
                
            # Load all JSON files for this attraction
            json_files = list(attraction_dir.glob('*.json'))
            logger.info(f"Found {len(json_files)} JSON files for {attraction}")
            
            for json_path in json_files:
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading JSON file {json_path}: {str(e)}")
                    continue
                    
                # Get base image folder from json
                base_folder = data.get('base_folder', '')
                image_path = data.get('image_path', '')
                if not image_path:
                    logger.warning(f"No image path in {json_path}")
                    continue
                    
                # Construct full image path
                try:
                    full_image_path = Path(base_folder) / attraction / image_path
                    if not full_image_path.exists():
                        logger.warning(f"Image not found: {full_image_path}")
                except Exception as e:
                    logger.error(f"Error processing image path: {str(e)}")
                    continue
                
                # Process conversations
                conversations = []
                
                # Process multi-turn conversations
                if 'conversations' in data and 'multi_turn' in data['conversations']:
                    if data['conversations']['multi_turn'] != None:
                        for qa_pair in data['conversations']['multi_turn'].get('qa_pairs', []):
                            conv = qa_pair.get('conversation', [])
                            if conv:
                                conversations.append(conv)
                
                # Process detailed info
                if 'conversations' in data and 'detailed_info' in data['conversations']:
                    if data['conversations']['detailed_info'] != None:
                        for qa_pair in data['conversations']['detailed_info'].get('qa_pairs', []):
                            qa_conv = [
                                {'role': 'user', 'content': qa_pair.get('question', '')},
                                {'role': 'assistant', 'content': qa_pair.get('answer', '')}
                            ]
                            conversations.append(qa_conv)
                
                if not conversations:
                    logger.warning(f"No conversations found in {json_path}")
                    continue
                
                # Add sample
                samples.append({
                    'image_path': str(full_image_path),
                    'landmark_name': data.get('landmark_name', ''),
                    'description': data.get('description', ''),
                    'conversations': conversations
                })
                logger.debug(f"Added sample for {data.get('landmark_name', '')} from {json_path}")
        
        logger.info(f"Total samples loaded: {len(samples)}")
        return samples

    def _prepare_conversation(self, conversations: List[List[Dict]]) -> str:
        """
        Format conversations for the model
        """
        formatted_conversations = []
        for conv in conversations:
            formatted_conv = []
            for turn in conv:
                role = turn.get('role', '')
                content = turn.get('content', '')
                if role and content:
                    formatted_conv.append(f"{role}: {content}")
            if formatted_conv:
                formatted_conversations.append("\n".join(formatted_conv))
        
        return "\n\n".join(formatted_conversations)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample from the dataset
        """
        sample = self.samples[idx]
        
        try:
            # Load and transform image
            image = Image.open(sample['image_path']).convert('RGB')
            image_tensor = self.transform(image)
            
            # Format conversations
            conversation_text = self._prepare_conversation(sample['conversations'])
            
            return {
                'image': image_tensor,
                'text': conversation_text,
                'landmark_name': sample['landmark_name'],
                'description': sample['description']
            }
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {str(e)}")
            raise

def create_dataloader(
    root_dir: str,
    tw_list_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (224, 224),
    max_tokens: int = 512,
    shuffle: bool = True
) -> DataLoader:
    """
    Create a DataLoader for the TWAttractionDataset
    """
    logger.info(f"Creating dataloader with root_dir: {root_dir}, tw_list_path: {tw_list_path}")
    
    dataset = TWAttractionDataset(
        root_dir=root_dir,
        tw_list_path=tw_list_path,
        image_size=image_size,
        max_tokens=max_tokens
    )
    
    logger.info(f"Dataset created with {len(dataset)} samples")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

"""
Example usage:
if __name__ == "__main__":
    # Configuration
    ROOT_DIR = "path/to/dataset"
    TW_LIST_PATH = "path/to/TW_List.json"
    
    # Create dataloader
    dataloader = create_dataloader(
        root_dir=ROOT_DIR,
        tw_list_path=TW_LIST_PATH,
        batch_size=32
    )
    
    # Test dataloader
    for batch in dataloader:
        print("Image shape:", batch['image'].shape)
        print("Sample conversation:", batch['text'][0][:100])
        print("Landmark name:", batch['landmark_name'][0])
        break
"""