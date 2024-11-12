import json
import os
from pathlib import Path
from typing import Dict, List

def process_multi_turn_conversations(conversations: List[Dict]) -> List[Dict]:
    """處理多輪對話，轉換成指定格式"""
    formatted_conversations = []
    
    for conv in conversations:
        formatted_conv = {}
        messages = conv["conversation"]
        
        for i, msg in enumerate(messages, 1):
            if msg["role"] == "user":
                formatted_conv[f"User_{int((i + 1) / 2)}"] = msg["content"]
            elif msg["role"] == "assistant":
                formatted_conv[f"Machine_{int(i / 2)}"] = msg["content"]
        
        if formatted_conv:  # 只有在有對話內容時才添加
            formatted_conversations.append(formatted_conv)
    
    return formatted_conversations

def process_single_turn_conversations(qa_pairs: List[Dict]) -> List[Dict]:
    """處理單輪對話，轉換成指定格式"""
    formatted_qa_pairs = []
    
    for qa in qa_pairs:
        formatted_qa = {
            "User_1": qa["question"],
            "Machine_1": qa["answer"]
        }
        formatted_qa_pairs.append(formatted_qa)
    
    return formatted_qa_pairs

def process_json_files(input_dir: str, output_dir: str, landmark: str):
    """處理資料夾中的所有JSON檔案並輸出JSONL"""
    input_path = Path(input_dir)
    
    multi_turn_output = os.path.join(output_dir, f"multi_turn_conversations_{landmark}.jsonl")
    single_turn_output = os.path.join(output_dir, f"single_turn_conversations_{landmark}.jsonl")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 用於收集所有對話
    all_multi_turn = []
    all_single_turn = []
    
    # 處理每個JSON檔案
    for json_file in input_path.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 處理多輪對話
            if "conversations" in data and "multi_turn" in data["conversations"]:
                multi_turn_convs = process_multi_turn_conversations(
                    data["conversations"]["multi_turn"]["qa_pairs"]
                )
                all_multi_turn.extend(multi_turn_convs)
            
            # 處理單輪對話
            if "conversations" in data and "detailed_info" in data["conversations"]:
                single_turn_convs = process_single_turn_conversations(
                    data["conversations"]["detailed_info"]["qa_pairs"]
                )
                all_single_turn.extend(single_turn_convs)
                
        except Exception as e:
            print(f"處理檔案 {json_file} 時發生錯誤: {str(e)}")
    
    # 寫入多輪對話JSONL
    with open(multi_turn_output, 'w', encoding='utf-8') as f:
        for conv in all_multi_turn:
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')
    
    # 寫入單輪對話JSONL
    with open(single_turn_output, 'w', encoding='utf-8') as f:
        for conv in all_single_turn:
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')

def concat_single_jsonl(input_dir: str, output_dir: str, output_file: str):
    """串接資料夾中所有single_turn開頭的JSONL檔案"""
    input_path = Path(input_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    all_conversations = []
    
    for jsonl_file in input_path.glob("single_turn_conversations_*.jsonl"):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                all_conversations.append(json.loads(line))
                
    # 寫入串接後的JSONL檔案
    with open(os.path.join(output_dir, output_file), 'w', encoding='utf-8') as f:
        for conv in all_conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')

def concat_multi_jsonl(input_dir: str, output_dir: str, output_file: str):
    """串接資料夾中所有multi_turn開頭的JSONL檔案"""
    input_path = Path(input_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    all_conversations = []
    
    for jsonl_file in input_path.glob("multi_turn_conversations_*.jsonl"):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                all_conversations.append(json.loads(line))
    
    # 寫入串接後的JSONL檔案
    with open(os.path.join(output_dir, output_file), 'w', encoding='utf-8') as f:
        for conv in all_conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')

def concat_jsonl(input_dir: str, output_dir: str, output_file: str):
    """串接資料夾中所有JSONL檔案"""
    input_path = Path(input_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    all_conversations = []
    
    for jsonl_file in input_path.glob("*.jsonl"):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                all_conversations.append(json.loads(line))
    
    # 寫入串接後的JSONL檔案
    with open(os.path.join(output_dir, output_file), 'w', encoding='utf-8') as f:
        for conv in all_conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    base_directory = os.path.join('dataset')
    output_directory = os.path.join('dataset_dialogue', 'single_landmark')
    for landmark in os.listdir(base_directory):
        input_directory = os.path.join(base_directory, landmark)
        process_json_files(input_directory, output_directory, landmark)
    
    concat_single_jsonl(os.path.join('dataset_dialogue', 'single_landmark'), 'dataset_dialogue', 'TW_Attraction_single_dataset.jsonl')
    concat_multi_jsonl(os.path.join('dataset_dialogue', 'single_landmark'), 'dataset_dialogue', 'TW_Attraction_multi_dataset.jsonl')
    concat_jsonl(os.path.join('dataset_dialogue', 'single_landmark'), 'dataset_dialogue', 'TW_Attraction_dataset.jsonl')