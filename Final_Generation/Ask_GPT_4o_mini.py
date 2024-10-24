import os
import json
import glob
from PIL import Image
import base64
from openai import OpenAI
from dotenv import load_dotenv
import time
import random
import uuid
import tiktoken
import requests
import wikipedia
from typing import Dict, List, Any, Tuple
import logging
import regex

class TaiwanLandmarkDatasetGenerator:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Configuration
        self.input_folder = os.path.abspath('/media/Pluto/stanley_hsu/TW_attraction/images/TW_Attractions')
        self.output_folder = 'dataset'
        self.model_name = 'gpt-4o-mini'
        self.better_model_name = 'gpt-4o'
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.max_retries = 3
        self.confidence_threshold = 0.7
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        # Initialize tokenizer
        self.tokenizer_mini = tiktoken.encoding_for_model(self.model_name)
        self.tokenizer_better = tiktoken.encoding_for_model(self.better_model_name)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('dataset_generation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)

    def count_tokens(self, text: str, model: str) -> int:
        """Count tokens for a given text using the appropriate tokenizer."""
        tokenizer = self.tokenizer_mini if model == self.model_name else self.tokenizer_better
        return len(tokenizer.encode(text))

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def get_wiki_content(self, landmark_name: str) -> str:
        """Fetch content from Wikipedia in Traditional Chinese."""
        wikipedia.set_lang("zh-tw")
        try:
            page = wikipedia.page(landmark_name)
            return page.content
        except Exception as e:
            self.logger.error(f"Error fetching Wikipedia content for {landmark_name}: {e}")
            return ""

    def generate_initial_description(self, image_path: str, landmark_name: str) -> Tuple[str, Dict]:
        """生成初始描述並追蹤token使用量"""
        try:
            base64_image = self.encode_image(image_path)
            
            system_prompt = "你是一個專業的圖像描述與台灣景點專家。請使用繁體中文，詳細描述圖片中的景點，包含其特色、建築風格、周圍環境等細節。請使用結構化的方式描述。給的景點資訊可能會出錯，請以圖片為主，有錯誤請指出。"
            user_prompt = f"這張圖片可能是台灣的{landmark_name}。請詳細描述圖片中的細節，並確認這是否確實為{landmark_name}。如果不是，請指出實際的景點名稱。"
            
            # 記錄輸入token
            input_tokens = self.count_tokens(system_prompt + user_prompt, self.better_model_name)
            
            response = self.client.chat.completions.create(
                model=self.better_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ],
                max_tokens=1000
            )
            
            output_content = response.choices[0].message.content
            output_tokens = self.count_tokens(output_content, self.better_model_name)
            
            tokens = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
            
            return output_content, tokens
            
        except Exception as e:
            self.logger.error(f"Error generating initial description: {e}")
            return "", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        
    def extract_json(self, text):
        """從文本中提取 JSON 字串"""
        # 使用 regex 模組的遞歸匹配功能
        json_match = regex.search(r'\{(?:[^{}]+|(?R))*\}', text, regex.DOTALL)
        if json_match:
            return json_match.group(0)
        else:
            return ""
    
    def generate_conversations(self, image_path: str, description: str, wiki_content: str) -> Dict:
        """Generate various types of conversations using GPT-4o-mini."""
        prompt_templates = {
    "multi_turn": """你是一個專業的導覽AI助理，也是一個專業的提示詞生成者，請根據以下資訊，生成多組自然的多輪對話。

思考步驟：
1. 場景分析：
   - 分析圖片中景點的獨特視覺特徵
   - 確認景點的地理位置和周邊環境
   - 思考這個景點可能引發的不同類型問題

2. 對話規劃：
   - 設計不同背景遊客可能的問題（例如：歷史愛好者、建築迷、一般觀光客、攝影玩家等）
   - 規劃從簡單到深入的問題順序
   - 考慮不同時間點的變化（例如：白天/夜晚、四季變化）

3. 知識整合：
   - 結合圖片描述中的視覺元素
   - 融入維基百科的歷史和文化背景
   - 加入在地特色和趣聞軼事

圖片描述：
{description}

維基百科內容：
{wiki_content}

請生成三組不同情境的對話，每組至少5輪，每一輪對話的回答需要足夠豐富且詳細，符合以下JSON格式：
{{
    "qa_pairs": [
        {{
            "conversation": [
                {{"role": "user", "content": "<使用者問題>"}},
                {{"role": "assistant", "content": "<助理回答>"}},
                // 更多對話輪次...
            ]
        }},
        // 更多樣化的對話...
    ]
}}

要求：
1. 對話要自然流暢，避免生硬的問答
2. 每組對話要有不同的重點和深度
3. 回答要結合圖片特徵和歷史資訊
4. 回答要足夠詳細且豐富，不要過於簡短
5. 使用繁體中文，口語要自然親切
6. 只輸出符合要求的JSON格式""",

    "detailed_info": """你是一個專業的文史研究員，請根據提供的資訊，生成關於這個景點的深入分析對話。

思考步驟：
1. 景點辨識：
   - 分析圖片中的建築特徵和標誌性元素
   - 對照維基百科資訊確認景點身份
   - 列出能確定是該景點的關鍵證據

2. 資訊組織：
   - 整理景點的基本資料（位置、創建時間、規模等）
   - 歸納建築特色和藝術價值
   - 梳理歷史沿革和重要事件
   - 整理文化意義和社會影響

3. 內容轉換：
   - 將專業資訊轉換為易懂的對話形式
   - 設計循序漸進的問答結構
   - 加入適當的解釋和舉例

圖片描述：
{description}

維基百科內容：
{wiki_content}

請生成以下JSON格式的深入分析對話，請使用繁體中文，如果維基百科內容或圖片沒有對應的資料且無法推測，請生成「我很抱歉，我還不清楚這部分的資訊」：
{{
    "qa_pairs": [
        {{
            "question_type": "basic_info",
            "question": "這個景點的基本資訊是什麼？包含位置、建立時間等。",
            "answer": "<回答此問題的思考過程並給出包涵整理基本資訊回答>",
        }},
        {{
            "question_type": "architectural_features",
            "question": "從建築特色來看，這個景點有什麼獨特之處？",
            "answer": "<回答此問題的思考過程並給出包涵整理基本資訊回答>",
        }},
        {{
            "question_type": "historical_significance",
            "question": "這個景點在歷史上有什麼重要意義？",
            "answer": "<回答此問題的思考過程並給出歷史意義回答>",
        }},
        {{
            "question_type": "cultural_impact",
            "question": "這個景點對當地文化有什麼影響？",
            "answer": "<回答此問題的思考過程並給出文化影響回答>",
        }},
        {{
            "question_type": "current_status",
            "question": "目前這個景點的保存狀況和使用情況如何？",
            "answer": "<回答此問題的思考過程並給出現況描述回答>",
        }},
        {{
            "question_type": "visual_features",
            "question": "從圖片中可以觀察到哪些特別的元素？",
            "answer": "<回答此問題的思考過程並給出圖片特徵分析回答>",
        }},
        {{
            "question_type": "tourist_information",
            "question": "對想要參觀這個景點的遊客，有什麼特別的建議？",
            "answer": "<回答此問題的思考過程並給出旅遊建議回答>",
        }}
    ]
}}

要求：
1. 每個回答都要包含具體且詳實的資訊
2. 回答要結合圖片特徵和歷史資料
3. 使用親切易懂的語氣，但保持專業性
4. 每個回答都要有明確的邏輯推理過程
5. 使用繁體中文
6. 回答長度要適中，避免過於冗長或過於簡短
7. 確保回答中包含足夠的細節和例證
8. 只輸出符合要求的JSON格式"""
}

        results = {}
        token_usage = {}
        
        for conv_type, prompt in prompt_templates.items():
            try:
                system_message = "你是一個專業的導遊兼歷史學家，擅長介紹台灣的景點。"
                formatted_prompt = prompt.format(
                    description=description,
                    wiki_content=wiki_content
                )
                input_text = system_message + formatted_prompt
                
                # 記錄輸入token
                input_tokens = self.count_tokens(input_text, self.model_name)
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": system_message
                        },
                        {
                            "role": "user",
                            "content": formatted_prompt
                        }
                    ],
                    temperature=0.7
                )
                
                output_content = response.choices[0].message.content
                output_tokens = self.count_tokens(output_content, self.model_name)
                results[conv_type] = json.loads(self.extract_json(output_content))
                
                token_usage[conv_type] = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                }
                
            except Exception as e:
                self.logger.error(f"Error generating {conv_type} conversation: {e}\nThe output content is: {output_content}\n\nAfter JSON extraction: {self.extract_json(content)}")
                results[conv_type] = None

        return results, token_usage

    def evaluate_content(self, content: Dict) -> bool:
        """Evaluate generated content using GPT-4 mini."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一個內容品質評估專家。請評估生成內容的品質、準確性和自然度。"
                    },
                    {
                        "role": "user",
                        "content": f"請評估以下內容的品質：\n{json.dumps(content, ensure_ascii=False, indent=2)}"
                    }
                ]
            )
            
            # Extract confidence score from the content
            confidence_score = content.get('confidence_score', 0)
            return confidence_score >= self.confidence_threshold
            
        except Exception as e:
            self.logger.error(f"Error evaluating content: {e}")
            return False

    def save_dataset(self, landmark_name: str, data: Dict):
        """Save generated dataset to JSON file."""
        # 創建landmark特定的目錄路徑
        landmark_dir = os.path.join(self.output_folder, landmark_name)
        os.makedirs(landmark_dir, exist_ok=True)
        
        output_path = os.path.join(
            landmark_dir,
            f"{landmark_name}_{uuid.uuid4().hex[:8]}.json"
        )
        
        try:
            # 確保目錄存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Dataset saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving dataset: {e}")
            raise  # 重新拋出異常以便追蹤問題
    
    def process_landmark(self, image_path: str, landmark_name: str, landmark_info: str):
        """處理單個景點圖片，並追蹤所有token使用量"""
        
        # 生成初始描述
        description, description_tokens = self.generate_initial_description(image_path, landmark_name)
        if not description:
            return
        
        # 生成對話
        conversations, conversation_tokens = self.generate_conversations(image_path, description, landmark_info)
        
        # 計算總token使用量
        total_tokens = {
            "input_tokens": description_tokens["input_tokens"] + 
                sum(usage["input_tokens"] for usage in conversation_tokens.values()),
            "output_tokens": description_tokens["output_tokens"] + 
                sum(usage["output_tokens"] for usage in conversation_tokens.values()),
            "total_tokens": description_tokens["total_tokens"] + 
                sum(usage["total_tokens"] for usage in conversation_tokens.values())
        }
        
        # 準備輸出數據
        filtered_data = {
            'landmark_name': landmark_name,
            'image_path': image_path,
            'description': description,
            'conversations': conversations,
            'token_usage': {
                'description': {
                    'model': self.better_model_name,
                    'usage': description_tokens
                },
                'conversations': {
                    'model': self.model_name,
                    'usage_by_type': conversation_tokens
                },
                'total': total_tokens
            }
        }
        
        # 如果包含有效對話就保存數據集
        if filtered_data['conversations']:
            self.save_dataset(landmark_name, filtered_data)
            
            # 記錄token使用情況到日誌
            self.logger.info(f"""
Token usage for image {os.path.basename(image_path)}:
Description ({self.better_model_name}): {description_tokens['total_tokens']} tokens
Conversations ({self.model_name}): {sum(usage['total_tokens'] for usage in conversation_tokens.values())} tokens
Total: {total_tokens['total_tokens']} tokens
""")

    def generate_dataset(self):
        """Generate dataset for all landmarks in the input folder."""
        print(f'self.input_folder: {self.input_folder}')
        for landmark_name in os.listdir(self.input_folder):
            self.logger.info(f"Processing {landmark_name}")
            landmark_info = self.get_wiki_content(landmark_name)
            landmark_path = os.path.join(self.input_folder, landmark_name)
            for image in os.listdir(landmark_path):
                image_path = os.path.join(landmark_path, image)
                self.process_landmark(image_path, landmark_name, landmark_info)

def main():
    generator = TaiwanLandmarkDatasetGenerator()
    generator.generate_dataset()

if __name__ == "__main__":
    main()