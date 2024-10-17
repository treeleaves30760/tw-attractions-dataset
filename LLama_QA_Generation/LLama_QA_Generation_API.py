import os
import json
import requests
import wikipedia
import regex
import base64
from PIL import Image
import io
import time

def get_wiki_knowledge(topic):
    """從維基百科獲取相關知識（繁體中文）"""
    wikipedia.set_lang("zh-tw")  # 設定為繁體中文
    try:
        page = wikipedia.page(topic)
        return page.content
    except:
        return "無法獲取維基百科內容"

def generate_llama_data(image_path, gpt4_description, wiki_content, landmark_name):
    multi_result = generate_llama_data_multi_turn(image_path, gpt4_description, wiki_content, landmark_name)
    single_result = generate_llama_data_single_turn(image_path, gpt4_description, wiki_content, landmark_name)
    result = {"multi_turn": multi_result, "detailed_explanation": single_result["detailed_explanation"], "complex_reasoning": single_result["complex_reasoning"]}
    return result

def generate_llama_data_single_turn(image_path, gpt4_description, wiki_content, landmark_name):
    """使用 API 生成訓練資料（繁體中文）"""
    data_types = {
        "detailed_explanation": "請提供這張圖片中元素的詳細解釋，以及相關的歷史背景。",
        "detailed_explanation": "請詳細解釋這張圖片中的元素，以及這個景點與之相匹配的特色。",
        "complex_reasoning": "請分析這個地標的重要性，以及它對文化和社會的影響。",
        "complex_reasoning": "這個景點在當地的文化和歷史中扮演了什麼樣的角色？"
    }

    results = {}
    for data_type, task_prompt in data_types.items():
        # 構建請求內容
        prompt = f"""你是一個知識淵博的 AI 助理。請使用提供的圖片、描述和維基百科內容來回答。請使用繁體中文回答。

圖片描述：
{gpt4_description}

維基百科內容：
{wiki_content}

任務：
{task_prompt}

請根據上述資訊，生成以下格式的 JSON，請使用繁體中文回答：

{{
    "image_path": "{image_path}",
    "qa_pairs": [
        {{
            "question_type": "{data_type}",
            "question": "<問題>",
            "answer": "<回答>"
        }}
        // 要有多組問答，請使用繁體中文回答，請用逗號分隔
    ]
}}

請確保輸出僅包含上述格式的 JSON，不要添加任何額外的說明或文字。"""

        # 讀取圖片並轉換為 base64
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        # 準備請求數據
        data = {
            "text": prompt,
            "image": image_base64,
            "max_new_tokens": 3000
        }

        # 發送 POST 請求到 /generate 端點
        response = requests.post('http://localhost:8000/generate', json=data)

        if response.status_code == 200:
            messages = response.json().get('messages', [])
            assistant_response = next((msg['content'] for msg in messages if msg['role'] == 'assistant'), '')
            
            # 嘗試解析回應為 JSON 格式
            try:
                # 清理回應文字，移除可能的非 JSON 內容
                json_str = extract_json(assistant_response)
                qa_data = json.loads(json_str)
                results[data_type] = qa_data
            except json.JSONDecodeError as e:
                print(f"解析 JSON 時出錯：{e}\n{assistant_response}")
                results[data_type] = None
        else:
            results[data_type] = None
            print(f"API 請求失敗，狀態碼：{response.status_code}")

    return results

def generate_llama_data_multi_turn(image_path, gpt4_description, wiki_content, landmark_name):
    """使用 API 生成訓練資料（繁體中文）"""
    data_types = {
        "multi_turn": "請針對這張圖片和相關知識生成一段多輪對話，對話內容包含詢問景點的名稱與相關資訊。",
    }

    results = {}
    for data_type, task_prompt in data_types.items():
        # 構建請求內容
        prompt = f"""你是一個知識淵博的 AI 助理。請使用提供的圖片、描述和維基百科內容來回答。請使用繁體中文回答。

圖片描述：
{gpt4_description}

維基百科內容：
{wiki_content}

任務：
{task_prompt}

請根據上述資訊，生成以下格式的 JSON，請使用繁體中文回答：

{{
    "image_path": "{image_path}",
    "qa_pairs": [
        {{
            "question_type": "{data_type}",
            "conversation": [
                {{
                    "role": "user",
                    "content": "請問這是什麼景點"
                }},
                {{
                    "role": "assistant",
                    "content": "這是{landmark_name}"
                }},
                // 請根據上述資料繼續生成多輪對話，請使用繁體中文回答，請用逗號分隔
            ]
        }},
        {{
            "question_type": "{data_type}",
            "conversation": [
                {{
                    "role": "user",
                    "content": "請問圖片中是什麼景點"
                }},
                {{
                    "role": "assistant",
                    "content": "這個景點是{landmark_name}"
                }},
                {{
                    "role": "user",
                    "content": "請問這個照片有什麼特色"
                }}
                // 請根據上述資料繼續生成多輪對話，請使用繁體中文回答，請用逗號分隔
            ]
        }}
        // 請繼續生成多組對話，請使用繁體中文回答，請用逗號分隔
    ]
}}

請確保輸出僅包含上述格式的 JSON，不要添加任何額外的說明或文字。"""

        # 讀取圖片並轉換為 base64
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        # 準備請求數據
        data = {
            "text": prompt,
            "image": image_base64,
            "max_new_tokens": 3000
        }

        # 發送 POST 請求到 /generate 端點
        response = requests.post('http://localhost:8000/generate', json=data)

        if response.status_code == 200:
            messages = response.json().get('messages', [])
            assistant_response = next((msg['content'] for msg in messages if msg['role'] == 'assistant'), '')
            
            # 嘗試解析回應為 JSON 格式
            try:
                # 清理回應文字，移除可能的非 JSON 內容
                json_str = extract_json(assistant_response)
                qa_data = json.loads(json_str)
                results[data_type] = qa_data
            except json.JSONDecodeError as e:
                print(f"解析 JSON 時出錯：{e}\n{assistant_response}")
                results[data_type] = None
        else:
            results[data_type] = None
            print(f"API 請求失敗，狀態碼：{response.status_code}")

    return results

def extract_json(text):
    """從文本中提取 JSON 字串"""
    # 使用 regex 模組的遞歸匹配功能
    json_match = regex.search(r'\{(?:[^{}]+|(?R))*\}', text, regex.DOTALL)
    if json_match:
        return json_match.group(0)
    else:
        # raise ValueError("未找到有效的 JSON")
        print("未找到有效的 JSON")
        return ""

def save_to_json(data, landmark_name, number):
    """將生成的資料保存為 JSON 文件"""
    os.makedirs(f"dataset/{landmark_name}-{number}", exist_ok=True)
    
    # 獲取當前時間戳  
    timestamp = time.time()
    struct_time = time.gmtime(timestamp)  
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", struct_time) 
    
    # 保存多輪對話資料
    if "multi_turn" in data and data["multi_turn"] is not None:
        filename = f"dataset/{landmark_name}-{number}/llama_generate_multi_turn_{formatted_time}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data["multi_turn"], f, ensure_ascii=False, indent=4)
        print(f"已保存多輪對話資料至 {filename}")
    
    # 保存單次對話資料
    single_turn_data = {
        "image_path": data.get("multi_turn", {}).get("image_path", ""),
        "qa_pairs": []
    }
    
    # 添加詳細解釋資料
    if "detailed_explanation" in data and data["detailed_explanation"] is not None:
        single_turn_data["qa_pairs"].extend(data["detailed_explanation"].get("qa_pairs", []))
    
    # 添加複雜推理資料
    if "complex_reasoning" in data and data["complex_reasoning"] is not None:
        single_turn_data["qa_pairs"].extend(data["complex_reasoning"].get("qa_pairs", []))
    
    # 保存單次對話資料
    if single_turn_data["qa_pairs"]:
        filename = f"dataset/{landmark_name}-{number}/llama_generate_single_turn_{formatted_time}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(single_turn_data, f, ensure_ascii=False, indent=4)
        print(f"已保存單次對話資料至 {filename}")

def main(image_path, landmark_name, gpt4_description, number):
    # 獲取維基百科知識
    wiki_content = get_wiki_knowledge(landmark_name)

    # 生成 Llama 資料
    llama_data = generate_llama_data(image_path, gpt4_description, wiki_content, landmark_name)

    # 保存資料
    save_to_json(llama_data, landmark_name, number)

    print(f"已完成 {landmark_name}-{number} 的資料生成。")

if __name__ == "__main__":
    image_folder = "/media/Pluto/andy/taiwan_chatgpt"
    image = "input_image/高雄85大樓/高雄85大樓-12.jpg"
    image_path = os.path.join(image_folder, image)
    landmark_name = "高雄85大樓"
    number = 12
    gpt4_description = """這張圖片展示的是**高雄85大樓**（85 Sky Tower），位於台灣高雄市苓雅區，是該市著名的地標建築。高雄85大樓是台灣第二高的摩天大樓，也是高雄市的主要觀光景點之一。

以下是圖片中的一些細節描述：

1. **大樓外觀**：
   - 大樓以深色玻璃幕牆為主體，帶有藍色和綠色的反光效果，給人現代且科技感十足的印象。
   - 大樓的中間部分有一個明顯的缺口設計（稱為懸空大堂），這是其最具標誌性的結構之一，讓人一眼就能辨認出這座大樓。

2. **建築結構**：
   - 大樓的設計以兩個對稱的塔樓構成，中間由一個水平橋梁連接，這使得大樓的視覺效果更加特別。
   - 塔樓的頂部逐漸變細，並帶有一些垂直線條，給人一種向上延伸的動感。

3. **周圍環境**：
   - 大樓前方有幾棵茂密的樹木，這些綠色的植物與大樓深色的玻璃幕牆形成了明顯的對比。
   - 天空中的白雲與藍天映襯出大樓的宏偉，整體畫面色彩明亮。

高雄85大樓建於1997年，曾是亞洲最高的摩天大樓之一。其樓層數達85層，因此得名。"""
    for i in range(10):
        main(image_path, landmark_name, gpt4_description, number)
