from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import base64
import re
import requests


def preprocess_image(image_path):
    # 使用 Hugging Face 的圖像分類模型進行預處理
    API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_TOKEN')}"}

    with open(image_path, "rb") as f:
        data = f.read()

    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()


def Llama_Filter(image_path, question, model="meta-llama/Llama-3.2-11B-Vision-Instruct", max_tokens=300):
    load_dotenv()
    HF_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
    client = InferenceClient(api_key=HF_TOKEN)

    # 預處理圖片
    preprocessed_result = preprocess_image(image_path)
    top_labels = ", ".join(
        [f"{item['label']} ({item['score']:.2f})" for item in preprocessed_result[:3]])

    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    messages = [
        {
            "role": "system",
            "content": f"""你是一位專業的旅遊景點分析師，擅長評估圖片中景點的可辨識程度。在判斷時，請考慮以下特徵：

1. 獨特的建築物或地標
2. 引人注目的自然景觀
3. 文化元素（如寺廟、博物館）
4. 遊憩設施（如主題公園、觀景台）
5. 清晰的資訊標示（如景點名稱牌）
6. 整體環境的規劃和維護程度

預處理模型識別出的前三個標籤是：{top_labels}

請根據這些特徵和預處理結果，評估圖片的可辨識程度。然後，給出一個0到1之間的數字，並解釋你的評分理由：

0-0.2: 完全無法辨識為景點
0.2-0.4: 有少量特徵，但不足以確定是景點
0.4-0.6: 有一些特徵，可能是景點
0.6-0.8: 有較多特徵，很可能是景點
0.8-1: 非常確定是一個可辨識的景點

格式如下：
評分：[你的評分]
理由：[你的解釋]"""
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"}},
                {"type": "text", "text": question},
            ],
        }
    ]

    response = ""
    for message in client.chat_completion(model=model, messages=messages, max_tokens=max_tokens, stream=True):
        chunk = message.choices[0].delta.content
        if chunk:
            response += chunk

    # 提取評分和理由
    score_match = re.search(r'評分：(\d+(\.\d+)?)', response)
    reason_match = re.search(r'理由：(.+)', response, re.DOTALL)

    if score_match and reason_match:
        score = float(score_match.group(1))
        reason = reason_match.group(1).strip()
        return max(0, min(score, 1)), reason
    else:
        return None, None


def write_to_markdown(image_path, score, reason, output_file):
    with open(output_file, 'a', encoding='utf-8') as f:
        image_path = image_path.replace('\\', '/')
        f.write(f"![image](/{image_path})\n")
        f.write(f"辨識分數：{score:.2f}\n")
        f.write(f"理由：{reason}\n\n")


if __name__ == "__main__":
    output_file = 'identify_score.md'
    open(output_file, 'w').close()

    for i in range(30):
        image_path = f"input_image/921地震教育園區/921地震教育園區-{i}.jpg"
        question = "請評估這張圖片是否為一個可辨識的景點，給出評分和理由。"
        score, reason = Llama_Filter(image_path, question)
        print(f"圖片 {i} - 分數: {score}, 理由: {reason}")
        write_to_markdown(
            image_path, score if score is not None else 0, reason or "無法解析回應", output_file)

    for i in range(30):
        image_path = f'input_image/國立自然科學博物館/國立自然科學博物館-{i * 5}.jpg'
        question = "請評估這張圖片是否為一個可辨識的景點，給出評分和理由。"
        score, reason = Llama_Filter(image_path, question)
        print(f"圖片 {i} - 分數: {score}, 理由: {reason}")
        write_to_markdown(
            image_path, score if score is not None else 0, reason or "無法解析回應", output_file)

    for i in range(30):
        image_path = f'input_image/彩虹眷村/彩虹眷村-{i * 5}.jpg'
        question = "請評估這張圖片是否為一個可辨識的景點，給出評分和理由。"
        score, reason = Llama_Filter(image_path, question)
        print(f"圖片 {i} - 分數: {score}, 理由: {reason}")
        write_to_markdown(
            image_path, score if score is not None else 0, reason or "無法解析回應", output_file)

    print("處理完成")
