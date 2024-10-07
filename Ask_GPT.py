import os
import json
import glob
from PIL import Image
import base64
from openai import OpenAI
from dotenv import load_dotenv
import time

# 載入環境變數
load_dotenv()

# 配置
INPUT_FOLDER = 'input_image'
OUTPUT_FOLDER = 'dataset'
API_KEY = os.getenv('OPENAI_API_KEY')
QA_AMOUNT = 10

# 初始化 OpenAI 客戶端
client = OpenAI(api_key=API_KEY)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def query_gpt4(image_path, question):
    base64_image = encode_image(image_path)
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=300
    )
    return response.choices[0].message.content


def process_image(image_path, output_folder):
    # 創建圖片對應的輸出資料夾
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image_output_folder = os.path.join(output_folder, image_name)
    os.makedirs(image_output_folder, exist_ok=True)

    questions = [
        ("這是台灣哪一個景點？", "A"),
        ("這是哪一個景點？", "B")
    ]

    for question, prefix in questions:
        for i in range(QA_AMOUNT):  # 每個問題生成兩個 JSON 文件
            result = query_gpt4(image_path, question)
            output_file = os.path.join(
                image_output_folder, f'{prefix}-{i+1:06d}.json')
            data = {
                "image_path": image_path,
                "question": question,
                "answer": result
            }
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"Processed {image_path}, saved to {output_file}")


def process_images():
    for root, dirs, files in os.walk(INPUT_FOLDER):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, INPUT_FOLDER)
                output_folder = os.path.join(OUTPUT_FOLDER, relative_path)
                os.makedirs(output_folder, exist_ok=True)
                process_image(image_path, output_folder)


if __name__ == "__main__":
    process_images()
