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


def get_next_file_number(folder_path, prefix):
    existing_files = glob.glob(os.path.join(folder_path, f'{prefix}-*.json'))
    if not existing_files:
        return 1
    max_number = max(int(os.path.splitext(os.path.basename(f))[
                     0].split('-')[1]) for f in existing_files)
    return max_number + 1


def process_images():
    for root, dirs, files in os.walk(INPUT_FOLDER):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, INPUT_FOLDER)
                output_folder = os.path.join(OUTPUT_FOLDER, relative_path)
                os.makedirs(output_folder, exist_ok=True)

                next_number_a = get_next_file_number(output_folder, 'A')
                next_number_b = get_next_file_number(output_folder, 'B')

                # 處理第一個問題「這是台灣哪一個景點？」
                result_taiwan = query_gpt4(image_path, "這是台灣哪一個景點？")
                output_file_a = os.path.join(
                    output_folder, f'A-{next_number_a:06d}.json')
                data_a = {
                    "image_path": image_path,
                    "question": "這是台灣哪一個景點？",
                    "answer": result_taiwan
                }
                with open(output_file_a, 'w', encoding='utf-8') as f:
                    json.dump(data_a, f, ensure_ascii=False, indent=4)
                print(f"Processed {image_path}, saved to {output_file_a}")

                time.sleep(1)  # 避免超過API速率限制

                # 處理第二個問題「這是哪一個景點？」
                result_general = query_gpt4(image_path, "這是哪一個景點？")
                output_file_b = os.path.join(
                    output_folder, f'B-{next_number_b:06d}.json')
                data_b = {
                    "image_path": image_path,
                    "question": "這是哪一個景點？",
                    "answer": result_general
                }
                with open(output_file_b, 'w', encoding='utf-8') as f:
                    json.dump(data_b, f, ensure_ascii=False, indent=4)
                print(f"Processed {image_path}, saved to {output_file_b}")


if __name__ == "__main__":
    process_images()
