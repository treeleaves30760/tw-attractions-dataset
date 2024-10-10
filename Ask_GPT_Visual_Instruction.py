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

# 載入環境變數
load_dotenv()

# 配置
INPUT_FOLDER = 'input_image'
OUTPUT_FOLDER = 'dataset'
MODEL_NAME = 'gpt-4o'
API_KEY = os.getenv('OPENAI_API_KEY')
MAX_RETRIES = 3
ITERATIONS_PER_TYPE = 3  # 每種類型問題重複的次數

# 初始化 OpenAI 客戶端
client = OpenAI(api_key=API_KEY)

# 初始化 tiktoken 編碼器
encoding = tiktoken.encoding_for_model(MODEL_NAME)

# 定義問題集
QUESTIONS = [
    {
        "question_type": "conversation",
        "questions": [
            "請問圖片中是台灣哪一個景點？請簡短介紹一下。",
            "請問圖片中的景點有什麼特色？可以進行哪些活動？",
            "請問圖片中的景點在不同季節有什麼變化？哪個季節最適合參觀？"
        ]
    },
    {
        "question_type": "detailed_description",
        "questions": [
            "請詳細描述圖片中的台灣景點的視覺特徵、歷史背景和文化意義。",
            "圖片中的這個景點有哪些著名的地標或建築？它們有什麼特殊之處？"
        ]
    },
    {
        "question_type": "complex_reasoning",
        "questions": [
            "比較圖片中的這個景點與台灣其他類似景點的異同。有什麼獨特之處？",
            "分析圖片中的這個景點對當地經濟和文化的影響。它如何促進了當地發展？",
        ]
    }
]


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def count_tokens(text):
    return len(encoding.encode(text))


def query_gpt4(image_path, prompt):
    base64_image = encode_image(image_path)
    for _ in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text",
                                "text": f"{prompt}"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            answer = response.choices[0].message.content

            # 計算tokens
            input_tokens = count_tokens(prompt)
            output_tokens = count_tokens(answer)

            return answer, input_tokens, output_tokens
        except Exception as e:
            print(f"Error occurred: {e}. Retrying...")
            time.sleep(5)
    raise Exception("Max retries reached. Failed to get response from GPT-4.")


def process_image(image_path, output_folder):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image_output_folder = os.path.join(output_folder, image_name)
    os.makedirs(image_output_folder, exist_ok=True)

    total_input_tokens = 0
    total_output_tokens = 0

    for question_set in QUESTIONS:
        question_type = question_set["question_type"]
        for iteration in range(ITERATIONS_PER_TYPE):
            data = {
                "image_path": image_path,
                "qa_pairs": []
            }

            for question in question_set["questions"]:
                random_seed = uuid.uuid4().hex
                modified_question = f"{question}\n\nRandom seed: {random_seed}"
                answer, input_tokens, output_tokens = query_gpt4(
                    image_path, modified_question)
                data["qa_pairs"].append({
                    "question": question,
                    "answer": answer
                })
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens

            output_file = os.path.join(
                image_output_folder, f'{image_name}_{question_type}_{iteration+1}.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(
                f"Processed {image_path}, {question_type} iteration {iteration+1}, saved to {output_file}")

    # 保存token使用統計
    usage_stats = {
        "image_path": image_path,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens
    }
    usage_file = os.path.join(
        image_output_folder, f"{image_name}_token_usage_stats.json")
    with open(usage_file, 'w', encoding='utf-8') as f:
        json.dump(usage_stats, f, ensure_ascii=False, indent=4)
    print(f"Token usage statistics for {image_path} saved to {usage_file}")


def process_images():
    global total_input_tokens, total_output_tokens
    for root, dirs, files in os.walk(INPUT_FOLDER):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, INPUT_FOLDER)
                output_folder = os.path.join(OUTPUT_FOLDER, relative_path)
                os.makedirs(output_folder, exist_ok=True)
                process_image(image_path, output_folder)


if __name__ == "__main__":
    # 指定要處理的單張圖片
    single_image_path = "input_image/國立故宮博物院/國立故宮博物院-0.jpg"  # 請替換為實際的圖片路徑

    if os.path.exists(single_image_path):
        process_image(single_image_path, OUTPUT_FOLDER)
    else:
        print(f"Image not found: {single_image_path}")
    # process_images()
