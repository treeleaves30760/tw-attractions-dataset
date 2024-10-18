import os
import json
import requests
import wikipedia
import time
from openai import OpenAI
from dotenv import load_dotenv
import base64
from LLama_QA_Generation_API import main, get_wiki_knowledge

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

BASE_FOLDER = "/media/Pluto/stanley_hsu/TW_attraction/input_image/"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def query_gpt4(image_path, landmark_name):
    base64_image = encode_image(image_path)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"這張圖片可能是台灣的{landmark_name}。請詳細描述圖片中的細節，並確認這是否確實為{landmark_name}。如果不是，請指出實際的景點名稱。"},
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
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error querying GPT-4: {e}")
        return None

def process_images():
    for landmark_name in os.listdir(BASE_FOLDER):
        landmark_path = os.path.join(BASE_FOLDER, landmark_name)
        if not os.path.isdir(landmark_path):
            continue

        print(f"Processing landmark: {landmark_name}")
        # Get Wikipedia content
        wiki_content = get_wiki_knowledge(landmark_name)

        for file in os.listdir(landmark_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_path = os.path.join(landmark_path, file)
                print(f"Processing image: {image_path}")
                
                # Query GPT-4 for description
                gpt4_description = query_gpt4(image_path, landmark_name)
                print(f"GPT-4 description: {gpt4_description}")
                if not gpt4_description:
                    continue

                # Extract number from filename
                try:
                    number = int(file.split('-')[-1].split('.')[0])
                except ValueError:
                    print(f"Warning: Could not extract number from filename: {file}")
                    continue

                # Call main function
                main(image_path, landmark_name, gpt4_description, number, wiki_content)

                # Optional: Add a delay to avoid rate limiting
                time.sleep(1)

if __name__ == "__main__":
    start_time = time.time()
    process_images()
    end_time = time.time()
    print(f'Total elapsed time: {end_time - start_time} seconds')