import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Union

# 載入環境變數
load_dotenv()

MODEL_NAME = 'gpt-4o'

# 設置OpenAI API金鑰
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def calculate_confidence_scores(data: Dict) -> List[Dict]:
    """
    計算給定資料中每個問題的信心分數。

    :param data: 包含景點資訊和問答對的字典
    :return: 包含原始資料和每個問題信心分數的列表
    """
    results = []

    for qa_pair in data.get('qa_pairs', []):
        prompt = f"""
        請分析以下資訊並給出一個0到1之間的信心分數，表示這個回答準確描述了景點的機率：

        景點：{data.get('image_path', '').split('/')[-1]}
        問題：{qa_pair.get('question', '')}
        回答：{qa_pair.get('answer', '')}

        請僅返回一個0到1之間的數字作為信心分數，不需要其他解釋。
        """

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是一個專業的數據分析師，擅長評估資訊的準確性。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10
            )

            confidence_score = float(
                response.choices[0].message.content.strip())
        except Exception as e:
            print(f"計算信心分數時出錯: {e}")
            confidence_score = 0.0

        result = qa_pair.copy() if isinstance(qa_pair, dict) else {}
        result["confidence_score"] = confidence_score
        results.append(result)

    return results


def process_attraction_data(input_file: str):
    """
    處理景點資料，計算每個問題的信心分數，並保存結果到新檔案。

    :param input_file: 輸入JSON文件的路徑
    """
    file_name, file_extension = os.path.splitext(input_file)
    output_file = f"{file_name}_filtered{file_extension}"

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"錯誤：無法解析 JSON 文件 '{input_file}'。請確保文件格式正確。")
        return
    except FileNotFoundError:
        print(f"錯誤：找不到文件 '{input_file}'。請確保文件路徑正確。")
        return

    processed_data = []

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                processed_item = item.copy()
                processed_item['qa_pairs'] = calculate_confidence_scores(item)
                processed_data.append(processed_item)
            else:
                print(f"警告：跳過無效的數據項 {item}")
    elif isinstance(data, dict):
        processed_data = [data.copy()]
        processed_data[0]['qa_pairs'] = calculate_confidence_scores(data)
    else:
        print(f"錯誤：無效的數據格式。預期是列表或字典，但得到了 {type(data)}")
        return

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    print(f"處理完成。結果已保存到 {output_file}")
    print(f"總共處理了 {len(processed_data)} 個項目")


# 使用示例
if __name__ == "__main__":
    input_file = 'dataset/台北小巨蛋-37/台北小巨蛋-37_conversation_2.json'
    process_attraction_data(input_file)
