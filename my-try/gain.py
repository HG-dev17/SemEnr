import requests
import json
from tqdm import tqdm

def askLocalQwen2Model(prompt):
    url = "http://localhost:11434/api/generate"
    
    # 创建要发送的JSON对象
    json_input = {
        "model": "qwen2.5-coder:latest",
        "prompt": prompt,
        "stream": False
    }
    
    try:
        # 发送POST请求
        response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(json_input))
        
        # 检查响应状态码
        if response.status_code == 200:
            # 解析JSON响应并提取response字段
            json_response = response.json()
            return json_response.get("response", "")
        else:
            print(f"Failed to get response from Qwen2 model. Status code: {response.status_code}")
            return ""
    except requests.RequestException as e:
        print(f"An error occurred: {e}")
        return ""

# 读取输入文件
input_file_path = "train_desc.txt"
output_file_path = "final_desc.txt"
train_prompt="Please generate only the similar descriptive text without any extra content. The output should be approximately the same length as the input and in English.\n"
with open(input_file_path, 'r', encoding='utf-8') as input_file:
    lines = input_file.readlines()

# 处理每一行并写入输出文件
with tqdm(total=len(lines), desc="Processing", unit="line") as pbar:
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for line in lines:
            # 去除行尾的换行符
            prompt = train_prompt + line.strip()
            
            # 使用模型处理生成相似文本
            result = askLocalQwen2Model(prompt)
            
            # 将结果写入输出文件
            output_file.write(result + "\n")
            
            # 更新进度条
            pbar.update(1)

print("处理完成，结果已保存到final.txt")