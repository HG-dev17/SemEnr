import requests
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
import signal
# 读取输入文件
input_file_path = "train_desc.txt" ##输入文件
output_file_path = "final_desc.txt" ##输出文件
model_name = "qwen2.5-coder:latest" ## 模型名称
num_workers = 1000 ## 线程数(目前1000最佳)
# temp_dir = "temp_files"
# os.makedirs(temp_dir, exist_ok=True)


# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def askLocalQwen2Model(prompt):
    url = "http://localhost:11434/api/generate"
    
    # 创建要发送的JSON对象
    json_input = {
        "model": model_name, 
        "prompt": prompt,
        "stream": False
    }
    
    try:
        # 发送POST请求
        response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(json_input))
        
        # 检查响应状态码
        if response.status_code == 200: #成功
            # 解析JSON响应并提取response字段
            json_response = response.json()
            return json_response.get("response", "")
        else:
            logging.error(f"Failed to get response from model. Status code: {response.status_code}")
            return ""
    except requests.RequestException as e:
        logging.error(f"An error occurred: {e}")
        return ""

## 模型提示词（要求）
train_prompt = "Please generate only the similar descriptive text without any extra content. The output should be approximately the same length as the input and in English.\n"
with open(input_file_path, 'r', encoding='utf-8') as input_file:
    lines = input_file.readlines()
results = []
def save_results_and_exit(signum, frame):
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for i in range(len(lines)):
            result = next((res for index, res in results if index == i), None)
            if result is not None:
                output_file.write(result + "\n")
    logging.info("Results saved to file. Exiting... 多线程无法返回命令提示符，请自行关闭窗口")
    exit(0)
# 捕获中断信号，防止Ctrl+C中断程序
signal.signal(signal.SIGINT, save_results_and_exit)
with tqdm(total=len(lines), desc="Processing", unit="line") as pbar:
    ##并行处理数量
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(askLocalQwen2Model, train_prompt + line.strip()): i for i, line in enumerate(lines)}
        
        for future in as_completed(futures):
            i = futures[future]
            try:
                result = future.result()
                results.append((i, result))
            except Exception as e:
                logging.error(f"Error processing line {i}: {e}")
                results.append((i, ""))
            
            pbar.update(1)
# 将结果写入输出文件
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for i in range(len(lines)):
        result = next((res for index, res in results if index == i), None)
        output_file.write(result + "\n")
print("处理完成，结果已保存到final.txt")