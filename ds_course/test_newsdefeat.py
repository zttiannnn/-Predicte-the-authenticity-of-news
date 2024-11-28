############################################################################################################
# # 读取 test.json 文件
# with open('/home/ubuntu-user/robot_repo/-Predicte-the-authenticity-of-news/robot_repo/ds_course/test.json', 'r') as f:
#     news_list = json.load(f)

# # 打开输出的 CSV 文件
# with open('/home/ubuntu-user/robot_repo/-Predicte-the-authenticity-of-news/robot_repo/ds_course/testnewsoutput.csv', 'w', newline='', encoding='utf-8') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     # 写入表头
#     csvwriter.writerow(['id', 'label'])

#     # 遍历每一条新闻
#     for news_item in news_list:
#         # 获取新闻的 id、statement 和 evidence
#         news_id = news_item.get('id', '')
#         statement = news_item.get('statement', '')
#         evidence = news_item.get('evidence', '')

#         # 构建输入文本
#         input_text = f"id: {news_id}, statement: {statement}, evidence: {evidence}"

#         # 格式化提示文本
#         prompt = alpaca_prompt.format(instruction, input_text, "")

#         inputs = tokenizer(prompt, return_tensors="pt").to(device)

#         outputs = model.generate(**inputs, max_new_tokens=128)
#         output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

#         # 提取模型的响应
#         response_start = output_text.find("### Response:") + len("### Response:")
#         response = output_text[response_start:].strip()

#         # 使用正则表达式从响应中提取标签
#         match = re.search(r"Label:\s*(\w+-?\w*)", response, re.IGNORECASE)
#         if match:
#             label = match.group(1).lower()
#         else:
#             label = 'unknown'

#         # 将 id 和标签写入 CSV
#         csvwriter.writerow([news_id, label])
















# # 修改读取和输出文件路径
# test_json_path = '/home/ubuntu-user/robot_repo/-Predicte-the-authenticity-of-news/robot_repo/ds_course/test.json'
# output_csv_path = '/home/ubuntu-user/robot_repo/-Predicte-the-authenticity-of-news/robot_repo/ds_course/testnewsoutput.csv'
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel
# import torch
# import json
# import csv
# import re

# # 模型路径
# base_model_name = "/home/ubuntu-user/robot_repo/-Predicte-the-authenticity-of-news/llama3.2_3Bi_64bs25ep"
# lora_model_name = "/home/ubuntu-user/robot_repo/-Predicte-the-authenticity-of-news/llama3.2_3Bi_64bs25ep"

# # 加载分词器
# tokenizer = AutoTokenizer.from_pretrained(lora_model_name)

# # 如果 eos_token_id 是 None，手动设置为 2（LLaMA 的 eos_token_id 通常为 2）
# if tokenizer.eos_token_id is None:
#     tokenizer.eos_token_id = 2

# # 设置 pad_token_id 为 eos_token_id
# tokenizer.pad_token_id = tokenizer.eos_token_id

# # 打印以验证设置
# print(f"EOS Token ID: {tokenizer.eos_token_id}")
# print(f"PAD Token ID: {tokenizer.pad_token_id}")

# # 加载模型，不传递未使用的关键词参数
# model = AutoModelForCausalLM.from_pretrained(base_model_name, return_dict=True)
# model = PeftModel.from_pretrained(model, lora_model_name)

# # 设置模型的 pad_token_id
# model.config.pad_token_id = tokenizer.pad_token_id

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # 定义提示模板
# alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

# ### Instruction:
# {}

# ### Input:
# {}

# ### Response:
# {}"""

# instruction = "please assess the authenticity of this news and provide reasons"


# # 读取 test.json 文件
# with open(test_json_path, 'r', encoding='utf-8') as f:
#     news_list = json.load(f)

# # 打开输出的 CSV 文件
# with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     # 写入表头
#     csvwriter.writerow(['id', 'label'])

#     # 遍历每一条新闻
#     for news_item in news_list:
#         # 获取新闻的 id、statement 和 evidence
#         news_id = news_item.get('id', '')
#         statement = news_item.get('statement', '')
#         evidence = news_item.get('evidence', '')

#         # 构建输入文本
#         input_text = f"id: {news_id}, statement: {statement}, evidence: {evidence}"

#         # 格式化提示文本
#         prompt = alpaca_prompt.format(instruction, input_text, "")

#         # 编码输入
#         inputs = tokenizer(prompt, return_tensors="pt").to(device)

#         # 生成输出，显式传递 pad_token_id
#         outputs = model.generate(**inputs, max_new_tokens=256, pad_token_id=tokenizer.pad_token_id)
#         output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

#         # 提取模型的响应
#         response_start = output_text.find("### Response:") + len("### Response:")
#         response = output_text[response_start:].strip()

#         # 使用正则表达式从响应中提取标签
#         match = re.search(r"Label:\s*([a-zA-Z-]+)", response, re.IGNORECASE)
#         if match:
#             label = match.group(1).lower()
#         else:
#             label = 'unknown'

#         # 将 id 和标签写入 CSV
#         csvwriter.writerow([news_id, label])
