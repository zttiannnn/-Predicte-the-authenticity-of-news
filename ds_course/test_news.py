from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import json
import csv
import re
# base_model_name = "unsloth/Meta-Llama-3.1-8B"
# lora_model_name = "zttiannnn/lora_model"
base_model_name = "/home/ubuntu-user/robot_repo/-Predicte-the-authenticity-of-news/llama3.2_3Bi_64bs25ep"
lora_model_name = "/home/ubuntu-user/robot_repo/-Predicte-the-authenticity-of-news/llama3.2_3Bi_64bs25ep"
tokenizer = AutoTokenizer.from_pretrained(lora_model_name)

model = AutoModelForCausalLM.from_pretrained(base_model_name, return_dict=True)
model = PeftModel.from_pretrained(model, lora_model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 如果 eos_token_id 是 None，手动设置为 2（LLaMA 的 eos_token_id 通常为 2）
if tokenizer.eos_token_id is None:
    tokenizer.eos_token_id = 2

# 设置 pad_token_id 为 eos_token_id
tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
# 打印以验证设置
# print(f"EOS Token ID: {tokenizer.eos_token_id}")
# print(f"PAD Token ID: {tokenizer.pad_token_id}")

# # 加载模型，不传递未使用的关键词参数
# model = AutoModelForCausalLM.from_pretrained(base_model_name, return_dict=True)
# model = PeftModel.from_pretrained(model, lora_model_name)

# 设置模型的 pad_token_id
model.config.pad_token_id = tokenizer.pad_token_id
# tokenizer.pad_token_id = tokenizer.eos_token_id
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

instruction = "please assess the authenticity of this news and provide reasons"
# input = "id: 12996.json, statement: The number of illegal immigrants could be 3 million. It could be 30 million., evidence: Is Trump right that there are 30 million or more illegal immigrants? We decided to see what the latest evidence shows.; The Department of Homeland Security says the number of illegal immigrants was about 11.4 million as of January 2012. Other independent groups that research illegal immigration put the number between 11 and 12 million. We found no compelling evidence that the number could as high as Trump said.; \"I don't think the 11 million -- which is a number you have been hearing for many many years, I've been hearing that number for five years -- I don't think that is an accurate number anymore,\" Trump said on MSNBC’s Morning Joe July 24. \"I am now hearing it's 30 million, it could be 34 million, which is a much bigger problem.\"; According to the department’s estimates, the number of illegal immigrants peaked around 12 million in 2007 and has gradually declined to closer to 11 million.; Camarota said that the reason the 30 million figure is unlikely is that the census asks other questions that allow researchers to estimate how good the data is -- for example, the number of births to immigrant mothers, school enrollment and death records, which helps shore up the figure. In other words, if there were three times the generally accepted number of illegal immigrants in the United States, they would show up in those other categories."

# # 格式化提示文本
# prompt = alpaca_prompt.format(instruction, input, "")

# inputs = tokenizer(prompt, return_tensors="pt").to(device)

# outputs = model.generate(**inputs, max_new_tokens=128,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
# output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)


# response_start = output_text.find("### Response:") + len("### Response:")
# response = output_text[response_start:].strip()
# print(response)
# Load the news data from test.json
with open('/home/ubuntu-user/robot_repo/-Predicte-the-authenticity-of-news/robot_repo/ds_course/test.json', 'r') as f:
    news_list = json.load(f)

# Open the CSV file to write results
with open('testnewsoutput.csv', 'w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['id', 'label'])

    # Process the first 100 news items
    for news_item in news_list[:10]:
        # Extract id, statement, and evidence
        # news_id = news_item.get('id', '')
        # statement = news_item.get('statement', '')
        # evidence = news_item.get('evidence', '')
        # print(f"{news_id},{statement},{evidence}")
        input_field = news_item.get('input', '')

        # 使用正则表达式提取 id、statement 和 evidence
        id_match = re.search(r'id:\s*([^,]+)', input_field)
        statement_match = re.search(r'statement:\s*([^,]+)', input_field)
        evidence_match = re.search(r'evidence:\s*(.*)', input_field)

        news_id = id_match.group(1).strip() if id_match else ''
        statement = statement_match.group(1).strip() if statement_match else ''
        evidence = evidence_match.group(1).strip() if evidence_match else ''

        # 打印以验证提取的数据
        print(f"{news_id},{statement},{evidence}")




        # Construct the input text
        input_text = f"id: {news_id}, statement: {statement}, evidence: {evidence}"

        # Format the prompt
        prompt = alpaca_prompt.format(instruction, input_text, "")

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate the model's response
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the response part
        response_start = output_text.find("### Response:") + len("### Response:")
        response = output_text[response_start:].strip()

        # Use regex to extract the label (assuming the model outputs "label: true/false, reason: ...")
        label_match = re.search(r"label\s*:\s*(true|false)", response, re.IGNORECASE)
        label = label_match.group(1).lower() if label_match else 'unknown'

        # Write the id and label to the CSV file
        csvwriter.writerow([news_id, label])
        print(f"Processed news id: {news_id}, label: {label}")