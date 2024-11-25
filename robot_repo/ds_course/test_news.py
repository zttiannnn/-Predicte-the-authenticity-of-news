from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base_model_name = "unsloth/Meta-Llama-3.1-8B"
lora_model_name = "zttiannnn/lora_model"
tokenizer = AutoTokenizer.from_pretrained(lora_model_name)

model = AutoModelForCausalLM.from_pretrained(base_model_name, return_dict=True)
model = PeftModel.from_pretrained(model, lora_model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

instruction = "please assess the authenticity of this news and provide reasons"
input = "id: 5209.json, statement: Suzanne Bonamici supports a plan that will cut choice for Medicare Advantage seniors., evidence: Mary Anne Ostrom, Cornilles’ campaign manager, says that’s besides the point. Bonamici supports the Affordable Care Act. Ostrom says that means Bonamici supports eliminating choice for seniors. ; Republican Rob Cornilles continues his attack on Democrat Suzanne Bonamici, claiming once again this week on Think Out Loud that her support of President Obama’s 2010 health care reform act means that she supports cuts to seniors on Medicare and Medicare Advantage. ; \"…She supports taking $500 billion away from Medicare. I don’t support that. It would largely affect those seniors who are on Medicare Advantage, which is about a quarter of a million Oregon seniors. They’ve chosen that option. I want to maintain they have that choice. She does not.\" ; \"Whether services are cut, contracts terminated or out-of-pocket expenses grow unaffordable, it is completely reasonable to suggest that when people must drop from a plan they like, they are forced out. A $900-a-year increase in out-of-pocket costs would be considered unaffordable by many seniors. They have no choice,\" Ostrom said in an email to PolitiFact Oregon in November 2011. ; In the past we gave Cornilles a False for perpetuating the inaccurate statement that Bonamici and other Democrats support $500 billion in cuts to Medicare, hurting seniors. Here, however, he’s making a more specific point about seniors on Medicare Advantage."

# 格式化提示文本
prompt = alpaca_prompt.format(instruction, input, "")

inputs = tokenizer(prompt, return_tensors="pt").to(device)

outputs = model.generate(**inputs, max_new_tokens=128)
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)


response_start = output_text.find("### Response:") + len("### Response:")
response = output_text[response_start:].strip()
print(response)