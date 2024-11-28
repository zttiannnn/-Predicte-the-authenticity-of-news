from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 2x faster
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # 4bit for 405b!
    "unsloth/Mistral-Small-Instruct-2409",     # Mistral 22b 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!

    "unsloth/Llama-3.2-1B-bnb-4bit",           # NEW! Llama 3.2 models
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-3B-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit", # or choose "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass
###########################################import dataset 
from datasets import load_dataset
dataset = load_dataset("json", data_files="/home/ubuntu-user/robot_repo/-Predicte-the-authenticity-of-news/robot_repo/ds_course/train.json", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import wandb

# wandb.init(
#     project = "llama3.2_3b_i_128_25",
#     name = "llama3.2_3b_i_128_25",
#     config = {
#         "model_name": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
#         "max_seq_length": max_seq_length,
#         "dtype": dtype,
#         "load_in_4bit": load_in_4bit,
#         "learning_rate": 1e-4,
#         "per_device_train_batch_size": 64,
#         "gradient_accumulation_steps": 2,
#         "warmup_steps": 5,
#         "max_steps": 60,
#         "fp16": not is_bfloat16_supported(),
#         "bf16": is_bfloat16_supported(),
#         "optim": "adamw_8bit",
#         "weight_decay": 0.01,
#         "lr_scheduler_type": "cosine",
#         "output_dir":"outputs",
#         "seed": 3407,
#     },
# )
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 64,
        gradient_accumulation_steps = 2,
        warmup_steps = 5,
        num_train_epochs = 25, # Set this for 1 full training run.
        # max_steps = 60,
        learning_rate = 1e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs",
        report_to = "wandb", # Use this for WandB etc
        save_strategy = "epoch",
        save_steps = 4,
    ),
)
# checkpoint_path= "outputs/checkpoint-1491"
# trainer.train(resume_from_checkpoint=checkpoint_path)

# trainer_stats = trainer.train()

# #@title Show final memory and time stats
# used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
# used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
# used_percentage = round(used_memory         /max_memory*100, 3)
# lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
# print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
# print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
# print(f"Peak reserved memory = {used_memory} GB.")
# print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
# print(f"Peak reserved memory % of max memory = {used_percentage} %.")
# print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# model.save_pretrained("llama3.2_3Bi_64bs25ep") # Local saving
# tokenizer.save_pretrained("llama3.2_3Bi_64bs25ep")
# model.push_to_hub("zttiannnn/llama3.2_3Bi_64bs25ep", token = "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") # Online saving
# tokenizer.push_to_hub("zttiannnn/llama3.2_3Bi_64bs25ep", token = "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") # Online saving



FastLanguageModel.for_inference(model)
inputs = tokenizer(
[
    alpaca_prompt.format(
        "Please judge whether this news is true or false and give resson.", # instruction
        "id: 12169.json, statement: There have not been any public safety issues in cities that allow transgender people to use the bathroom of the gender they identify as., evidence: \"There have not been any public safety issues\" in cities that allow transgender people to use the bathroom of the gender they identify as. ; Chris Sgro, the executive director of Equality NC, said that \"There have not been any public safety issues in those other communities\" with ordinances allowing transgender people to use the bathroom of their choice.; We asked the N.C. GOP if they could point to anything that backs up the safety fears. They provided a link to a news story in Seattle from earlier this year, about a man who had twice gone into a women’s locker room and began undressing. Seattle does allow transgender people to use the bathroom of the gender they identify as.; Maza has also polled public school systems that allow transgender students to use the bathroom of the gender they identify as. In a June 2015 article, he wrote that in 17 districts with a total of 600,00 students, officials hadn’t reported a single incident of \"harassment or inappropriate behavior\" related to transgender students and bathrooms.; \"There have not been any public safety issues in those other communities,\" Sgro said at a rally outside the legislature just days before Charlotte’s bill was overturned.", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 256)