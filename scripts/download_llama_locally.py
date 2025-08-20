from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import yaml
import os
from huggingface_hub import login

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
local_dir = "./models/llama-3-8b"  # Directory where the model will be saved

with open('./config.yaml', 'r') as file:
        config = yaml.safe_load(file)
hugging_face_token = config['api_keys']['hugging_face']
os.environ["HF_TOKEN"] = hugging_face_token

login(token=hugging_face_token)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # or torch.bfloat16 if supported
    low_cpu_mem_usage=True,
)
model.save_pretrained(local_dir)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(local_dir)

print(f"Model and tokenizer saved to {local_dir}")
