import json
import yaml
import os
import torch
import argparse
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

hugging_face_token = config['api_keys']['hugging_face']
os.environ["HF_TOKEN"] = hugging_face_token

def load_model_and_tokenizer(model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.bos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        token=hugging_face_token  
    )
    return model, tokenizer

def generate(query, model, tokenizer):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Give your output directly."},
        {"role": "user", "content": f"Refine this for me please: {query}"},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    outputs = model.generate(
        input_ids,
        max_new_tokens=1000,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.bos_token_id
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

def rewrite(input_path, output_path, model, tokenizer):
    with open(input_path, 'r') as f:
        data = json.load(f)

    rewrite_data = []
    for item in tqdm(data):
        refined = generate(item["abs"], model, tokenizer)
        rewrite_data.append({
            "title": item["title"],
            "original": item["abs"],
            "refined": refined
        })

    with open(output_path, 'w') as f:
        json.dump(rewrite_data, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Refine abstracts using a language model.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input JSON file.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output JSON file.')
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer()
    rewrite(args.input, args.output, model, tokenizer)
