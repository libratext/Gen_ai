import os
import json
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)


# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model_name = "./models/llama-3-8b/"

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
    low_cpu_mem_usage=True
)

def generate(query):
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

def rewrite(domains):
    for domain in domains:
        print(domain)
        
        with open(f'../dataset/{domain}/human_train.json', 'r') as file:
            human = json.load(file)

        with open(f'../dataset/{domain}/AI_train.json', 'r') as file:
            AI = json.load(file)

        rewrite_human, rewrite_AI = [], []
        for i in tqdm(range(len(human))):
            rewrite_human.append({"input": human[i]['input'], "Refine this for me please": generate(human[i]['input'])})
            with open(f"../dataset/{domain}/human_rewrite_train.json", 'w') as f:
                json.dump(rewrite_human, f)

        for i in tqdm(range(len(AI))):
            rewrite_AI.append({"input": AI[i]['input'], "Refine this for me please": generate(AI[i]['input'])})
            with open(f"../dataset/{domain}/AI_rewrite_train.json", 'w') as f:
                json.dump(rewrite_AI, f)

        with open(f'../dataset/{domain}/human_test.json', 'r') as file:
            human = json.load(file)

        with open(f'../dataset/{domain}/AI_test.json', 'r') as file:
            AI = json.load(file)

        rewrite_human, rewrite_AI = [], []
        for i in tqdm(range(len(human))):
            rewrite_human.append({"input": human[i]['input'], "Refine this for me please": generate(human[i]['input'])})
            with open(f"../dataset/{domain}/human_rewrite_test.json", 'w') as f:
                json.dump(rewrite_human, f)

        for i in tqdm(range(len(AI))):
            rewrite_AI.append({"input": AI[i]['input'], "Refine this for me please": generate(AI[i]['input'])})
            with open(f"../dataset/{domain}/AI_rewrite_test.json", 'w') as f:
                json.dump(rewrite_AI, f)

if __name__ == "__main__":
    domains = ["AcademicResearch", "ArtCulture", "Business", "Code", "EducationMaterial", "Entertainment", "Environmental", "Finance", "FoodCusine", "GovernmentPublic", "LegalDocument", "LiteratureCreativeWriting", "MedicalText", "NewsArticle", "OnlineContent", "PersonalCommunication", "ProductReview", "Religious", "Sports", "TechnicalWriting", "TravelTourism"]
    
    rewrite(domains)