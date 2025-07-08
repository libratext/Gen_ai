import os
import openai
import json

import yaml 

with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

openai_api_key = config['api_keys']['openai']
openai.api_key = openai_api_key

#openai.api_key='KEY'

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def openai_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

debug=False


def GPT_self_prompt(prompt_str, content_to_be_detected):

    # import pdb; pdb.set_trace()

    response = openai_backoff(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "user",
                            "content": f"{prompt_str}: \"{content_to_be_detected}\"",
                        }
                    ],
                )
    spit_out = response["choices"][0]["message"]["content"].strip()
    print(spit_out)
    return spit_out

prompt_list = ['Revise this with your best effort', 'Help me polish this', 'Rewrite this for me', 
                'Make this fluent while doing minimal change', 'Refine this for me please', 'Concise this for me and keep all the information',
                'Improve this in GPT way']

with open(f'./datasets/gen-micro_retracted-fake_papers_train_part_public_extended.json', 'r') as file:
    human = json.load(file)

with open(f'./datasets/gen-micro_retracted-fake_papers_train_part_public_extended.json', 'r') as file:
    GPT = json.load(file)


def rewrite_json(input_json, prompt_list, human=False):
    all_data = []
    for cc, data in enumerate(input_json):
        tmp_dict ={}
        
        tmp_dict['input'] = data['abs']

        for ep in prompt_list:
            tmp_dict[ep] = GPT_self_prompt(ep, tmp_dict['input'])
        
        all_data.append(tmp_dict)

        if debug:
            break
    return all_data

human_rewrite = rewrite_json(human, prompt_list, True)
with open(f'./results/Raidar/rewrite_arxiv_human_inv.json', 'w') as file:
    json.dump(human_rewrite, file, indent=4)

GPT_rewrite = rewrite_json(GPT, prompt_list)
with open(f'./results/Raidar/rewrite_arxiv_GPT_inv.json', 'w') as file:
    json.dump(GPT_rewrite, file, indent=4)


