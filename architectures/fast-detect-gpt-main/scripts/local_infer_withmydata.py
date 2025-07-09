import argparse
import json
import time
import numpy as np
import torch
from scipy.stats import norm
from sklearn.metrics import precision_score, recall_score, f1_score
from model import load_tokenizer, load_model
from fast_detect_gpt import get_sampling_discrepancy_analytic

def compute_prob_norm(x, mu0, sigma0, mu1, sigma1):
    pdf_value0 = norm.pdf(x, loc=mu0, scale=sigma0)
    pdf_value1 = norm.pdf(x, loc=mu1, scale=sigma1)
    prob = pdf_value1 / (pdf_value0 + pdf_value1)
    return prob

class FastDetectGPT:
    def __init__(self, args):
        self.args = args
        self.criterion_fn = get_sampling_discrepancy_analytic
        self.scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.cache_dir)
        self.scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
        self.scoring_model.eval()

        if args.sampling_model_name != args.scoring_model_name:
            self.sampling_tokenizer = load_tokenizer(args.sampling_model_name, args.cache_dir)
            self.sampling_model = load_model(args.sampling_model_name, args.device, args.cache_dir)
            self.sampling_model.eval()

        distrib_params = {
            'gpt-j-6B_gpt-neo-2.7B': {'mu0': 0.2713, 'sigma0': 0.9366, 'mu1': 2.2334, 'sigma1': 1.8731},
            'gpt-neo-2.7B_gpt-neo-2.7B': {'mu0': -0.2489, 'sigma0': 0.9968, 'mu1': 1.8983, 'sigma1': 1.9935},
            'falcon-7b_falcon-7b-instruct': {'mu0': -0.0707, 'sigma0': 0.9520, 'mu1': 2.9306, 'sigma1': 1.9039},
        }
        key = f'{args.sampling_model_name}_{args.scoring_model_name}'
        self.classifier = distrib_params[key]

    def compute_crit(self, text):
        tokenized = self.scoring_tokenizer(text, truncation=True, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.args.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = self.scoring_model(**tokenized).logits[:, :-1]
            if self.args.sampling_model_name == self.args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = self.sampling_tokenizer(text, truncation=True, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.args.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = self.sampling_model(**tokenized).logits[:, :-1]
            crit = self.criterion_fn(logits_ref, logits_score, labels)
        return crit, labels.size(1)

    def compute_prob(self, text):
        crit, ntoken = self.compute_crit(text)
        mu0 = self.classifier['mu0']
        sigma0 = self.classifier['sigma0']
        mu1 = self.classifier['mu1']
        sigma1 = self.classifier['sigma1']
        prob = compute_prob_norm(crit, mu0, sigma0, mu1, sigma1)
        return prob

def load_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def evaluate_predictions(true_labels, pred_labels):
    precision = precision_score(true_labels, pred_labels, pos_label='generated')
    recall = recall_score(true_labels, pred_labels, pos_label='generated')
    f1 = f1_score(true_labels, pred_labels, pos_label='generated')
    return precision, recall, f1

def run(args, human_file_path, generated_file_path, output_file_path):
    
    start_time = time.time()

    detector = FastDetectGPT(args)
    start_time = time.time()

    human_data = load_json_file(human_file_path)
    generated_data = load_json_file(generated_file_path)

    predictions = []

    with open(output_file_path, 'w') as file:
        json.dump({"predictions": [], "evaluation_metrics": {}, "input_data_paths": {"human_data": human_file_path, "generated_data": generated_file_path}}, file, indent=4)

    for entry in human_data:
        text = entry['abs']
        prob = detector.compute_prob(text)
        prediction = {
            'text': text,
            'title': entry['title'],
            'prediction': 'human' if prob < 0.5 else 'generated',
            'probability': prob
        }
        predictions.append(prediction)

        with open(output_file_path, 'r+') as file:
            data = json.load(file)
            data["predictions"].append(prediction)
            file.seek(0)
            json.dump(data, file, indent=4)

    for entry in generated_data:
        text = entry['abs']
        prob = detector.compute_prob(text)
        prediction = {
            'text': text,
            'title': entry['title'],
            'prediction': 'human' if prob < 0.5 else 'generated',
            'probability': prob
        }
        predictions.append(prediction)

        with open(output_file_path, 'r+') as file:
            data = json.load(file)
            data["predictions"].append(prediction)
            file.seek(0)
            json.dump(data, file, indent=4)

    true_labels = ['human'] * len(human_data) + ['generated'] * len(generated_data)
    pred_labels = [pred['prediction'] for pred in predictions]

    precision, recall, f1 = evaluate_predictions(true_labels, pred_labels)

    with open(output_file_path, 'r+') as file:
        data = json.load(file)
        data["evaluation_metrics"] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        data["running_time"] = time.time() - start_time
        file.seek(0)
        json.dump(data, file, indent=4)

    running_time = time.time() - start_time

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Running Time: {running_time:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampling_model_name', type=str, default="gpt-j-6B")
    parser.add_argument('--scoring_model_name', type=str, default="gpt-neo-2.7B")
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    parser.add_argument('--human_file_path', type=str, required=True, help='Path to the JSON file with human-written text')
    parser.add_argument('--generated_file_path', type=str, required=True, help='Path to the JSON file with generated text')
    parser.add_argument('--output_file_path', type=str, required=True, help='Path to save the output JSON file')
    args = parser.parse_args()

    run(args, args.human_file_path, args.generated_file_path, args.output_file_path)
