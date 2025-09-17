import torch
import argparse
import numpy as np
from scipy.stats import norm
from model import load_tokenizer, load_model
from fast_detect_gpt import get_sampling_discrepancy_analytic

def compute_prob_norm(x, mu0, sigma0, mu1, sigma1):
    """Compute the probability of the text being generated using normal distribution parameters."""
    pdf_value0 = norm.pdf(x, loc=mu0, scale=sigma0)
    pdf_value1 = norm.pdf(x, loc=mu1, scale=sigma1)
    prob = pdf_value1 / (pdf_value0 + pdf_value1)
    return prob

class FastDetectGPT:
    def __init__(self, sampling_model_name, scoring_model_name, device, cache_dir):
        """
        Initialize the FastDetectGPT model.

        Args:
            sampling_model_name (str): Name of the sampling model.
            scoring_model_name (str): Name of the scoring model.
            device (str): Device to run the model on (e.g., "cpu" or "cuda").
            cache_dir (str): Directory to cache the models.
        """
        self.device = device
        self.criterion_fn = get_sampling_discrepancy_analytic
        self.scoring_tokenizer = load_tokenizer(scoring_model_name, cache_dir)
        self.scoring_model = load_model(scoring_model_name, device, cache_dir)
        self.scoring_model.eval()

        if sampling_model_name != scoring_model_name:
            self.sampling_tokenizer = load_tokenizer(sampling_model_name, cache_dir)
            self.sampling_model = load_model(sampling_model_name, device, cache_dir)
            self.sampling_model.eval()
        else:
            self.sampling_model = self.scoring_model
            self.sampling_tokenizer = self.scoring_tokenizer

        distrib_params = {
            'gpt-j-6B_gpt-neo-2.7B': {'mu0': 0.2713, 'sigma0': 0.9366, 'mu1': 2.2334, 'sigma1': 1.8731},
            'gpt-neo-2.7B_gpt-neo-2.7B': {'mu0': -0.2489, 'sigma0': 0.9968, 'mu1': 1.8983, 'sigma1': 1.9935},
            'falcon-7b_falcon-7b-instruct': {'mu0': -0.0707, 'sigma0': 0.9520, 'mu1': 2.9306, 'sigma1': 1.9039},
        }

        key = f'{sampling_model_name}_{scoring_model_name}'
        self.classifier = distrib_params[key]

    def compute_crit(self, text):
        """Compute the sampling discrepancy criterion for the input text."""
        tokenized = self.scoring_tokenizer(text, truncation=True, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.device)
        labels = tokenized.input_ids[:, 1:]

        with torch.no_grad():
            logits_score = self.scoring_model(**tokenized).logits[:, :-1]
            if hasattr(self, 'sampling_model'):
                tokenized_ref = self.sampling_tokenizer(text, truncation=True, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.device)
                assert torch.all(tokenized_ref.input_ids[:, 1:] == labels), "Tokenizer mismatch."
                logits_ref = self.sampling_model(**tokenized_ref).logits[:, :-1]
            else:
                logits_ref = logits_score

            crit = self.criterion_fn(logits_ref, logits_score, labels)
        return crit, labels.size(1)

    def compute_prob(self, text):
        """Compute the probability of the text being generated."""
        crit, _ = self.compute_crit(text)
        mu0 = self.classifier['mu0']
        sigma0 = self.classifier['sigma0']
        mu1 = self.classifier['mu1']
        sigma1 = self.classifier['sigma1']
        prob = compute_prob_norm(crit, mu0, sigma0, mu1, sigma1)
        return prob

def predict_text(text, sampling_model_name="gpt-j-6B", scoring_model_name="gpt-neo-2.7B", device="cpu", cache_dir="../cache"):
    """
    Predict whether the input text is human-written or machine-generated.

    Args:
        text (str): Input text (sentence or document).
        sampling_model_name (str): Name of the sampling model.
        scoring_model_name (str): Name of the scoring model.
        device (str): Device to run the model on (e.g., "cpu" or "cuda").
        cache_dir (str): Directory to cache the models.

    Returns:
        tuple: (probability, prediction_label)
    """
    detector = FastDetectGPT(sampling_model_name, scoring_model_name, device, cache_dir)
    prob = detector.compute_prob(text)
    prediction_label = 'generated' if prob >= 0.5 else 'human'
    return prob, prediction_label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_text', type=str, required=True, help='Add your text or document to be predicted as generated or human-written')
    parser.add_argument('--sampling_model_name', type=str, default="gpt-j-6B", required=False,help="")
    parser.add_argument('--scoring_model_name', type=str, default="gpt-neo-2.7B",required=False,help="")
    parser.add_argument('--device', type=str, default="cpu",required=False)
    parser.add_argument('--cache_dir', type=str, default="../cache",required=False)

    args = parser.parse_args()

    input_text = args.input_text

    probability, prediction = predict_text(input_text, args.sampling_model_name,args.scoring_model_name,args.device,args.cache_dir)

    print(f"Probability of being generated: {probability:.4f}")
    print(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
