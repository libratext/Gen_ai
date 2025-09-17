from mosaic import Mosaic
import argparse

def predict_text(text, model_set="gpt2"):
    """
    Predict whether the input text is human-written or machine-generated using MOSAIC.

    Args:
        text (str): Input text (sentence or document).
        model_set (str): The model set to use. Options: "gpt2", "llama", or "tower".

    Returns:
        tuple: (score, prediction_label)
    """
    MODEL_SETS = {
        "gpt2": ["openai-community/gpt2-medium", "openai-community/gpt2"],
        "llama": ["meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-7b-hf"],  # Ensure Hugging Face token is set
        "tower": ["Unbabel/TowerBase-13B-v0.1", "Unbabel/TowerBase-7B-v0.1"]  # Bigger model
    }

    mosaic = Mosaic(MODEL_SETS[model_set])

    score = mosaic.compute_end_score(text)

    threshold = 0
    prediction_label = "Generated" if score < threshold else "Not Generated"

    return score, prediction_label

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_text", type=str, required=True, help='Add your text or document to be predicted as generated or human-written')
    parser.add_argument("--model", type=str, required=True, default="gpt2", help='Use one model from Mosaic ["gpt2"/"llama"/"tower"]')
    args = parser.parse_args()

    input_text = args.input_text

    model_set = args.model

    score, prediction = predict_text(input_text, model_set)

    print(f"Score: {score:.4f}")
    print(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
