from model import GPT2PPLV2 as GPT2PPL
import argparse


def predict_text(text, perturbation_count=100, version="v1.1"):
    """
    Predict whether the input text is human-written or machine-generated using DetectGPT.

    Args:
        text (str): Input text (sentence or document).
        perturbation_count (int): Number of perturbations for DetectGPT.
        version (str): Version of DetectGPT to use.

    Returns:
        str: Prediction result ("Human" or "A.I.").
    """

    model = GPT2PPL()

    probability, prediction_text = model(text, perturbation_count, version)
    return prediction_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_text', type=str, required=True, help='Add your text or document to be predicted as generated or human-written')

    args = parser.parse_args()

    input_text = args.input_text

    prediction = predict_text(input_text)

    print(f"The document : {input_text}\n")
    print(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
