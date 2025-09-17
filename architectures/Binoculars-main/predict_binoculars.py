import torch
from binoculars import Binoculars
import argparse

class BinocularsDetector:
    def __init__(self, model_size="small"):
        """
        Initialize the Binoculars model.

        Args:
            model_size (str): Choose between "small" or "big" model.
        """
        if model_size == "small":
            self.bino = Binoculars(
                observer_name_or_path="tiiuae/falcon-rw-1b",
                performer_name_or_path="tiiuae/falcon-rw-1b"
            )
        elif model_size == "big":
            self.bino = Binoculars(
                observer_name_or_path="tiiuae/falcon-7b",
                performer_name_or_path="tiiuae/falcon-7b-instruct"
            )
        else:
            raise ValueError("Invalid model size. Choose 'small' or 'big'.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict(self, text):
        """
        Predict whether the input text is human-written or machine-generated.

        Args:
            text (str): Input text (sentence or document).

        Returns:
            str: Prediction result ("Most likely Human-Written" or "Most likely Machine-Generated").
        """
        return self.bino.predict(text)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_text', type=str, required=True, help='Add your text or document to be predicted as generated or human-written')
    parser.add_argument('--model', type=str, required=True, default="small", help='Use small or big model from Binoculars ["small"/"big"]')

    args = parser.parse_args()

    detector = BinocularsDetector(model_size=args.model)

    input_text = args.input_text

    prediction = detector.predict(input_text)

    print(f"The document : {input_text}\n")
    print(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
