import json
from mosaic import Mosaic
from sklearn.metrics import precision_score, recall_score, f1_score
import time
import yaml
import argparse

start_time = time.time()

threshold = 0

with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

#model_list = ["openai-community/gpt2-medium", "openai-community/gpt2"]
#model_list = ["meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-7b-hf"] # Make sure your Hugging Face token key is in ./config.yaml
#model_list = ["Unbabel/TowerBase-13B-v0.1", "TowerBase-7B-v0.1"] # Bigger model

MODEL_SETS = {
    "gpt2": ["openai-community/gpt2-medium", "openai-community/gpt2"],
    "llama": ["meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-7b-hf"], # Make sure your Hugging Face token key is in ./config.yaml and you have acces to meta-llama/Llama-2-7b-hf
    "tower": ["Unbabel/TowerBase-13B-v0.1", "Unbabel/TowerBase-7B-v0.1"] # Bigger model
}

#mosaic = Mosaic(model_list)

def load_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_predictions_to_file(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def evaluate_model(human_file_path, generated_file_path, output_file_path, mosaic):
    human_data = load_json_file(human_file_path)
    generated_data = load_json_file(generated_file_path)

    y_true = []
    y_pred = []
    predictions = []

    # Open the output file in write mode to start fresh
    with open(output_file_path, 'w') as file:
        json.dump({"predictions": []}, file, indent=4)

    for entry in human_data:
        text = entry["abs"]
        score = mosaic.compute_end_score(text)
        prediction = 0 if score >= threshold else 1
        y_true.append(0)  # 0 for human
        y_pred.append(prediction)

        prediction_entry = {
            "abs": text,
            "score": score,
            "prediction": "Generated" if prediction == 1 else "Not Generated",
            "source": "Human"
        }

        predictions.append(prediction_entry)

        # Append the new prediction to the output file
        with open(output_file_path, 'r+') as file:
            data = json.load(file)
            data["predictions"].append(prediction_entry)
            file.seek(0)
            json.dump(data, file, indent=4)

    for entry in generated_data:
        text = entry["abs"]
        score = mosaic.compute_end_score(text)
        prediction = 0 if score >= threshold else 1
        y_true.append(1)  # 1 for generated
        y_pred.append(prediction)

        prediction_entry = {
            "abs": text,
            "score": score,
            "prediction": "Generated" if prediction == 1 else "Not Generated",
            "source": "Generated"
        }

        predictions.append(prediction_entry)

        # Append the new prediction to the output file
        with open(output_file_path, 'r+') as file:
            data = json.load(file)
            data["predictions"].append(prediction_entry)
            file.seek(0)
            json.dump(data, file, indent=4)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    output_data = {
        "input_files": {
            "human_data": human_file_path,
            "generated_data": generated_file_path
        },
        "evaluation_metrics": {
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        },
        "predictions": predictions
    }

    # Save the complete output data at the end
    save_predictions_to_file(output_data, output_file_path)

    return precision, recall, f1

def main():
    parser = argparse.ArgumentParser(description="Evaluate models using different model sets.")
    parser.add_argument("--model_set", type=str, choices=MODEL_SETS.keys(), help="The model set to use.", default="gpt2")
    args = parser.parse_args()

    start_time = time.time()
    threshold = 0

    with open('./config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    model_list = MODEL_SETS[args.model_set]
    mosaic = Mosaic(model_list)

    human_file_path = config['datasets']['default_hum']
    generated_file_path = config['datasets']['default_gen']
    output_file_path = './results/Mosaic/test_gen_human-micro_retracted-fake_papers_train_part_public_extended.json'

    precision, recall, f1 = evaluate_model(human_file_path, generated_file_path, output_file_path, mosaic)

    end_time = time.time()
    running_time = end_time - start_time

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Running Time: {running_time:.2f} seconds")

if __name__ == "__main__":
    main()
