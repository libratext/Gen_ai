import json
from sklearn.metrics import precision_score, recall_score, f1_score

from model import GPT2PPLV2 as GPT2PPL

import time
import yaml
import os

start_time = time.time()

model = GPT2PPL()

with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def load_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def append_to_json_file(data, file_path):
    with open(file_path, 'a') as file:
        json.dump(data, file)
        file.write('\n')

def map_prediction_to_label(prediction_text):
    if "Human" in prediction_text:
        return 0
    elif "A.I." in prediction_text:
        return 1
    else:
        print(f"Unexpected prediction text: {prediction_text}")
        return None

def evaluate_model(human_file_path, generated_file_path, output_predictions_path, output_metrics_path):
    human_data = load_json_file(human_file_path)
    generated_data = load_json_file(generated_file_path)

    y_true = []
    y_pred = []

    with open(output_predictions_path, 'w') as file:
        pass

    start_time = time.time()

    for entry in human_data:
        sentence = entry["abs"]
        prediction = model(sentence, 100, "v1.1") 
        y_true.append(0)  # 0 for human
        label = map_prediction_to_label(prediction[1])
        y_pred.append(label)

        prediction_data = {
            "content": sentence,
            "prediction": label,
            "probability": prediction[0],
            "source": human_file_path
        }
        append_to_json_file(prediction_data, output_predictions_path)

    for entry in generated_data:
        sentence = entry["abs"]
        prediction = model(sentence, 100, "v1.1")
        y_true.append(1)  # 1 for generated
        label = map_prediction_to_label(prediction[1])
        y_pred.append(label)

        prediction_data = {
            "content": sentence,
            "prediction": label,
            "probability": prediction[0],
            "source": generated_file_path
        }
        append_to_json_file(prediction_data, output_predictions_path)

    valid_indices = [i for i, label in enumerate(y_pred) if label is not None]
    y_true_filtered = [y_true[i] for i in valid_indices]
    y_pred_filtered = [y_pred[i] for i in valid_indices]

    precision = precision_score(y_true_filtered, y_pred_filtered)
    recall = recall_score(y_true_filtered, y_pred_filtered)
    f1 = f1_score(y_true_filtered, y_pred_filtered)

    metrics_data = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "human_data_path": human_file_path,
        "generated_data_path": generated_file_path
    }

    with open(output_metrics_path, 'w') as file:
        json.dump(metrics_data, file, indent=4)

    end_time = time.time()

    running_time = end_time - start_time
    print(f"Running Time: {running_time:.2f} seconds")

    return precision, recall, f1

human_file_path = (
        config['datasets']['your-dataset_hum']
        if config['datasets']['your-dataset_hum'] != "the_path_to_your_hum_dataset"
        else config['datasets']['default_hum']
    )

generated_file_path = (
        config['datasets']['your-dataset_gen']
        if config['datasets']['your-dataset_gen'] != "the_path_to_your_gen_dataset"
        else config['datasets']['default_gen']
    )

output_predictions_path = './results/DetectGPT/'+os.path.basename(generated_file_path)+'_predictions_output.json'
output_metrics_path = './results/DetectGPT/'+os.path.basename(generated_file_path)+'_evaluation_metrics.json'

precision, recall, f1 = evaluate_model(human_file_path, generated_file_path, output_predictions_path, output_metrics_path)

end_time = time.time()

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
running_time = end_time - start_time