import json
import time
from sklearn.metrics import precision_score, recall_score, f1_score

from model import GPT2PPLV2 as GPT2PPL

model = GPT2PPL()

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
        raise ValueError(f"Unexpected prediction text: {prediction_text}")

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

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

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

human_file_path = './datasets/human-micpro_original-fake_papers_train_part_public_extended.json'
generated_file_path = './datasets/gen-micro_retracted-fake_papers_train_part_public_extended.json'
output_predictions_path = './results/DetectGPT/kaggle_predictions_output.json'
output_metrics_path = './results/DetectGPT/kaggle_evaluation_metrics.json'

precision, recall, f1 = evaluate_model(human_file_path, generated_file_path, output_predictions_path, output_metrics_path)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
