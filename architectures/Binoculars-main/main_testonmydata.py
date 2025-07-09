import json
from binoculars import Binoculars
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
import time

start_time = time.time()

torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)

bino = Binoculars(
    observer_name_or_path="tiiuae/falcon-rw-1b", #tiiuae/falcon-7b bigger model
    performer_name_or_path="tiiuae/falcon-rw-1b" #tiiuae/falcon-7b-instruct bigger model
)

def load_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_predictions_to_file(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def evaluate_model(human_file_path, ai_file_path, output_file_path):
    human_data = load_json_file(human_file_path)
    ai_data = load_json_file(ai_file_path)

    y_true = []
    y_pred = []

    human_predictions = []
    ai_predictions = []

    for entry in human_data:
        text = entry["abs"]
        prediction = bino.predict(text)
        prediction_label = 0 if prediction == "Most likely Human-Written" else 1

        y_true.append(0)  # 0 for human
        y_pred.append(prediction_label)

        human_predictions.append({
            "abs": text,
            "prediction": prediction
        })

    for entry in ai_data:
        text = entry["abs"]
        prediction = bino.predict(text)
        prediction_label = 0 if prediction == "Most likely Human-Written" else 1

        y_true.append(1)  # 1 for AI
        y_pred.append(prediction_label)

        ai_predictions.append({
            "abs": text,
            "prediction": prediction
        })

    save_predictions_to_file({
        "human_predictions": human_predictions,
        "ai_predictions": ai_predictions
    }, output_file_path)

    # Calculate evaluation metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return precision, recall, f1

human_file_path = './datasets/human-micpro_original-fake_papers_train_part_public_extended.json'
ai_file_path = './datasets/gen-micro_retracted-fake_papers_train_part_public_extended.json'
output_file_path = './results/Binoculars/falcon-rw-1b_Binoculars_gen_human-micro_retracted-fake_papers_train_part_public_extended.json'

precision, recall, f1 = evaluate_model(human_file_path, ai_file_path, output_file_path)

end_time = time.time()

running_time = end_time - start_time

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Running Time: {running_time:.2f} seconds")