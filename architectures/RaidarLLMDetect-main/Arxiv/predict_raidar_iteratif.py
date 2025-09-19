import os
import openai
import argparse
import json
import time
import yaml
import numpy as np
from fuzzywuzzy import fuzz

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score,accuracy_score, classification_report, f1_score

# Load configuration
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

openai_api_key = config['api_keys']['openai']
openai.api_key = openai_api_key

# Retry mechanism for OpenAI API
from tenacity import retry, stop_after_attempt, wait_random_exponential

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def openai_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

# Function to rewrite text using LLM
def GPT_self_prompt(prompt_str, content_to_be_detected):
    response = openai_backoff(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"{prompt_str}: \"{content_to_be_detected}\"",
            }
        ],
    )
    return response["choices"][0]["message"]["content"].strip()

# Prompts for rewriting
prompt_list = [
    'Revise this with your best effort',
    'Help me polish this',
    'Rewrite this for me',
    'Make this fluent while doing minimal change',
    'Refine this for me please',
    'Concise this for me and keep all the information',
    'Improve this in GPT way'
]

# Tokenize and normalize text
def tokenize_and_normalize(sentence):
    return [word.lower().strip() for word in sentence.split()]

# Extract n-grams
def extract_ngrams(tokens, n):
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

# Find common elements between two lists
def common_elements(list1, list2):
    return set(list1) & set(list2)

# Calculate common n-grams and words between two sentences
def calculate_sentence_common(sentence1, sentence2):
    tokens1 = tokenize_and_normalize(sentence1)
    tokens2 = tokenize_and_normalize(sentence2)
    common_words = common_elements(tokens1, tokens2)
    number_common_hierarchy = [len(list(common_words))]

    for n in range(2, 5):  # 2-grams to 4-grams
        ngrams1 = extract_ngrams(tokens1, n)
        ngrams2 = extract_ngrams(tokens2, n)
        common_ngrams = common_elements(ngrams1, ngrams2)
        number_common_hierarchy.append(len(list(common_ngrams)))

    return number_common_hierarchy

# Sum two lists element-wise
def sum_for_list(a, b):
    return [aa + bb for aa, bb in zip(a, b)]

# Extract features for classification
def get_data_stat(data):
    processed_data = []
    for entry in data:
        original = entry['input']
        raw = tokenize_and_normalize(original)
        statistic_res = {}
        ratio_fzwz = {}
        all_statistic_res = [0 for _ in range(4)]  # For 1-gram to 4-gram
        cnt = 0
        whole_combined = ''

        for key in entry.keys():
            if key != 'input':
                whole_combined += (' ' + entry[key])
                res = calculate_sentence_common(original, entry[key])
                statistic_res[key] = res
                all_statistic_res = sum_for_list(all_statistic_res, res)
                ratio_fzwz[key] = [fuzz.ratio(original, entry[key]), fuzz.token_set_ratio(original, entry[key])]
                cnt += 1

        entry['fzwz_features'] = ratio_fzwz
        entry['common_features'] = statistic_res
        entry['avg_common_features'] = [a / cnt for a in all_statistic_res]
        entry['common_features_ori_vs_allcombined'] = calculate_sentence_common(original, whole_combined)
        processed_data.append(entry)

    return processed_data

# Prepare feature vector for classification
def get_feature_vec(input_json):
    all_list = []
    for each in input_json:
        raw = tokenize_and_normalize(each['input'])
        r_len = len(raw) * 1.0
        each_data_fea = []

        if len(raw) < cutoff_start or len(raw) > cutoff_end:
            continue

        each_data_fea = [ind_d / r_len for ind_d in each['avg_common_features']]

        for key in each['common_features'].keys():
            each_data_fea.extend([ind_d / r_len for ind_d in each['common_features'][key]])

        each_data_fea.extend([ind_d / r_len for ind_d in each['common_features_ori_vs_allcombined']])

        for key in each['fzwz_features'].keys():
            each_data_fea.extend(each['fzwz_features'][key])

        all_list.append(np.array(each_data_fea))

    return np.vstack(all_list) if all_list else np.array([])

# Train the classifier and evaluate
def xgboost_classifier(human, gpt4):
    human_processed = get_data_stat(human)
    gpt4_processed = get_data_stat(gpt4)

    human_all = get_feature_vec(human_processed)
    gpt4_all = get_feature_vec(gpt4_processed)

    h_train, h_test, yh_train, yh_test = train_test_split(human_all, np.zeros(human_all.shape[0]), test_size=0.2, random_state=42)
    g4_train, g4_test, yg4_train, yg4_test = train_test_split(gpt4_all, np.ones(gpt4_all.shape[0]), test_size=0.2, random_state=42)

    X_train = np.concatenate((g4_train, h_train), axis=0)
    y_train = np.concatenate((yg4_train, yh_train), axis=0)
    X_test = np.concatenate((g4_test, h_test), axis=0)
    y_test = np.concatenate((yg4_test, yh_test), axis=0)

    #clf = LogisticRegression(max_iter=32000) #65.6
    #clf.fit(X_train, y_train)
    #y_pred = clf.predict(X_test)

    # # Neural network
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, activation='relu', solver='adam', random_state=42) # 80.5
    #clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, activation='relu', solver='adam', random_state=42) # 80.2, using fuzzywazzy, get 78.5% acc.
    #clf = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=1000, activation='relu', solver='adam', random_state=42) # 80.8
    #clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)  # 72.5
    #clf = RandomForestClassifier(n_estimators=100, random_state=42) # 81.8
    #clf = KNeighborsClassifier(n_neighbors=3) # 77.4
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, random_state=42) # 86.5
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred), "F1 score:", f1_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return clf

def load_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_predictions_to_file(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Predict if the text is human-written or generated
def predict_text(human_file_path, generated_file_path, model, output_file_path):
    start_time = time.time()

    human_data = load_json_file(human_file_path)
    generated_data = load_json_file(generated_file_path)
    
    y_true = []
    y_pred = []
    predictions = []

    with open(output_file_path, 'w') as file:
        json.dump({"predictions": []}, file, indent=4)

    for entry in human_data:
        text = entry["abs"]
        y_true.append(0)

        rewritten_data = {prompt: GPT_self_prompt(prompt, text) for prompt in prompt_list}
        rewritten_data['input'] = text
        processed_data = get_data_stat([rewritten_data])
        feature_vector = get_feature_vec(processed_data)

        if feature_vector.size > 0:
            prediction = model.predict(feature_vector)
            prediction = "Generated" if prediction[0] == 1 else "Human-written"
            #return "Generated" if prediction[0] == 1 else "Human-written"
        else:
            prediction = "Unable to predict due to insufficient features."
            #return "Unable to predict due to insufficient features."
        
        y_pred.append(prediction)

        prediction_entry = {
            "abs" : text,
            "title": entry["title"],
            "score": feature_vector.size,
            "prediction" : "Generated" if prediction[0] == 1 else "Not Generated",
            "source" : "Human"
        }

        predictions.append(prediction_entry)

        with open(output_file_path, 'r+') as file:
            data = json.load(file)
            data["predictions"].append(prediction_entry)
            file.seek(0)
            json.dump(data, file, indent=4)
    
    for entry in generated_data:
        text = entry["abs"]
        y_true.append(0)
        y_pred.append(prediction)

        rewritten_data = {prompt: GPT_self_prompt(prompt, text) for prompt in prompt_list}
        rewritten_data['input'] = text
        processed_data = get_data_stat([rewritten_data])
        feature_vector = get_feature_vec(processed_data)

        if feature_vector.size > 0:
            prediction = model.predict(feature_vector)
            #prediction = "Generated" if prediction[0] == 1 else "Human-written"
            #return "Generated" if prediction[0] == 1 else "Human-written"
        else:
            prediction = "Unable to predict due to insufficient features."
            #return "Unable to predict due to insufficient features."
        
        prediction_entry = {
            "abs" : text,
            "title": entry["title"],
            "score": feature_vector.size,
            "prediction" : "Generated" if prediction[0] == 1 else "Not Generated",
            "source" : "Generated"
        }

        predictions.append(prediction_entry)

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
    data["running_time"] = time.time() - start_time
    running_time = time.time() - start_time

    print(f"Running Time: {running_time:.2f} seconds")

    save_predictions_to_file(output_data, output_file_path)

    return precision, recall, f1


def main():

    parser = argparse.ArgumentParser(description="Predict if a text is human-written or generated.")
    parser.add_argument("--input_folder", type=str, help="The input text to analyze.")
    args = parser.parse_args()

    # Load datasets
    train_human_file_path = "/home/breton/Dev/Gen_Article/Gen_ai_v2/Gen_ai/architectures/RaidarLLMDetect-main/Arxiv/ada_rewrite_arxiv_human_inv.json"
    train_generated_file_path = "/home/breton/Dev/Gen_Article/Gen_ai_v2/Gen_ai/architectures/RaidarLLMDetect-main/Arxiv/ada_rewrite_arxiv_GPT_inv.json"

    with open(train_human_file_path, 'r') as file:
        train_human = json.load(file)

    with open(train_generated_file_path, 'r') as file:
        train_gpt4 = json.load(file)

    global cutoff_start, cutoff_end
    cutoff_start = 0
    cutoff_end = 6000000

    # Train the classifier
    model = xgboost_classifier(train_human, train_gpt4)

    test_human_file_path = (
        config['datasets']['your-dataset_hum']
        if config['datasets']['your-dataset_hum'] != "the_path_to_your_hum_dataset"
        else config['datasets']['default_hum']
    )

    test_generated_file_path = (
        config['datasets']['your-dataset_gen']
        if config['datasets']['your-dataset_gen'] != "the_path_to_your_gen_dataset"
        else config['datasets']['default_gen']
    )

    output_file_path = f'./results/Raidar/iteration_test.json'

    prediction = predict_text(test_human_file_path, test_generated_file_path, model, output_file_path)
    print(f"The input text : \n{args.input_text}\nis predicted to be: {prediction}")

if __name__ == "__main__":
    main()