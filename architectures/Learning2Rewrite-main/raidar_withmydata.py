import json
import numpy as np
from fuzzywuzzy import fuzz
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score
import argparse

def tokenize_and_normalize(sentence):
    return [word.lower().strip() for word in sentence.split()]

def extract_ngrams(tokens, n):
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def common_elements(list1, list2):
    return set(list1) & set(list2)

def calculate_sentence_common(sentence1, sentence2):
    tokens1 = tokenize_and_normalize(sentence1)
    tokens2 = tokenize_and_normalize(sentence2)
    common_words = common_elements(tokens1, tokens2)
    number_common_hierarchy = [len(list(common_words))]
    for n in range(2, 5):
        ngrams1 = extract_ngrams(tokens1, n)
        ngrams2 = extract_ngrams(tokens2, n)
        common_ngrams = common_elements(ngrams1, ngrams2)
        number_common_hierarchy.append(len(list(common_ngrams)))
    return number_common_hierarchy

ngram_num = 4
cutoff_start = 0
cutoff_end = 6000000

def sum_for_list(a, b):
    return [aa + bb for aa, bb in zip(a, b)]

def get_data_stat(data_json):
    for idx in range(len(data_json)):
        each = data_json[idx]
        original = each['original']
        raw = tokenize_and_normalize(original)
        if len(raw) < cutoff_start or len(raw) > cutoff_end:
            continue
        statistic_res = {}
        ratio_fzwz = {}
        all_statistic_res = [0 for _ in range(ngram_num)]
        cnt = 0
        whole_combined = ''

        # Pour les textes raffinés par IA, on compare "original" et "refined"
        if 'refined' in each:
            whole_combined += (' ' + each['refined'])
            res = calculate_sentence_common(original, each['refined'])
            statistic_res['refined'] = res
            all_statistic_res = sum_for_list(all_statistic_res, res)
            ratio_fzwz['refined'] = [fuzz.ratio(original, each['refined']), fuzz.token_set_ratio(original, each['refined'])]
            cnt += 1

        each['fzwz_features'] = ratio_fzwz
        each['common_features'] = statistic_res
        each['avg_common_features'] = [a / cnt for a in all_statistic_res] if cnt > 0 else [0] * ngram_num
        each['common_features_ori_vs_allcombined'] = calculate_sentence_common(original, whole_combined)
    return data_json

def get_feature_vec(input_json):
    all_list = []
    for idx in range(len(input_json)):
        each = input_json[idx]
        try:
            raw = tokenize_and_normalize(each['original'])
            r_len = len(raw) * 1.0
        except:
            continue
        each_data_fea = []
        if r_len == 0:
            continue
        if len(raw) < cutoff_start or len(raw) > cutoff_end:
            continue
        each_data_fea = [ind_d / r_len for ind_d in each['avg_common_features']]
        for ek in each['common_features'].keys():
            each_data_fea.extend([ind_d / r_len for ind_d in each['common_features'][ek]])
        each_data_fea.extend([ind_d / r_len for ind_d in each['common_features_ori_vs_allcombined']])
        for ek in each['fzwz_features'].keys():
            each_data_fea.extend(each['fzwz_features'][ek])
        all_list.append(np.array(each_data_fea))
    if len(all_list) > 0:
        all_list = np.vstack(all_list)
    return all_list

def load_data(input_file, label):
    with open(input_file, 'r') as f:
        data = json.load(f)
    for entry in data:
        entry['label'] = label
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate AI-refined vs human-written texts.')
    parser.add_argument('--ai-input', type=str, required=True, help='Path to the AI-refined JSON file.')
    parser.add_argument('--human-input', type=str, required=True, help='Path to the human-written JSON file.')
    args = parser.parse_args()

    # Charger les données
    ai_data = load_data(args.ai_input, label=1)  # Label 1 pour les textes raffinés par IA
    human_data = load_data(args.human_input, label=0)  # Label 0 pour les textes écrits par des humains

    # Préparer les données
    ai_data = get_data_stat(ai_data)
    human_data = get_data_stat(human_data)

    ai_features = get_feature_vec(ai_data)
    human_features = get_feature_vec(human_data)

    X_train = np.concatenate((ai_features, human_features), axis=0)
    y_train = np.concatenate((np.ones(ai_features.shape[0]), np.zeros(human_features.shape[0])), axis=0)

    # Normaliser les données
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Entraîner un classifieur
    # clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, activation='relu', solver='adam', random_state=42)

    clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, activation='relu', solver='adam', random_state=42)
    # clf = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=1000, activation='relu', solver='adam', random_state=42)
    # clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    # clf = RandomForestClassifier(n_estimators=100, random_state=42)
    # clf = KNeighborsClassifier(n_neighbors=3)
    # clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, random_state=42)

    clf.fit(X_train, y_train)

    # Évaluer le modèle
    y_pred = clf.predict(X_train)
    y_prob = clf.predict_proba(X_train)

    f1 = f1_score(y_train, y_pred)
    AUROC = roc_auc_score(y_train, y_prob[:, 1])

    print("AUROC:", AUROC)
    print("F1 score:", f1)
    print(classification_report(y_train, y_pred))
