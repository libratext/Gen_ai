import json
import numpy as np
from fuzzywuzzy import fuzz

# import xgboost as xgb
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score


def tokenize_and_normalize(sentence):
    # Tokenization and normalization
    return [word.lower().strip() for word in sentence.split()]

def extract_ngrams(tokens, n):
    # Extract n-grams from the list of tokens
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def common_elements(list1, list2):
    # Find common elements between two lists
    return set(list1) & set(list2)

def calculate_sentence_common(sentence1, sentence2):
    tokens1 = tokenize_and_normalize(sentence1)
    tokens2 = tokenize_and_normalize(sentence2)

    # Find common words
    common_words = common_elements(tokens1, tokens2)

    # Find common n-grams (let's say up to 3-grams for this example)
    common_ngrams = set()
    

    number_common_hierarchy = [len(list(common_words))]

    for n in range(2, 5):  # 2-grams to 3-grams
        ngrams1 = extract_ngrams(tokens1, n)
        ngrams2 = extract_ngrams(tokens2, n)
        common_ngrams = common_elements(ngrams1, ngrams2) 
        number_common_hierarchy.append(len(list(common_ngrams)))

    return number_common_hierarchy

ngram_num = 4
cutoff_start = 0
cutoff_end = 6000000
def sum_for_list(a,b):
    return [aa+bb for aa, bb in zip(a,b)]

def get_data_stat(data_json):
    total_len = len(data_json)
    for idxx in range(len(data_json)):
        
        each = data_json[idxx]
        original = each['input']

        # remove too short ones
        
        # import pdb; pdb.set_trace()
        raw = tokenize_and_normalize(each['input'])
        if len(raw)<cutoff_start or len(raw)>cutoff_end:
            continue
        # else:
        #     print(idxx, total_len)

        statistic_res = {}
        ratio_fzwz = {}
        all_statistic_res = [0 for i in range(ngram_num)]
        cnt = 0
        whole_combined=''
        for pp in each.keys():
            if pp != 'common_features':
                whole_combined += (' ' + each[pp])
                

                res = calculate_sentence_common(original, each[pp])
                statistic_res[pp] = res
                all_statistic_res = sum_for_list(all_statistic_res, res)

                ratio_fzwz[pp] = [fuzz.ratio(original, each[pp]), fuzz.token_set_ratio(original, each[pp])]
                cnt += 1
        
        each['fzwz_features'] = ratio_fzwz
        each['common_features'] = statistic_res
        each['avg_common_features'] = [a/cnt for a in all_statistic_res]

        each['common_features_ori_vs_allcombined'] = calculate_sentence_common(original, whole_combined)

    return data_json

def get_feature_vec(input_json):
    all_list = []
    for idxx in range(len(input_json)):

        each = input_json[idxx]
        try:
            raw = tokenize_and_normalize(each['input'])
            r_len = len(raw)*1.0
        except:
            import pdb; pdb.set_trace()
        each_data_fea  = []

        if r_len ==0:
            continue
        if len(raw)<cutoff_start or len(raw)>cutoff_end:
            continue

        # each_data_fea  = [len(raw) / 100.]

        each_data_fea = [ind_d / r_len for ind_d in each['avg_common_features']]
        for ek in each['common_features'].keys():
            each_data_fea.extend([ind_d / r_len for ind_d in each['common_features'][ek]])

        each_data_fea.extend([ind_d / r_len for ind_d in each['common_features_ori_vs_allcombined']])

        for ek in each['fzwz_features'].keys():
            each_data_fea.extend(each['fzwz_features'][ek])

        all_list.append(np.array(each_data_fea))

    all_list = np.vstack(all_list)

    return all_list


domains = ["AcademicResearch", "Code", "Entertainment", "GovernmentPublic", "NewsArticle", "Religious", "ArtCulture", "Environmental", "LegalDocument", "OnlineContent", "Sports", "Business", "Finance", "LiteratureCreativeWriting", "PersonalCommunication", "TechnicalWriting", "EducationMaterial", "FoodCusine", "MedicalText", "ProductReview", "TravelTourism"]
data_gpt_davinci, data_human, idx_gpt, idx_human = {}, {}, 0, 0

for domain in domains:
    with open(f"../dataset/{domain}/AI_rewrite_train.json", 'r') as f:
        d = json.load(f)
    for i in range(len(d)):
        data_gpt_davinci[idx_gpt+i] = d[i]
    idx_gpt += len(d)
    
    with open(f"../dataset/{domain}/human_rewrite_train.json", 'r') as f:
        d = json.load(f)
    for i in range(len(d)):
        data_human[idx_human+i] = d[i]
    idx_human += len(d)

print(len(data_gpt_davinci), len(data_human))

gpt_davinci = get_data_stat(data_gpt_davinci)
human = get_data_stat(data_human)

gpt_davinci_all = get_feature_vec(gpt_davinci)
human_all = get_feature_vec(human)

X_train = np.concatenate((gpt_davinci_all, human_all), axis=0)
y_train = np.concatenate((np.ones(gpt_davinci_all.shape[0]), np.zeros(human_all.shape[0])), axis=0)

# Neural network
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, activation='relu', solver='adam', random_state=42)
clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, activation='relu', solver='adam', random_state=42)
# clf = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=1000, activation='relu', solver='adam', random_state=42)
# clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf = KNeighborsClassifier(n_neighbors=3)
# clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, random_state=42)
clf.fit(X_train, y_train)

AUROC_total, f1_total = [], []

# domains = ["M4"]

for domain in domains:
    print(domain)
    
    with open(f"../dataset/{domain}/AI_rewrite_test.json", 'r') as f:
        data_gpt_davinci = json.load(f)
    
    with open(f"../dataset/{domain}/human_rewrite_test.json", 'r') as f:
        data_human = json.load(f)
    
    # print(len(data_gpt_davinci), len(data_human))
    
    gpt_davinci = get_data_stat(data_gpt_davinci)
    human = get_data_stat(data_human)
    
    gpt_davinci_all = get_feature_vec(gpt_davinci)
    human_all = get_feature_vec(human)
    
    X_test = np.concatenate((gpt_davinci_all, human_all), axis=0)
    y_test = np.concatenate((np.ones(gpt_davinci_all.shape[0]), np.zeros(human_all.shape[0])), axis=0)
    
    X_test = scaler.transform(X_test)
    
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    
    f1 = f1_score(y_test, y_pred)
    AUROC = roc_auc_score(y_test, y_prob[:, 1])
    
    f1_total.append(f1)
    AUROC_total.append(AUROC)
    
    print("AUROC:", AUROC, "F1 score", f1)
    print(classification_report(y_test, y_pred))
    
print()
print("Average AUROC:", np.average(AUROC_total), "Average F1 Score:", np.average(f1_total))