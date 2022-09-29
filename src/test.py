import tokenizer_fix
import optuna
import numpy as np
import Datasets, Utils
import pandas as pd
from transformers import AutoTokenizer
import copy
import gc
import logging
logging.getLogger().setLevel(logging.INFO)


class Config:
    train_path = "/users10/lyzhang/opt/tiger/feedback/data/train_fold10.csv"
    fold = 0
    fix_length = 1600
    num_labels = 15


test_list = [
    ([
        ("/users10/lyzhang/opt/tiger/feedback/output_5fold/4_folddeberta_xlarge.npy", "/users10/lyzhang/model/deberta_xlarge"),
    ], "/users10/lyzhang/opt/tiger/feedback/data/train_fold5.csv", 4),
    ([
        ("/users10/lyzhang/opt/tiger/feedback/output_5fold/3_folddeberta_xlarge.npy", "/users10/lyzhang/model/deberta_xlarge"),
    ], "/users10/lyzhang/opt/tiger/feedback/data/train_fold5.csv", 3),
    ([
        ("/users10/lyzhang/opt/tiger/feedback/output_5fold/2_folddeberta_xlarge.npy", "/users10/lyzhang/model/deberta_xlarge"),
    ], "/users10/lyzhang/opt/tiger/feedback/data/train_fold5.csv", 2),
    ([
        ("/users10/lyzhang/opt/tiger/feedback/output_5fold/1_folddeberta_xlarge.npy", "/users10/lyzhang/model/deberta_xlarge"),
    ], "/users10/lyzhang/opt/tiger/feedback/data/train_fold5.csv", 1),
]
results_list = []
df_text_list = []
for item in test_list:
    model_list = item[0]
    train_path = item[1]
    fold = item[2]
    num_net = len(model_list)
    df = pd.read_csv(train_path)
    valid_df = df[df["fold"]==fold]
    valid_id = valid_df["id"].unique()
    num_valid = len(valid_id)
    df_text = []
    for id in valid_id:
        text_file = "/users10/lyzhang/opt/tiger/feedback/data/train" + f"/{id}.txt"
        with open(text_file, "r") as f:
            text = f.read()
        df_text.append((id, text))
    df_text = pd.DataFrame(df_text, columns=["id", "text"])
    df_text_list.append(df_text)
    results = []
    for model in model_list:
        if "deberta_v" in model[1]:
            tokenizer = tokenizer_fix.get_deberta_tokenizer(model[1])
        else:
            tokenizer = AutoTokenizer.from_pretrained(model[1])
        valid_samples = Utils.prepare_training_data(valid_df, tokenizer, 1)
        valid_datasets = Datasets.FeedbackDatasetValid(valid_samples, 1600, tokenizer)
        collate = Datasets.Collate(tokenizer, fix_length=1600)
        pred = np.load(model[0])
        results.append({
            "probability": pred,
            "token_offset": [copy.deepcopy(sample["offset_mapping"]) for sample in valid_samples]
        })
        del valid_samples, valid_datasets
        del tokenizer, collate
        gc.collect()
    results_list.append(results)


def objective_overall():
    args = Config()
    # min_thresh = {
    #     "Lead": trial.suggest_int(name="min_Lead", low=6, high=12),
    #     "Position": trial.suggest_int(name="min_Position", low=3, high=8),
    #     "Evidence": trial.suggest_int(name="min_Evidence", low=8, high=17),
    #     "Claim": trial.suggest_int(name="min_Claim", low=1, high=6),
    #     "Concluding Statement": trial.suggest_int(name="min_Concluding", low=5, high=14),
    #     "Counterclaim": trial.suggest_int(name="min_Counterclaim", low=3, high=9),
    #     "Rebuttal": trial.suggest_int(name="min_Rebuttal", low=2, high=7),
    # }
    min_thresh = {
        "Lead": 8, # 9
        "Position": 4, # 5
        "Evidence": 8, # 14
        "Claim": 1, # 3
        "Concluding Statement": 10, # 11
        "Counterclaim": 9,
        "Rebuttal": 2, # 4
    }
    proba_thresh = {
        "Lead": 0.5647964444385352,
        "Position": 0.6211823905704472,
        "Evidence": 0.6037415312070282,
        "Claim": 0.5655015619409717,
        "Concluding Statement": 0.5605916604200145,
        "Counterclaim": 0.5589241228663976,
        "Rebuttal": 0.6279143972926252
    }
    magic_params = [1.0151040677346772, 0.9030902168873051, 1.150794726450591, 1.1071525118845413, 0.8923895084086448, 0.9156323986648087, 0.8376600826469922, 1.0863350013753394, 0.8034240506162517, 0.8838929237772057, 1.167908517815809, 1.0758516272886205, 0.921439773646722, 1.175718027875091, 1.073194555851536]
    magic_params = np.array(magic_params)
    # min_thresh = {
    #     "Lead": 9,
    #     "Position": 5,
    #     "Evidence": 14,
    #     "Claim": 3,
    #     "Concluding Statement": 11,
    #     "Counterclaim": 6,
    #     "Rebuttal": 4,
    # }
    # proba_thresh = {
    #     "Lead": trial.suggest_float(name="proba_Lead", low=0.55, high=0.7),
    #     "Position": trial.suggest_float(name="proba_Position", low=0.5, high=0.7),
    #     "Evidence": trial.suggest_float(name="proba_Evidence", low=0.5, high=0.7),
    #     "Claim": trial.suggest_float(name="proba_Claim", low=0.45, high=0.7),
    #     "Concluding Statement": trial.suggest_float(name="proba_Concluding", low=0.55, high=0.7),
    #     "Counterclaim": trial.suggest_float(name="proba_Counterclaim", low=0.4, high=0.65),
    #     "Rebuttal": trial.suggest_float(name="proba_Rebuttal", low=0.4, high=0.65),
    # }
    # proba_thresh = {
    #     "Lead": 0.5609530780576544,
    #     "Position": 0.5159084828612942,
    #     "Evidence": 0.6022643606659289,
    #     "Claim": 0.5225115220999192,
    #     "Concluding Statement": 0.6462790454790648,
    #     "Counterclaim": 0.5059685465982959,
    #     "Rebuttal": 0.5333730782903967
    # }
    # weight = [
    #     trial.suggest_float(name="B-Lead", low=0.8, high=1.2),
    #     trial.suggest_float(name="I_Lead", low=0.8, high=1.2),
    #     trial.suggest_float(name="B_Position", low=0.8, high=1.2),
    #     trial.suggest_float(name="I_Position", low=0.8, high=1.2),
    #     trial.suggest_float(name="B_Evidence", low=0.8, high=1.2),
    #     trial.suggest_float(name="I_Evidence", low=0.8, high=1.2),
    #     trial.suggest_float(name="B_Claim", low=0.8, high=1.2),
    #     trial.suggest_float(name="I_Claim", low=0.8, high=1.2),
    #     trial.suggest_float(name="B_Concluding", low=0.8, high=1.2),
    #     trial.suggest_float(name="I_Concluding", low=0.8, high=1.2),
    #     trial.suggest_float(name="B_Counterclaim", low=0.8, high=1.2),
    #     trial.suggest_float(name="I_Counterclaim", low=0.8, high=1.2),
    #     trial.suggest_float(name="B_Rebuttal", low=0.8, high=1.2),
    #     trial.suggest_float(name="I_Rebuttal", low=0.8, high=1.2),
    #     trial.suggest_float(name="O", low=0.8, high=1.2),
    # ]
    # weight = [1] * 15
    # logging.info(weight)
    # weight = np.array(weight)
    # baseline: 7115
    test_list = [
        ([
            ("/users10/lyzhang/opt/tiger/feedback/output_5fold/4_folddeberta_xlarge.npy", "/users10/lyzhang/model/deberta_xlarge"),
        ], "/users10/lyzhang/opt/tiger/feedback/data/train_fold5.csv", 4),
        ([
            ("/users10/lyzhang/opt/tiger/feedback/output_5fold/3_folddeberta_xlarge.npy", "/users10/lyzhang/model/deberta_xlarge"),
        ], "/users10/lyzhang/opt/tiger/feedback/data/train_fold5.csv", 3),
        ([
            ("/users10/lyzhang/opt/tiger/feedback/output_5fold/2_folddeberta_xlarge.npy", "/users10/lyzhang/model/deberta_xlarge"),
        ], "/users10/lyzhang/opt/tiger/feedback/data/train_fold5.csv", 2),
        ([
            ("/users10/lyzhang/opt/tiger/feedback/output_5fold/1_folddeberta_xlarge.npy", "/users10/lyzhang/model/deberta_xlarge"),
        ], "/users10/lyzhang/opt/tiger/feedback/data/train_fold5.csv", 1),
    ]
    convert_table = {
        'Concluding Statement': {'Lead': 0.6180546139059481, 'Position': 1.106376532299193, 'Evidence': 0.8308079139611104, 'Claim': 0.9263323897452017, 'Concluding Statement': 0.7427996729640948, 'Counterclaim': 1.1255410558775625, 'Rebuttal': 0.9443065711964579},
        # 'Evidence': {'Lead': 0.6935170533407621, 'Position': 1.1604061898464963, 'Evidence': 1.053929298774972, 'Claim': 1.0154215894307475, 'Concluding Statement': 1.101286190449023, 'Counterclaim': 0.9458791358573785, 'Rebuttal': 1.0100877841954223},
        'Evidence': {'Lead': 0.7393941341170962, 'Position': 1.0724463092198406, 'Evidence': 1.0058653210627613, 'Claim': 1.022313944728398, 'Concluding Statement': 1.052367501549448, 'Counterclaim': 1.0090474774696865, 'Rebuttal': 1.0333100430656466},
        'Rebuttal': {'Lead': 0.5399304201316384, 'Position': 1.08856196786258, 'Evidence': 1.0103291548166256, 'Claim': 1.0847765835179881, 'Concluding Statement': 0.9549480752091022, 'Counterclaim': 1.092666136606834, 'Rebuttal': 0.7110649743074405},
        'Counterclaim': {'Lead': 0.9050794118735646, 'Position': 0.9160850157706609, 'Evidence': 1.1418506967909068, 'Claim': 1.049570707730134, 'Concluding Statement': 1.1146750148380493, 'Counterclaim': 0.9746304751663061, 'Rebuttal': 1.1594205982055437},
        'Position': {'Lead': 0.8393628836469293, 'Position': 0.792198366631391, 'Evidence': 1.0359996942895344, 'Claim': 1.0163778318063577, 'Concluding Statement': 1.0834964446969553, 'Counterclaim': 1.1085107194393697, 'Rebuttal': 0.6460859598497333},
    }
    logging.info(convert_table)
    f1_list = []
    for idx, item in enumerate(test_list):
        model_list = item[0]
        train_path = item[1]
        fold = item[2]
        num_net = len(model_list)
        df = pd.read_csv(train_path)
        valid_df = df[df["fold"]==fold]
        valid_id = valid_df["id"].unique()
        num_valid = len(valid_id)
        df_text = df_text_list[idx]
        results = results_list[idx]
        submit_df = []
        for i in range(num_valid):
            d = df_text.iloc[i]
            id =  d.id
            text = d.text
            word, word_offset = Utils.text_to_word(text)
            token_to_text_probability = np.full((len(text),args.num_labels),0, np.float32)
            for j in range(num_net):
                p = results[j]["probability"][i][1:]
                for t, (start, end) in enumerate(results[j]["token_offset"][i]):
                    if t==args.fix_length-1: break
                    token_to_text_probability[start: end] += p[t]
            token_to_text_probability = token_to_text_probability / num_net

            text_to_word_probability = np.full((len(word),args.num_labels),0, np.float32)
            for t,(start,end) in enumerate(word_offset):
                text_to_word_probability[t]=token_to_text_probability[start:end].mean(0)
            # predict_df = Utils.word_probability_to_predict_df(text_to_word_probability, id)
            # predict_df = Utils.word_probability_to_prediction_string(text_to_word_probability, id, word)
            # predict_df = Utils.word_probability_to_prediction_string_v4(text_to_word_probability, id, word)
            text_to_word_probability = text_to_word_probability * magic_params
            predict_df = Utils.word_probability_to_prediction_string_v5(text_to_word_probability, id, word, convert_table)
            submit_df.append(predict_df)
        submit_df = pd.concat(submit_df).reset_index(drop=True)
        submit_df = Utils.do_threshold_test(submit_df, min_thresh, proba_thresh, use=['length', 'probability'])
        f1, tmp = Utils.score_feedback_comp(submit_df, valid_df, return_class_scores=True)
        logging.info(tmp)
        f1_list.append(f1)
    f1_mean = 0
    for f1 in f1_list:
        f1_mean += f1
    logging.info(f1_list)
    logging.info("f1: {:.4f}".format(f1_mean / len(test_list)))
    return f1_mean / len(test_list)


objective_overall()