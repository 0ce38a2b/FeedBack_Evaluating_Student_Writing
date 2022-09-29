import tokenizer_fix
import optuna
import numpy as np
import Datasets, Utils
import pandas as pd
from transformers import AutoTokenizer
import copy
import gc
import logging


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


def objective_overall(trial):
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
        # 'Lead': {
        #     'Lead':  trial.suggest_float(name="L_L", low=0.5, high=1.0),
        #     'Position':  trial.suggest_float(name="L_P", low=0.8, high=1.2),
        #     'Evidence':  trial.suggest_float(name="L_E", low=0.8, high=1.2),
        #     'Claim':  trial.suggest_float(name="L_C", low=0.8, high=1.2),
        #     'Concluding Statement': trial.suggest_float(name="L_CS", low=0.8, high=1.2),
        #     'Counterclaim':  trial.suggest_float(name="L_CT", low=0.5, high=1.0),
        #     'Rebuttal':  trial.suggest_float(name="L_R", low=0.5, high=1.0),
        # },
        # 'Position': {
        #     'Lead':  trial.suggest_float(name="P_L", low=0.5, high=1.0),
        #     'Position':  trial.suggest_float(name="P_P", low=0.5, high=1.0),
        #     'Evidence':  trial.suggest_float(name="P_E", low=0.8, high=1.2),
        #     'Claim':  trial.suggest_float(name="P_C", low=0.8, high=1.2),
        #     'Concluding Statement': trial.suggest_float(name="P_CS", low=0.8, high=1.2),
        #     'Counterclaim':  trial.suggest_float(name="P_CT", low=0.8, high=1.2),
        #     'Rebuttal':  trial.suggest_float(name="P_R", low=0.5, high=1.0),
        # },
        # 'Evidence': {
        #     'Lead':  trial.suggest_float(name="E_L", low=0.5, high=1.0),
        #     'Position':  trial.suggest_float(name="E_P", low=0.8, high=1.2),
        #     'Evidence':  trial.suggest_float(name="E_E", low=0.8, high=1.2),
        #     'Claim':  trial.suggest_float(name="E_C", low=0.8, high=1.2),
        #     'Concluding Statement': trial.suggest_float(name="E_CS", low=0.8, high=1.2),
        #     'Counterclaim':  trial.suggest_float(name="E_CT", low=0.8, high=1.2),
        #     'Rebuttal':  trial.suggest_float(name="E_R", low=0.8, high=1.2),
        # },
        'Claim': {
            'Lead':  trial.suggest_float(name="C_L", low=0.5, high=1.0),
            'Position':  trial.suggest_float(name="C_P", low=0.8, high=1.2),
            'Evidence':  trial.suggest_float(name="C_E", low=0.8, high=1.2),
            'Claim':  trial.suggest_float(name="C_C", low=0.8, high=1.2),
            'Concluding Statement': trial.suggest_float(name="C_CS", low=0.8, high=1.2),
            'Counterclaim':  trial.suggest_float(name="C_CT", low=0.8, high=1.2),
            'Rebuttal':  trial.suggest_float(name="C_R", low=0.5, high=1.0),
        },
        # 'Concluding Statement': {
        #     'Lead':  trial.suggest_float(name="CS_L", low=0.5, high=1.0),
        #     'Position':  trial.suggest_float(name="CS_P", low=0.8, high=1.2),
        #     'Evidence':  trial.suggest_float(name="CS_E", low=0.5, high=1.0),
        #     'Claim':  trial.suggest_float(name="CS_C", low=0.5, high=1.0),
        #     'Concluding Statement': trial.suggest_float(name="CS_CS", low=0.5, high=1.0),
        #     'Counterclaim':  trial.suggest_float(name="CS_CT", low=0.8, high=1.2),
        #     'Rebuttal':  trial.suggest_float(name="CS_R", low=0.5, high=1.0),
        # },
        # 'Counterclaim': {
        #     'Lead':  trial.suggest_float(name="CT_L", low=0.5, high=1.0),
        #     'Position':  trial.suggest_float(name="CT_P", low=0.8, high=1.1),
        #     'Evidence':  trial.suggest_float(name="CT_E", low=0.8, high=1.2),
        #     'Claim':  trial.suggest_float(name="CT_C", low=0.8, high=1.2),
        #     'Concluding Statement': trial.suggest_float(name="CT_CS", low=0.8, high=1.2),
        #     'Counterclaim':  trial.suggest_float(name="CT_CT", low=0.8, high=1.1),
        #     'Rebuttal':  trial.suggest_float(name="CT_R", low=0.8, high=1.2),
        # },
        # 'Rebuttal': {
        #     'Lead':  trial.suggest_float(name="R_L", low=0.5, high=1.0),
        #     'Position':  trial.suggest_float(name="R_P", low=0.8, high=1.1),
        #     'Evidence':  trial.suggest_float(name="R_E", low=0.8, high=1.2),
        #     'Claim':  trial.suggest_float(name="R_C", low=0.8, high=1.2),
        #     'Concluding Statement': trial.suggest_float(name="R_CS", low=0.8, high=1.2),
        #     'Counterclaim':  trial.suggest_float(name="R_CT", low=0.8, high=1.2),
        #     'Rebuttal':  trial.suggest_float(name="R_R", low=0.5, high=1.0),
        # },
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
        # logging.info(tmp)
        f1_list.append(f1)
    f1_mean = 0
    for f1 in f1_list:
        f1_mean += f1
    logging.info(f1_list)
    logging.info("f1: {:.4f}".format(f1_mean / len(test_list)))
    return f1_mean / len(test_list)

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_overall, n_trials=3000, timeout=100000000)