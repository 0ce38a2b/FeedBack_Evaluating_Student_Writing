import tokenizer_fix
import numpy as np
import Datasets, Utils
import pandas as pd
from transformers import AutoTokenizer
import copy
import gc
import logging


class Config:
    train_path = "/users10/hsheng/opt/tiger/feedback/data/train_fold10.csv"
    fold = 0
    fix_length = 1600
    num_labels = 15


def test():
    args = Config()
    model_list = [
        ("/users10/hsheng/chye/output/funnel_large.npy", "/users10/hsheng/model/funnel_large"),    # 0.6794
        ("/users10/hsheng/chye/output/deberta_v3_large.npy", "/users10/hsheng/model/deberta_v3_large"), # 0.6842
        ("/users10/hsheng/chye/output/bigbird_roberta_large.npy", "/users10/hsheng/model/bigbird_roberta_large"), # 0.6549
        ("/users10/hsheng/chye/output/deberta_large.npy", "/users10/hsheng/model/deberta_large"), # 0.7012
        ("/users10/hsheng/chye/output/longformer_large_4096.npy", "/users10/hsheng/model/longformer_large_4096"), # 0.6748
    ]
    num_net = len(model_list)
    df = pd.read_csv(args.train_path)
    valid_df = df[df["fold"]==args.fold]
    valid_id = valid_df["id"].unique()
    num_valid = len(valid_id)
    df_text = []
    for id in valid_id:
        text_file = "/users10/hsheng/opt/tiger/feedback/data/train" + f"/{id}.txt"
        with open(text_file, "r") as f:
            text = f.read()
        df_text.append((id, text))
    df_text = pd.DataFrame(df_text, columns=["id", "text"])
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
    submit_df = []
    weights = [
        18,
        62,
        0,
        97,
        35,
    ]
    sum_weight = 0
    for weight in weights:
        sum_weight += weight
    logging.info(f"weights: {weights}")
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
                token_to_text_probability[start: end] += p[t] * weights[j]
        token_to_text_probability = token_to_text_probability / sum_weight

        text_to_word_probability = np.full((len(word),args.num_labels),0, np.float32)
        for t,(start,end) in enumerate(word_offset):
            text_to_word_probability[t]=token_to_text_probability[start:end].mean(0)
        predict_df = Utils.word_probability_to_predict_df(text_to_word_probability, id)
        submit_df.append(predict_df)
    submit_df = pd.concat(submit_df).reset_index(drop=True)
    submit_df = Utils.do_threshold(submit_df, use=['length', 'probability'])
    f1, tmp = Utils.score_feedback_comp(submit_df, valid_df, return_class_scores=True)
    # f1 = Utils.score_feedback_comp_micro
    logging.info("f1: {:.4f}".format(f1))
    logging.info(tmp)
    return f1



if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    test()