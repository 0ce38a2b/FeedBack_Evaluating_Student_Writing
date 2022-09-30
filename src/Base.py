import tokenizer_fix
import os
import logging
import datetime
import torch
import Config, Datasets, Model, Trainer, Utils
from transformers import AutoTokenizer
import pandas as pd
import time
import gc
from tqdm import tqdm
import copy
import numpy as np


@Utils.timer
def feedbackBase(args):
    logging.debug(args.tokenizer_path)
    # if args.deberta:
    #     tokenizer = tokenizer_fix.get_deberta_tokenizer(args.tokenizer_path)
    # else:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    df = pd.read_csv(args.train_path)
    train_df = df[df["fold"]!=args.fold]
    valid_df = df[df["fold"]==args.fold]
    if args.da:
        df_da = pd.read_csv("/users10/hsheng/opt/tiger/feedback/data/train_da_fold_0.csv")
        if args.debug:
            df_da = df_da.sample(1000)
        train_da_samples = Utils.prepare_training_data(df_da, tokenizer, 1, dir="train_da")
    if args.debug:
        train_df = train_df.sample(1000)
        valid_df = valid_df.sample(1000)
    train_samples = Utils.prepare_training_data(train_df, tokenizer, 1, deberta=args.deberta)
    valid_samples = Utils.prepare_training_data(valid_df, tokenizer, 1, deberta=args.deberta)
    if args.da:
        train_samples.extend(train_da_samples)
    train_datasets = Datasets.FeedbackDatasetCollate(train_samples, args.fix_length, tokenizer)
    # train_datasets = Datasets.FeedbackDataset(train_samples, args.fix_length, tokenizer)
    valid_datasets = Datasets.FeedbackDatasetValid(valid_samples, args.fix_length, tokenizer)
    valid_datasets.valid_df = valid_df
    collate = Datasets.Collate(tokenizer)
    collate_train = Datasets.CollateTrain(tokenizer)
    train_iter = torch.utils.data.DataLoader(train_datasets, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_train)
    trainer = Trainer.BaseTrainer(args)
    trainer.set_training_size(len(train_iter))
    trainer.model_init()
    trainer.optimizer_init()
    logging.info(f"Train Size: {len(train_iter)}")
    for epoch in range(args.epoch):
        logging.info(f"Start Epoch: {epoch}")
        if epoch >= 8:
            continue
        t_s = time.time()
        loss = 0
        if args.debug:
            for batch in tqdm(train_iter):
                loss += trainer.step(batch)
        else:
            for batch in train_iter:
                loss += trainer.step(batch)
                if trainer.should_eval():
                    trainer.eval(valid_datasets, collate)
        logging.info("Train Loss: {:.4f}".format(loss / len(train_iter)))
        trainer.eval(valid_datasets, collate)
        t_e = time.time()
        logging.info("Cost {:.2f} s.".format(t_e - t_s))
    f1_maxn = trainer.f1_maxn
    logging.info("Best F1: {:.4f}".format(f1_maxn))
    del train_df, valid_df
    del train_samples, valid_samples
    del train_datasets, valid_datasets
    del train_iter, trainer
    gc.collect()
    return f1_maxn


@Utils.timer
def main(args):
    while True:
        model_save = "/".join([args.model_save, Utils.d2s(datetime.datetime.now(), time=True)])
        if not args.debug:
            if os.path.exists(model_save):
                logging.warning("save path exists, sleep 60s")
                time.sleep(60)
            else:
                os.mkdir(model_save)
                args.model_save = model_save
                break
        else:
            break
    MODEL_PREFIX = args.model_save
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info(f"device: {args.device}")
    if args.train_all:
        num = args.fold
        for fold in range(num):
            args.fold = fold
            args.model_save = "/".join([MODEL_PREFIX, f"Fold_{args.fold}.bin"])
            logging.info(f"model save path: {args.model_save}")
            feedbackBase(args)
    else:
        args.model_save = "/".join([MODEL_PREFIX, f"Fold_{args.fold}.bin"])
        logging.info(f"model save path: {args.model_save}")
        feedbackBase(args)


@Utils.timer
def predict(args):
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_list = [
        # ("/users10/hsheng/opt/tiger/feedback/model/LF/2022_02_27_12_46/Fold_0.bin", "/users10/hsheng/model/funnel_large"),    # 0.6742
        # ("/users10/hsheng/opt/tiger/feedback/model/LF/2022_03_09_22_58/Fold_0.bin", "/users10/hsheng/model/deberta_xlarge"), # 0.7092
        # ("/users10/hsheng/opt/tiger/feedback/model/LF/2022_02_28_14_00/Fold_0.bin", "/users10/hsheng/model/funnel_large"),    # 0.6794
        # # ("/users10/hsheng/opt/tiger/feedback/model/LF/2022_02_27_13_49/Fold_0.bin", "/users10/hsheng/model/deberta_v3_large")
        # ("/users10/hsheng/opt/tiger/feedback/model/LF/2022_02_28_22_27/Fold_0.bin", "/users10/hsheng/model/deberta_v3_large"), # 0.6842
        # ("/users10/hsheng/opt/tiger/feedback/model/LF/2022_03_04_18_39/Fold_0.bin", "/users10/hsheng/model/bigbird_roberta_large"), # 0.6549
        # ("/users10/hsheng/opt/tiger/feedback/model/LF/2022_03_05_00_00/Fold_0.bin", "/users10/hsheng/model/deberta_large"), # 0.7012
        # ("/users10/hsheng/opt/tiger/feedback/model/LF/2022_03_03_21_33/Fold_0.bin", "/users10/hsheng/model/longformer_large_4096"), # 0.6748
        # ("/users10/hsheng/opt/tiger/feedback/model/LF/2022_02_28_22_30/Fold_1.bin", "/users10/hsheng/model/funnel_large"),
        # ("/users10/hsheng/opt/tiger/feedback/model/LF/2022_03_01_18_57/Fold_1.bin", "/users10/hsheng/model/deberta_v3_large"),
        # ("/users10/hsheng/opt/tiger/feedback/model/LF/2022_03_10_23_46/Fold_2.bin", "/users10/hsheng/model/deberta_xlarge"),
        # ("/users10/hsheng/opt/tiger/feedback/model/LF/2022_03_07_23_08/Fold_2.bin", "/users10/hsheng/model/deberta_large"),

        # ("/users10/hsheng/opt/tiger/feedback/model/LF/2022_03_09_22_12/Fold_3.bin", "/users10/hsheng/model/funnel_large"),
        # ("/users10/hsheng/opt/tiger/feedback/model/LF/2022_03_09_22_11/Fold_3.bin", "/users10/hsheng/model/longformer_large_4096"),
        # ("/users10/hsheng/opt/tiger/feedback/model/LF/2022_03_07_12_35/Fold_3.bin", "/users10/hsheng/model/deberta_xlarge"),
        
        # ("/users10/hsheng/opt/tiger/feedback/model/LF/2022_03_09_20_48/Fold_5.bin", "/users10/hsheng/model/longformer_large_4096"),
        # ("/users10/hsheng/opt/tiger/feedback/model/LF/2022_03_08_22_12/Fold_5.bin", "/users10/hsheng/model/deberta_v3_large"),
        
        # ("/users10/hsheng/opt/tiger/feedback/model/LF/2022_03_08_05_58/Fold_6.bin", "/users10/hsheng/model/deberta_large"),


    ]
    model_list = [
        # ("/users10/hsheng/opt/tiger/feedback/model/LF/2022_02_27_12_46/Fold_0.bin", "/users10/hsheng/model/funnel_large"),    # 0.6742
        ("/users10/hsheng/opt/tiger/feedback/model/LF/2022_03_11_14_54/Fold_4.bin", "/users10/hsheng/model/deberta_xlarge"),
        ("/users10/hsheng/opt/tiger/feedback/model/LF/2022_03_11_14_53/Fold_3.bin", "/users10/hsheng/model/deberta_xlarge"),
        ("/users10/hsheng/opt/tiger/feedback/model/LF/2022_03_11_14_52/Fold_2.bin", "/users10/hsheng/model/deberta_xlarge"),
        ("/users10/hsheng/opt/tiger/feedback/model/LF/2022_03_11_14_48/Fold_1.bin", "/users10/hsheng/model/deberta_xlarge"),
    ]
    num_net = len(model_list)
    df = pd.read_csv(args.train_path)
    valid_df = df[df["fold"]==args.fold]
    if args.debug:
        valid_df = valid_df.sample(8)
    valid_id = valid_df["id"].unique()
    num_valid = len(valid_id)
    logging.info(f"num_valid: {num_valid}")
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
        valid_datasets = Datasets.FeedbackDatasetValid(valid_samples, args.fix_length, tokenizer)
        # valid_datasets = Datasets.FeedbackMaxTokenValid(valid_datasets, 12288, args.fix_length)
        # if "longformer_large_4096" in model[1]:
        #     collate = Datasets.Collate(tokenizer, fix_length=args.fix_length, fixed=True)
        # else:
        collate = Datasets.Collate(tokenizer, fix_length=args.fix_length)
        predicter = Trainer.Predicter(args)
        predicter.set_pretrain(model[1])
        predicter.model_init()
        predicter.model_load(model[0])
        # if "bigbird" in model[1]:
        #     torch.backends.cudnn.enabled = False
        # else:
        #     torch.backends.cudnn.enabled = True
        pred = predicter.predict(valid_datasets, collate)
        # pred = predicter.predict_max_tokens(valid_datasets)
        pred = np.concatenate(pred)
        np.save("/users10/hsheng/opt/tiger/feedback/output_5fold/{}_fold{}.npy".format(model[0].split(".")[0][-1], model[1].split("/")[-1]), pred)
        results.append({
            "probability": pred,
            "token_offset": [copy.deepcopy(sample["offset_mapping"]) for sample in valid_samples]
        })
        del valid_samples, valid_datasets
        del tokenizer, collate, predicter
        gc.collect()
    submit_df = []
    for i in tqdm(range(num_valid)):
        d = df_text.iloc[i]
        id =  d.id
        text = d.text
        word, word_offset = Utils.text_to_word(text)
        token_to_text_probability = np.full((len(text),args.num_labels),0, np.float32)
        for j in range(num_net):
            p = results[j]["probability"][i][1:]
            # logging.info(p.shape)
            for t, (start, end) in enumerate(results[j]["token_offset"][i]):
                if t==args.fix_length-1: break
                token_to_text_probability[start: end] += p[t]
        token_to_text_probability = token_to_text_probability / num_net

        text_to_word_probability = np.full((len(word),args.num_labels),0, np.float32)
        for t,(start,end) in enumerate(word_offset):
            text_to_word_probability[t]=token_to_text_probability[start:end].mean(0)
        predict_df = Utils.word_probability_to_predict_df(text_to_word_probability, id)
        submit_df.append(predict_df)
    submit_df = pd.concat(submit_df).reset_index(drop=True)
    submit_df.to_csv("/users10/hsheng/opt/tiger/feedback/output/submit.csv", index=False)
    submit_df = Utils.do_threshold(submit_df, use=['length', 'probability'])
    # submit_df = Utils.do_threshold(submit_df, use=['length'])
    submit_df.to_csv("/users10/hsheng/opt/tiger/feedback/output/submit_drop.csv", index=False)
    logging.info(submit_df.head())
    logging.info(valid_df.head())
    f1, tmp = Utils.score_feedback_comp(submit_df, valid_df, return_class_scores=True)
    # f1 = Utils.score_feedback_comp_micro
    logging.info("f1: {:.4f}".format(f1))
    logging.info(tmp)


if __name__ == "__main__":
    args = Config.BaseConfig()
    if not args.debug:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.DEBUG)
    Utils.set_seed(args.seed)
    if args.train:
        logging.info(f"args: {args}".replace(" ", "\n"))
        main(args)
    elif args.predict:
        predict(args)
