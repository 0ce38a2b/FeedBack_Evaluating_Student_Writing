import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
import math
import random
import numpy as np
import time
import logging
import os
import copy
from joblib import Parallel, delayed
import pandas as pd
from torch import Tensor
from typing import List, Optional
from nltk.tokenize import sent_tokenize


target_id_map = {
    "B-Lead": 0,
    "I-Lead": 1,
    "B-Position": 2,
    "I-Position": 3,
    "B-Evidence": 4,
    "I-Evidence": 5,
    "B-Claim": 6,
    "I-Claim": 7,
    "B-Concluding Statement": 8,
    "I-Concluding Statement": 9,
    "B-Counterclaim": 10,
    "I-Counterclaim": 11,
    "B-Rebuttal": 12,
    "I-Rebuttal": 13,
    "O": 14,
    "PAD": -100,
}
id_target_map = {v: k for k, v in target_id_map.items()}


class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(
            math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples: (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples


def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def d2s(dt, time=False):
    if time is False:
        return dt.strftime("%Y_%m_%d")
    else:
        return dt.strftime("%Y_%m_%d_%H_%M")


def timer(func):
    def deco(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        logging.info("Function {} run {:.2f}s.".format(func.__name__, end_time - start_time))
        return res

    return deco


def sentence_token_nltk(str):
    sent_tokenize_list = sent_tokenize(str)
    return sent_tokenize_list


def _prepare_training_data_helper(tokenizer, df, train_ids, dir="train"):
    training_samples = []
    for idx in train_ids:
        filename = os.path.join("/users10/hsheng/opt/tiger/feedback/data", dir, idx + ".txt")
        with open(filename, "r") as f:
            text = f.read()

        encoded_text = tokenizer.encode_plus(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        input_ids = encoded_text["input_ids"]
        input_labels = copy.deepcopy(input_ids)
        offset_mapping = encoded_text["offset_mapping"]

        for k in range(len(input_labels)):
            input_labels[k] = "O"

        sample = {
            "id": idx,
            "input_ids": input_ids,
            "text": text,
            "offset_mapping": offset_mapping,
        }
        temp_df = df[df["id"] == idx]
        for _, row in temp_df.iterrows():
            text_labels = [0] * len(text)
            discourse_start = int(row["discourse_start"])
            discourse_end = int(row["discourse_end"])
            prediction_label = row["discourse_type"]
            text_labels[discourse_start:discourse_end] = [1] * (discourse_end - discourse_start)
            target_idx = []
            for map_idx, (offset1, offset2) in enumerate(encoded_text["offset_mapping"]):
                if sum(text_labels[offset1:offset2]) > 0:
                    if len(text[offset1:offset2].split()) > 0:
                        target_idx.append(map_idx)

            targets_start = target_idx[0]
            targets_end = target_idx[-1]
            pred_start = "B-" + prediction_label
            pred_end = "I-" + prediction_label
            input_labels[targets_start] = pred_start
            input_labels[targets_start + 1: targets_end + 1] = [pred_end] * (targets_end - targets_start)

        sample["input_ids"] = input_ids
        sample["input_labels"] = input_labels
        training_samples.append(sample)
    return training_samples


def _prepare_deberta_training_data_helper(tokenizer, df, train_ids, dir="train"):
    training_samples = []
    for idx in train_ids:
        filename = os.path.join("/users10/hsheng/opt/tiger/feedback/data", dir, idx + ".txt")
        with open(filename, "r") as f:
            text = f.read()

        tokens = tokenizer.tokenize(text)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        offset_mapping = []
        l = 0
        for token in tokens:
            if len(token) == 1:
                word = token
            else:
                word = token[1:]
            pos = text.find(word, l)
            offset_mapping.append((pos, pos + len(word)))
            l = pos + len(word)
        input_labels = copy.deepcopy(input_ids)

        for k in range(len(input_labels)):
            input_labels[k] = "O"

        sample = {
            "id": idx,
            "input_ids": input_ids,
            "text": text,
            "offset_mapping": offset_mapping,
        }
        temp_df = df[df["id"] == idx]
        for _, row in temp_df.iterrows():
            text_labels = [0] * len(text)
            discourse_start = int(row["discourse_start"])
            discourse_end = int(row["discourse_end"])
            prediction_label = row["discourse_type"]
            text_labels[discourse_start:discourse_end] = [1] * (discourse_end - discourse_start)
            target_idx = []
            for map_idx, (offset1, offset2) in enumerate(offset_mapping):
                if sum(text_labels[offset1:offset2]) > 0:
                    if len(text[offset1:offset2].split()) > 0:
                        target_idx.append(map_idx)

            targets_start = target_idx[0]
            targets_end = target_idx[-1]
            pred_start = "B-" + prediction_label
            pred_end = "I-" + prediction_label
            input_labels[targets_start] = pred_start
            input_labels[targets_start + 1: targets_end + 1] = [pred_end] * (targets_end - targets_start)

        sample["input_ids"] = input_ids
        sample["input_labels"] = input_labels
        training_samples.append(sample)
    return training_samples


@timer
def prepare_training_data(df, tokenizer, num_jobs, dir="train", deberta=False):
    training_samples = []
    train_ids = df["id"].unique()

    train_ids_splits = np.array_split(train_ids, num_jobs)

    if deberta:
        results = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
            delayed(_prepare_deberta_training_data_helper)(tokenizer, df, idx, dir) for idx in train_ids_splits
        )
    else:
        results = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
            delayed(_prepare_training_data_helper)(tokenizer, df, idx, dir) for idx in train_ids_splits
        )
    for result in results:
        training_samples.extend(result)

    return training_samples


def fetch_submission(preds_iter, valid_samples):
    final_preds = []
    final_scores = []
    for preds in preds_iter:
        # logging.debug(f"sub. preds: {preds}")
        pred_class = np.argmax(preds, axis=2)
        pred_scrs = np.max(preds, axis=2)
        # logging.debug(f"sub. pred_class: {pred_class}")
        # logging.debug(f"sub. pred_scrs: {pred_scrs}")
        for pred, pred_scr in zip(pred_class, pred_scrs):
            final_preds.append(pred.tolist())
            final_scores.append(pred_scr.tolist())

    for j in range(len(valid_samples)):
        tt = [id_target_map[p] for p in final_preds[j][1:]]
        tt_score = final_scores[j][1:]
        valid_samples[j]["preds"] = tt
        valid_samples[j]["pred_scores"] = tt_score

    submission = []
    min_thresh = {
        "Lead": 9,
        "Position": 5,
        "Evidence": 14,
        "Claim": 3,
        "Concluding Statement": 11,
        "Counterclaim": 6,
        "Rebuttal": 4,
    }
    proba_thresh = {
        "Lead": 0.7,
        "Position": 0.55,
        "Evidence": 0.65,
        "Claim": 0.55,
        "Concluding Statement": 0.7,
        "Counterclaim": 0.5,
        "Rebuttal": 0.55,
    }

    for _, sample in enumerate(valid_samples):
        preds = sample["preds"]
        offset_mapping = sample["offset_mapping"]
        sample_id = sample["id"]
        sample_text = sample["text"]
        sample_pred_scores = sample["pred_scores"]

        # pad preds to same length as offset_mapping
        if len(preds) < len(offset_mapping):
            preds = preds + ["O"] * (len(offset_mapping) - len(preds))
            sample_pred_scores = sample_pred_scores + [0] * (len(offset_mapping) - len(sample_pred_scores))

        idx = 0
        phrase_preds = []
        while idx < len(offset_mapping):
            start, _ = offset_mapping[idx]
            if preds[idx] != "O":
                label = preds[idx][2:]
            else:
                label = "O"
            phrase_scores = []
            phrase_scores.append(sample_pred_scores[idx])
            idx += 1
            while idx < len(offset_mapping):
                if label == "O":
                    matching_label = "O"
                else:
                    matching_label = f"I-{label}"
                if preds[idx] == matching_label:
                    _, end = offset_mapping[idx]
                    phrase_scores.append(sample_pred_scores[idx])
                    idx += 1
                else:
                    break
            if "end" in locals():
                phrase = sample_text[start:end]
                phrase_preds.append((phrase, start, end, label, phrase_scores))

        temp_df = []
        for phrase_idx, (phrase, start, end, label, phrase_scores) in enumerate(phrase_preds):
            word_start = len(sample_text[:start].split())
            word_end = word_start + len(sample_text[start:end].split())
            word_end = min(word_end, len(sample_text.split()))
            ps = " ".join([str(x) for x in range(word_start, word_end)])
            if label != "O":
                if sum(phrase_scores) / len(phrase_scores) >= proba_thresh[label]:
                    temp_df.append((sample_id, label, ps))

        temp_df = pd.DataFrame(temp_df, columns=["id", "class", "predictionstring"])

        submission.append(temp_df)

    submission = pd.concat(submission).reset_index(drop=True)
    submission["len"] = submission.predictionstring.apply(lambda x: len(x.split()))

    def threshold(df):
        df = df.copy()
        for key, value in min_thresh.items():
            index = df.loc[df["class"] == key].query(f"len<{value}").index
            df.drop(index, inplace=True)
        return df

    submission = threshold(submission)

    # drop len
    submission = submission.drop(columns=["len"])
    return submission


def calc_overlap(row):
    """
    Calculates the overlap between prediction and
    ground truth and overlap percentages used for determining
    true positives.
    """
    set_pred = set(row.predictionstring_pred.split(" "))
    set_gt = set(row.predictionstring_gt.split(" "))
    # Length of each and intersection
    len_gt = len(set_gt)
    len_pred = len(set_pred)
    inter = len(set_gt.intersection(set_pred))
    overlap_1 = inter / len_gt
    overlap_2 = inter / len_pred
    return [overlap_1, overlap_2]


def score_feedback_comp_micro(pred_df, gt_df):
    """
    A function that scores for the kaggle
        Student Writing Competition
    Uses the steps in the evaluation page here:
        https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    This code is from Rob Mulla's Kaggle kernel.
    """
    gt_df = gt_df[["id", "discourse_type", "predictionstring"]].reset_index(drop=True).copy()
    pred_df = pred_df[["id", "class", "predictionstring"]].reset_index(drop=True).copy()
    pred_df["pred_id"] = pred_df.index
    gt_df["gt_id"] = gt_df.index
    # Step 1. all ground truths and predictions for a given class are compared.
    joined = pred_df.merge(
        gt_df,
        left_on=["id", "class"],
        right_on=["id", "discourse_type"],
        how="outer",
        suffixes=("_pred", "_gt"),
    )
    joined["predictionstring_gt"] = joined["predictionstring_gt"].fillna(" ")
    joined["predictionstring_pred"] = joined["predictionstring_pred"].fillna(" ")

    joined["overlaps"] = joined.apply(calc_overlap, axis=1)

    # 2. If the overlap between the ground truth and prediction is >= 0.5,
    # and the overlap between the prediction and the ground truth >= 0.5,
    # the prediction is a match and considered a true positive.
    # If multiple matches exist, the match with the highest pair of overlaps is taken.
    joined["overlap1"] = joined["overlaps"].apply(lambda x: eval(str(x))[0])
    joined["overlap2"] = joined["overlaps"].apply(lambda x: eval(str(x))[1])

    joined["potential_TP"] = (joined["overlap1"] >= 0.5) & (joined["overlap2"] >= 0.5)
    joined["max_overlap"] = joined[["overlap1", "overlap2"]].max(axis=1)
    tp_pred_ids = (
        joined.query("potential_TP")
            .sort_values("max_overlap", ascending=False)
            .groupby(["id", "predictionstring_gt"])
            .first()["pred_id"]
            .values
    )

    # 3. Any unmatched ground truths are false negatives
    # and any unmatched predictions are false positives.
    fp_pred_ids = [p for p in joined["pred_id"].unique() if p not in tp_pred_ids]

    matched_gt_ids = joined.query("potential_TP")["gt_id"].unique()
    unmatched_gt_ids = [c for c in joined["gt_id"].unique() if c not in matched_gt_ids]

    # Get numbers of each type
    TP = len(tp_pred_ids)
    FP = len(fp_pred_ids)
    FN = len(unmatched_gt_ids)
    # calc microf1
    my_f1_score = TP / (TP + 0.5 * (FP + FN))
    return my_f1_score


def score_feedback_comp(pred_df, gt_df, return_class_scores=False):
    class_scores = {}
    pred_df = pred_df[["id", "class", "predictionstring"]].reset_index(drop=True).copy()
    for discourse_type, gt_subset in gt_df.groupby("discourse_type"):
        pred_subset = pred_df.loc[pred_df["class"] == discourse_type].reset_index(drop=True).copy()
        class_score = score_feedback_comp_micro(pred_subset, gt_subset)
        class_scores[discourse_type] = class_score
    f1 = np.mean([v for v in class_scores.values()])
    if return_class_scores:
        return f1, class_scores
    return f1


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.6, emb_name='embeddings.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embeddings.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(RAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('RAdam does not support sparse gradients')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            radam(params_with_grad,
                  grads,
                  exp_avgs,
                  exp_avg_sqs,
                  state_steps,
                  beta1=beta1,
                  beta2=beta2,
                  lr=group['lr'],
                  weight_decay=group['weight_decay'],
                  eps=group['eps'])
        return loss


def radam(params: List[Tensor],
          grads: List[Tensor],
          exp_avgs: List[Tensor],
          exp_avg_sqs: List[Tensor],
          state_steps: List[int],
          *,
          beta1: float,
          beta2: float,
          lr: float,
          weight_decay: float,
          eps: float):
    r"""Functional API that performs RAdam algorithm computation.
    See :class:`~torch.optim.RAdam` for details.
    """

    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # correcting bias for the first moving moment
        bias_corrected_exp_avg = exp_avg / bias_correction1

        # maximum length of the approximated SMA
        rho_inf = 2 / (1 - beta2) - 1
        # compute the length of the approximated SMA
        rho_t = rho_inf - 2 * step * (beta2 ** step) / bias_correction2

        if rho_t > 5.:
            # Compute the variance rectification term and update parameters accordingly
            rect = math.sqrt((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
            adaptive_lr = math.sqrt(bias_correction2) / exp_avg_sq.sqrt().add_(eps)

            param.add_(bias_corrected_exp_avg * lr * adaptive_lr * rect, alpha=-1.0)
        else:
            param.add_(bias_corrected_exp_avg * lr, alpha=-1.0)


def init_normal(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=0.0, std=0.6)
        if m.bias is not None:
            m.bias.data.zero_()


def text_to_word(text):
    word = text.split()
    word_offset = []

    start = 0
    for w in word:
        r = text[start:].find(w)

        if r == -1:
            raise NotImplementedError
        else:
            start = start + r
            end = start + len(w)
            word_offset.append((start, end))
            # print('%32s'%w, '%5d'%start, '%5d'%r, text[start:end])
        start = end

    return word, word_offset


def text_to_word_sentence_cut(text):
    sentences = sentence_token_nltk(text)
    word_offset = []
    word = []
    start = 0
    word_sentence_id = []
    sentence_offset = []
    for idx, sentence in enumerate(sentences):
        sentence = sentence.split()
        s_start = len(word)
        for w in sentence:
            r = text[start:].find(w)
            if r == -1:
                raise NotImplementedError
            else:
                start = start + r
                end = start + len(w)
                word_offset.append((start, end))
                word_sentence_id.append(idx)
                # print('%32s'%w, '%5d'%start, '%5d'%r, text[start:end])
            start = end
        word.extend(sentence)
        s_end = len(word)
        sentence_offset.append((s_start, s_end))
    return word, word_offset, word_sentence_id, sentence_offset


def word_probability_to_predict_df(text_to_word_probability, id):
    len_word = len(text_to_word_probability)
    word_predict = text_to_word_probability.argmax(-1)
    word_score = text_to_word_probability.max(-1)
    #########################################################
    # for i in range(len_word):
    #     if text_to_word_probability[i][10] >= 0.40:
    #         word_predict[i] = 10
    #         word_score[i] = text_to_word_probability[i][10]
    #     if text_to_word_probability[i][11] >= 0.40:
    #         word_predict[i] = 11
    #         word_score[i] = text_to_word_probability[i][11]
    #     if text_to_word_probability[i][12] >= 0.40:
    #         word_predict[i] = 12
    #         word_score[i] = text_to_word_probability[i][12]
    #     if text_to_word_probability[i][13] >= 0.40:
    #         word_predict[i] = 13
    #         word_score[i] = text_to_word_probability[i][13]
    #########################################################
    predict_df = []

    t = 0
    # logging.debug(target_id_map)
    while 1:
        if word_predict[t] not in [
            target_id_map['O'],
            target_id_map['PAD'],
        ]:
            start = t
            b_marker_label = word_predict[t]
        else:
            t = t + 1
            if t == len_word - 1: break
            continue

        t = t + 1
        if t == len_word - 1: break

        # ----
        if id_target_map[b_marker_label][0] == 'B':
            i_marker_label = b_marker_label + 1
        elif id_target_map[b_marker_label][0] == 'I':
            i_marker_label = b_marker_label
        else:
            raise NotImplementedError

        while 1:
            # print(t)
            if (word_predict[t] != i_marker_label) or (t == len_word - 1):
                end = t
                prediction_string = ' '.join([str(i) for i in range(start, end)])  # np.arange(start,end).tolist()
                discourse_type = id_target_map[b_marker_label][2:]
                discourse_score = word_score[start:end].tolist()
                predict_df.append((id, discourse_type, prediction_string, str(discourse_score)))
                # print(predict_df[-1])
                break
            else:
                t = t + 1
                continue
        if t == len_word - 1: break

    predict_df = pd.DataFrame(predict_df, columns=['id', 'class', 'predictionstring', 'score'])
    return predict_df


def do_threshold(submit_df, use=['length', 'probability']):
    df = submit_df.copy()
    df = df.fillna('')
    min_thresh = {
        "Lead": 9,
        "Position": 5,
        "Evidence": 14,
        "Claim": 3,
        "Concluding Statement": 11,
        "Counterclaim": 6,
        "Rebuttal": 4,
    }
    proba_thresh = {
        "Lead": 0.7,
        "Position": 0.55,
        "Evidence": 0.65,
        "Claim": 0.55,
        "Concluding Statement": 0.7,
        "Counterclaim": 0.5,  # 0.5
        "Rebuttal": 0.55,  # 0.55
    }
    if 'length' in use:
        df['l'] = df.predictionstring.apply(lambda x: len(x.split()))
        for key, value in min_thresh.items():
            # value=3
            index = df.loc[df['class'] == key].query('l<%d' % value).index
            df.drop(index, inplace=True)

    if 'probability' in use:
        df['s'] = df.score.apply(lambda x: np.mean(eval(x)))
        for key, value in proba_thresh.items():
            index = df.loc[df['class'] == key].query('s<%f' % value).index
            df.drop(index, inplace=True)

    df = df[['id', 'class', 'predictionstring']]
    return df


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def do_threshold_test(submit_df, min_thresh, proba_thresh, use=['length', 'probability']):
    df = submit_df.copy()
    df = df.fillna('')
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
    #     "Lead": 0.7,
    #     "Position": 0.55,
    #     "Evidence": 0.65,
    #     "Claim": 0.55,
    #     "Concluding Statement": 0.7,
    #     "Counterclaim": 0.5, # 0.5
    #     "Rebuttal": 0.55, # 0.55
    # }
    if 'length' in use:
        df['l'] = df.predictionstring.apply(lambda x: len(x.split()))
        for key, value in min_thresh.items():
            # value=3
            index = df.loc[df['class'] == key].query('l<%d' % value).index
            df.drop(index, inplace=True)

    if 'probability' in use:
        df['s'] = df.score.apply(lambda x: np.mean(eval(x)))
        for key, value in proba_thresh.items():
            index = df.loc[df['class'] == key].query('s<%f' % value).index
            df.drop(index, inplace=True)

    df = df[['id', 'class', 'predictionstring']]
    return df


def word_probability_to_prediction_string(text_to_word_probability, text_id, word):
    length_threshold = {
        "Lead": 9,
        "Position": 5,
        "Evidence": 14,
        "Claim": 3,
        "Concluding Statement": 11,
        "Counterclaim": 6,
        "Rebuttal": 4,
    }
    word_predict = text_to_word_probability.argmax(-1)
    word_score = text_to_word_probability.max(-1)
    predict_df = []

    t = 0
    while 1:
        # if word_predict[t] in [1,3,5,7,9,11,13]:
        if word_predict[t] in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
            start = t
            b_marker_label = word_predict[t]
        else:
            t = t + 1
            if t == len(word) - 1:
                break
            continue

        t = t + 1
        if t == len(word) - 1:
            break

        # i_marker_label = b_marker_label+1
        i_marker_label = [b_marker_label +
                          1] if b_marker_label % 2 == 0 else [b_marker_label]
        marker_text = id_target_map[i_marker_label[0]]

        # modified1
        consecutive_list = ['Lead', 'Position', 'Concluding', 'Rebuttal']
        if any([x in marker_text for x in consecutive_list]):
            i_marker_label.append(i_marker_label[0] - 1)
            # print(i_marker_label)
        # i_marker_label = [b_marker_label,b_marker_label+1] if b_marker_label%2==1 else [b_marker_label-1,b_marker_label]

        total_others_count = 0
        cur_others_count = 0
        tolerance = 0
        while 1:
            # print(t)
            if word_predict[t] not in i_marker_label and total_others_count < tolerance and t < len(word) - 1:
                total_others_count += 1
                cur_others_count += 1
                t += 1
            elif (word_predict[t] not in i_marker_label) or (t == len(word) - 1):
                t -= cur_others_count
                end = t
                # have bug here
                # ' '.join([str(i) for i in range(start,end)]) #np.arange(start,end).tolist()
                prediction_string = [i for i in range(start, end + 1)]
                # ' '.join(word[i] for i in range(start, end))
                prediction_text = [word[i] for i in range(start, end + 1)]
                discourse_type = id_target_map[b_marker_label][2:]
                if end == start:
                    discourse_score = [word_score[start]]
                elif end == start + 1:  # length = 2
                    # + [np.mean(word_score[start: end])]
                    discourse_score = word_score[start: end].tolist()
                else:
                    # + [np.mean(word_score[start: end])]
                    discourse_score = word_score[start + 1: end].tolist()

                # discourse_score = word_score[start + 1: end].tolist() + [np.mean(word_score[start: end])]# if end - start <= 1 else []
                # 将Concluding延长到最后一个词语
                # if 'Concluding' in discourse_type and len(word) - 1 > t >= len(word) - 3:
                # 	print(discourse_type)
                # 	t += 1

                predict_df.append(
                    (text_id, discourse_type, prediction_text, prediction_string, discourse_score))
                # print(predict_df[-1])
                break
            else:
                cur_others_count = 0
                t = t + 1
                continue
        if t == len(word) - 1:
            break

    # modified 3 keep Lead
    filtered_predict_df = list(
        filter(lambda x: 'Lead' not in x[1], predict_df))
    lead_df = list(filter(lambda x: 'Lead' in x[1], predict_df))
    min_lead_score = 0.91
    if len(lead_df) > 1:
        lead_df = sorted(lead_df, key=lambda x: np.mean(x[4]), reverse=True)
        lead_df = [lead_df[0]] + \
                  list(filter(lambda x: np.mean(x[4]) > min_lead_score, lead_df[1:]))
        begin = min([x[3][0] for x in lead_df])
        end = max([x[3][-1] for x in lead_df])
        lead_df = [(lead_df[0][0], lead_df[0][1], [word[i] for i in range(
            begin, end + 1)], [i for i in range(begin, end + 1)], word_score[begin:end + 1].tolist())]
        predict_df = lead_df + filtered_predict_df

    # modified4 keep Concluding
    filtered_predict_df = list(
        filter(lambda x: 'Concluding' not in x[1], predict_df))
    con_df = list(filter(lambda x: 'Concluding' in x[1], predict_df))
    min_con_score = 0.7
    if len(con_df) > 1:
        con_df = sorted(con_df, key=lambda x: np.mean(x[4]), reverse=True)
        # + list(filter(lambda x: np.mean(x[4]) > min_con_score, con_df[1:]))
        con_df = con_df[:2]
        begin = min(con_df[0][3][0], con_df[1][3][0])
        end = max(con_df[0][3][-1], con_df[1][3][-1])
        con_df = [(con_df[0][0], con_df[0][1], [word[i] for i in range(
            begin, end + 1)], [i for i in range(begin, end + 1)], word_score[begin + 1:end].tolist())]
        predict_df = filtered_predict_df + con_df

    # modified6 keep Position
    filtered_predict_df = list(
        filter(lambda x: 'Position' not in x[1], predict_df))
    pos_df = list(filter(lambda x: 'Position' in x[1] and len(
        x[2]) > length_threshold['Position'], predict_df))
    min_pos_score = 0.9
    if len(pos_df) > 1:
        pos_df = sorted(pos_df, key=lambda x: np.mean(x[4]), reverse=True)
        pos_df = pos_df[:1] + \
                 list(filter(lambda x: np.mean(x[4]) > min_pos_score, pos_df[1:]))
        if len(pos_df) >= 2:
            pos_df = sorted(pos_df, key=lambda x: x[3], reverse=False)
        # begin = min(pos_df[0][3][0], pos_df[1][3][0])
        # end = max(pos_df[0][3][-1], pos_df[1][3][-1])
        # pos_df = [(pos_df[0][0], pos_df[0][1], [word[i] for i in range(begin, end + 1)], [i for i in range(begin,end+1)], word_score[begin:end].tolist())]
        predict_df = filtered_predict_df + pos_df

    for i in range(len(predict_df)):
        predict_df[i] = (predict_df[i][0], predict_df[i][1], ' '.join(
            predict_df[i][2]), ' '.join(str(x) for x in predict_df[i][3]), str(predict_df[i][4]))
    predict_df = pd.DataFrame(predict_df, columns=[
        'id', 'class', 'predict_text', 'predictionstring', 'score'])
    return predict_df


def word_probability_to_prediction_string_v4(text_to_word_probability, text_id, word):
    # print(11)
    length_threshold = {
        "Lead": 9,
        "Position": 5,
        "Evidence": 14,
        "Claim": 3,
        "Concluding Statement": 11,
        "Counterclaim": 6,
        "Rebuttal": 4,
    }
    word_predict = text_to_word_probability.argmax(-1)
    word_score = text_to_word_probability.max(-1)
    predict_df = []

    t = 0
    while 1:
        # if word_predict[t] in [1,3,5,7,9,11,13]:
        if t == len(word):
            break

        if word_predict[t] in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
            start = t
            b_marker_label = word_predict[t]
        else:
            t = t + 1
            if t == len(word) - 1:
                break
            continue

        t = t + 1
        if t == len(word) - 1:
            break

        # i_marker_label = b_marker_label+1
        i_marker_label = [b_marker_label +
                          1] if b_marker_label % 2 == 0 else [b_marker_label]
        marker_text = id_target_map[i_marker_label[0]]

        total_others_count = 0
        cur_others_count = 0
        tolerance_cur = 0
        tolerance_total = 0
        # modified1
        consecutive_list = ['Lead', 'Position', 'Concluding', 'Rebuttal']
        if any([x in marker_text for x in consecutive_list]):
            i_marker_label.append(i_marker_label[0] - 1)
            # if any([x in marker_text for x in consecutive_list]):
            # # modified 7 rebuttal tolerance
        #     if 'Rebuttal' in marker_text:
        #         tolerance_cur = 7
        #         tolerance_total = 15
        # # i_marker_label = [b_marker_label,b_marker_label+1] if b_marker_label%2==1 else [b_marker_label-1,b_marker_label]

        while 1:
            # print(t)
            if t < len(word) and word_predict[t] not in i_marker_label and total_others_count < tolerance_total and cur_others_count < tolerance_cur:
                total_others_count += 1
                cur_others_count += 1
                t += 1
            elif t == len(word) or (word_predict[t] not in i_marker_label):
                t -= cur_others_count
                end = t
                # ' '.join([str(i) for i in range(start,end)]) #np.arange(start,end).tolist()

                # modified 6
                if 20 > end - start > 5:
                    prediction_string = [i for i in range(start, end + 1 if end != len(word) else end)]
                    # ' '.join(word[i] for i in range(start, end))
                    prediction_text = [word[i] for i in range(start, end + 1 if end != len(word) else end)]
                else:
                    prediction_string = [i for i in range(start, end)]
                    # ' '.join(word[i] for i in range(start, end))
                    prediction_text = [word[i] for i in range(start, end)]

                # extend

                # prediction_string = [i for i in range(start, end)]
                # prediction_text = [word[i] for i in range(start, end)]


                discourse_type = id_target_map[b_marker_label][2:]
                # modified 5
                if end == start or end == start + 1:
                    discourse_score = [word_score[start]]
                elif end == start + 2:  # length = 2
                    # + [np.mean(word_score[start: end])]
                    discourse_score = word_score[start: end - 1].tolist()
                else:
                    # + [np.mean(word_score[start: end])]
                    discourse_score = word_score[start: end - 1].tolist() + [np.mean(word_score[start:end])]

                # discourse_score = word_score[start: end].tolist()
                # 将Concluding延长到最后一个词语

                predict_df.append(
                    (text_id, discourse_type, prediction_text, prediction_string, discourse_score))
                # print(predict_df[-1])
                break
            else:
                cur_others_count = 0
                t = t + 1
                continue
        if t == len(word) - 1:
            break

    # modified 3 keep Lead
    filtered_predict_df = list(
        filter(lambda x: 'Lead' not in x[1], predict_df))
    lead_df = list(filter(lambda x: 'Lead' in x[1], predict_df))
    min_lead_score = 0.95
    if len(lead_df) > 1:
        lead_df = sorted(lead_df, key=lambda x: np.mean(x[4]), reverse=True)
        lead_df = [lead_df[0]] + \
                  list(filter(lambda x: np.mean(x[4]) > min_lead_score, lead_df[1:]))
        begin = min([x[3][0] for x in lead_df])
        end = max([x[3][-1] for x in lead_df])
        lead_df = [(lead_df[0][0], lead_df[0][1], [word[i] for i in range(
            begin, end + 1)], [i for i in range(begin, end + 1)], word_score[begin:end + 1].tolist())]
        predict_df = lead_df + filtered_predict_df

    # modified4 keep Concluding
    filtered_predict_df = list(
        filter(lambda x: 'Concluding' not in x[1], predict_df))
    con_df = list(filter(lambda x: 'Concluding' in x[1], predict_df))
    min_con_score = 0.7
    if len(con_df) > 1:
        con_df = sorted(con_df, key=lambda x: np.mean(x[4]), reverse=True)
        # + list(filter(lambda x: np.mean(x[4]) > min_con_score, con_df[1:]))
        con_df = con_df[:2]
        begin = min(con_df[0][3][0], con_df[1][3][0])
        end = max(con_df[0][3][-1], con_df[1][3][-1])
        con_df = [(con_df[0][0], con_df[0][1], [word[i] for i in range(
            begin, end + 1)], [i for i in range(begin, end + 1)], word_score[begin + 1:end].tolist())]
        predict_df = filtered_predict_df + con_df

    # # modified 8 keep Position
    # filtered_predict_df = list(
    #     filter(lambda x: 'Position' not in x[1], predict_df))
    # pos_df = list(filter(lambda x: 'Position' in x[1] and len(
    #     x[2]) > length_threshold['Position'], predict_df))
    # min_pos_score = 0.9
    # if len(pos_df) > 1:
    #     pos_df = sorted(pos_df, key=lambda x: np.mean(x[4]), reverse=True)
    #     pos_df = pos_df[:1] + \
    #              list(filter(lambda x: np.mean(x[4]) > min_pos_score, pos_df[1:]))
    #     if len(pos_df) >= 2:
    #         pos_df = sorted(pos_df, key=lambda x: x[3], reverse=False)
    #     # begin = min(pos_df[0][3][0], pos_df[1][3][0])
    #     # end = max(pos_df[0][3][-1], pos_df[1][3][-1])
    #     # pos_df = [(pos_df[0][0], pos_df[0][1], [word[i] for i in range(begin, end + 1)], [i for i in range(begin,end+1)], word_score[begin:end].tolist())]
    #     predict_df = filtered_predict_df + pos_df

    for i in range(len(predict_df)):
        predict_df[i] = (predict_df[i][0], predict_df[i][1], ' '.join(
            predict_df[i][2]), ' '.join(str(x) for x in predict_df[i][3]), str(predict_df[i][4]))
    predict_df = pd.DataFrame(predict_df, columns=[
        'id', 'class', 'predict_text', 'predictionstring', 'score'])
    return predict_df


def word_probability_to_prediction_string_v3(text_to_word_probability, text_id, word):
    length_threshold = {
        "Lead": 9,
        "Position": 5,
        "Evidence": 14,
        "Claim": 3,
        "Concluding Statement": 11,
        "Counterclaim": 6,
        "Rebuttal": 4,
    }
    word_predict = text_to_word_probability.argmax(-1)
    word_score = text_to_word_probability.max(-1)
    predict_df = []

    t = 0
    while 1:
        # if word_predict[t] in [1,3,5,7,9,11,13]:
        if t == len(word):
            break

        if word_predict[t] in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
            start = t
            b_marker_label = word_predict[t]
        else:
            t = t + 1
            if t == len(word) - 1:
                break
            continue

        t = t + 1
        if t == len(word) - 1:
            break

        # i_marker_label = b_marker_label+1
        i_marker_label = [b_marker_label +
                          1] if b_marker_label % 2 == 0 else [b_marker_label]
        marker_text = id_target_map[i_marker_label[0]]

        total_others_count = 0
        cur_others_count = 0
        tolerance_cur = 0
        tolerance_total = 0
        # modified1
        consecutive_list = ['Lead', 'Position', 'Concluding', 'Rebuttal']
        if any([x in marker_text for x in consecutive_list]):
            i_marker_label.append(i_marker_label[0] - 1)
            # if any([x in marker_text for x in consecutive_list]):
            # # modified 7 rebuttal tolerance
        #     if 'Rebuttal' in marker_text:
        #         tolerance_cur = 7
        #         tolerance_total = 15
        # # i_marker_label = [b_marker_label,b_marker_label+1] if b_marker_label%2==1 else [b_marker_label-1,b_marker_label]

        while 1:
            # print(t)
            if t < len(word) and word_predict[t] not in i_marker_label and total_others_count < tolerance_total and cur_others_count < tolerance_cur:
                total_others_count += 1
                cur_others_count += 1
                t += 1
            elif t == len(word) or (word_predict[t] not in i_marker_label):
                t -= cur_others_count
                end = t
                # ' '.join([str(i) for i in range(start,end)]) #np.arange(start,end).tolist()

                # modified 6
                # if 20 > end - start > 5:
                #     prediction_string = [i for i in range(start, end + 1 if end != len(word) else end)]
                #     # ' '.join(word[i] for i in range(start, end))
                #     prediction_text = [word[i] for i in range(start, end + 1 if end != len(word) else end)]
                # else:
                #     prediction_string = [i for i in range(start, end)]
                #     # ' '.join(word[i] for i in range(start, end))
                #     prediction_text = [word[i] for i in range(start, end)]

                # extend

                prediction_string = [i for i in range(start, end)]
                prediction_text = [word[i] for i in range(start, end)]


                discourse_type = id_target_map[b_marker_label][2:]
                # modified 5
                # if end == start or end == start + 1:
                #     discourse_score = [word_score[start]]
                # elif end == start + 2:  # length = 2
                #     # + [np.mean(word_score[start: end])]
                #     discourse_score = word_score[start: end - 1].tolist()
                # else:
                #     # + [np.mean(word_score[start: end])]
                #     discourse_score = word_score[start: end - 1].tolist() + [np.mean(word_score[start:end])]

                discourse_score = word_score[start: end].tolist()
                # 将Concluding延长到最后一个词语

                predict_df.append(
                    (text_id, discourse_type, prediction_text, prediction_string, discourse_score))
                # print(predict_df[-1])
                break
            else:
                cur_others_count = 0
                t = t + 1
                continue
        if t == len(word) - 1:
            break

    # modified 3 keep Lead
    filtered_predict_df = list(
        filter(lambda x: 'Lead' not in x[1], predict_df))
    lead_df = list(filter(lambda x: 'Lead' in x[1], predict_df))
    min_lead_score = 0.95
    if len(lead_df) > 1:
        lead_df = sorted(lead_df, key=lambda x: np.mean(x[4]), reverse=True)
        lead_df = [lead_df[0]] + \
                  list(filter(lambda x: np.mean(x[4]) > min_lead_score, lead_df[1:]))
        begin = min([x[3][0] for x in lead_df])
        end = max([x[3][-1] for x in lead_df])
        lead_df = [(lead_df[0][0], lead_df[0][1], [word[i] for i in range(
            begin, end + 1)], [i for i in range(begin, end + 1)], word_score[begin:end + 1].tolist())]
        predict_df = lead_df + filtered_predict_df

    # modified4 keep Concluding
    filtered_predict_df = list(
        filter(lambda x: 'Concluding' not in x[1], predict_df))
    con_df = list(filter(lambda x: 'Concluding' in x[1], predict_df))
    min_con_score = 0.7
    if len(con_df) > 1:
        con_df = sorted(con_df, key=lambda x: np.mean(x[4]), reverse=True)
        # + list(filter(lambda x: np.mean(x[4]) > min_con_score, con_df[1:]))
        con_df = con_df[:2]
        begin = min(con_df[0][3][0], con_df[1][3][0])
        end = max(con_df[0][3][-1], con_df[1][3][-1])
        con_df = [(con_df[0][0], con_df[0][1], [word[i] for i in range(
            begin, end + 1)], [i for i in range(begin, end + 1)], word_score[begin + 1:end].tolist())]
        predict_df = filtered_predict_df + con_df

    # # modified 8 keep Position
    # filtered_predict_df = list(
    #     filter(lambda x: 'Position' not in x[1], predict_df))
    # pos_df = list(filter(lambda x: 'Position' in x[1] and len(
    #     x[2]) > length_threshold['Position'], predict_df))
    # min_pos_score = 0.9
    # if len(pos_df) > 1:
    #     pos_df = sorted(pos_df, key=lambda x: np.mean(x[4]), reverse=True)
    #     pos_df = pos_df[:1] + \
    #              list(filter(lambda x: np.mean(x[4]) > min_pos_score, pos_df[1:]))
    #     if len(pos_df) >= 2:
    #         pos_df = sorted(pos_df, key=lambda x: x[3], reverse=False)
    #     # begin = min(pos_df[0][3][0], pos_df[1][3][0])
    #     # end = max(pos_df[0][3][-1], pos_df[1][3][-1])
    #     # pos_df = [(pos_df[0][0], pos_df[0][1], [word[i] for i in range(begin, end + 1)], [i for i in range(begin,end+1)], word_score[begin:end].tolist())]
    #     predict_df = filtered_predict_df + pos_df

    for i in range(len(predict_df)):
        predict_df[i] = (predict_df[i][0], predict_df[i][1], ' '.join(
            predict_df[i][2]), ' '.join(str(x) for x in predict_df[i][3]), str(predict_df[i][4]))
    predict_df = pd.DataFrame(predict_df, columns=[
        'id', 'class', 'predict_text', 'predictionstring', 'score'])
    return predict_df


def word_probability_to_prediction_string_v5(text_to_word_probability, text_id, word, convert_table):
    # print(11)

    word_predict = text_to_word_probability.argmax(-1)
    word_score = text_to_word_probability.max(-1)
    predict_df = []

    t = 0
    while 1:
        # if word_predict[t] in [1,3,5,7,9,11,13]:
        if t == len(word):
            break

        if word_predict[t] in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
            start = t
            b_marker_label = word_predict[t]
        else:
            t = t + 1
            if t == len(word) - 1:
                break
            continue

        t = t + 1
        if t == len(word) - 1:
            break

        # i_marker_label = b_marker_label+1
        i_marker_label = [b_marker_label + 1] if b_marker_label % 2 == 0 else [b_marker_label]
        marker_text = id_target_map[i_marker_label[0]]

        total_others_count = 0
        cur_others_count = 0
        tolerance_cur = 0
        tolerance_total = 0
        # modified1
        consecutive_list = ['Lead', 'Position', 'Concluding', 'Rebuttal']
        if any([x in marker_text for x in consecutive_list]):
            i_marker_label.append(i_marker_label[0] - 1)
            # if any([x in marker_text for x in consecutive_list]):
            # modified 7 rebuttal tolerance
            # if 'Rebuttal' in marker_text:
            #     tolerance_cur = 6
            #     tolerance_total = 15
            # #         'Rebuttal': ('0.5308', '0.6101', '0.4697')
            # # [7, 15] 'Rebuttal': ('0.5249', '0.6109', '0.4600')
            # # [4, 15] 'Rebuttal': ('0.5241', '0.6090', '0.4600')
            # # [6, 15] 

        # # i_marker_label = [b_marker_label,b_marker_label+1] if b_marker_label%2==1 else [b_marker_label-1,b_marker_label]

        while 1:
            # print(t)
            if t < len(word) and word_predict[t] not in i_marker_label and total_others_count < tolerance_total and cur_others_count < tolerance_cur:
                total_others_count += 1
                cur_others_count += 1
                t += 1
            elif t == len(word) or (word_predict[t] not in i_marker_label):
                t -= cur_others_count
                end = t
                # ' '.join([str(i) for i in range(start,end)]) #np.arange(start,end).tolist()

                # # modified 6
                if 20 > end - start > 5:
                    prediction_string = [i for i in range(start, end + 1 if end != len(word) else end)]
                    # ' '.join(word[i] for i in range(start, end))
                    prediction_text = [word[i] for i in range(start, end + 1 if end != len(word) else end)]
                else:
                    prediction_string = [i for i in range(start, end)]
                    # ' '.join(word[i] for i in range(start, end))
                    prediction_text = [word[i] for i in range(start, end)]

                # prediction_string = [i for i in range(start, end)]
                # prediction_text = [word[i] for i in range(start, end)]

                discourse_type = id_target_map[b_marker_label][2:]

                discourse_score = word_score[start: end].tolist()
                # 将Concluding延长到最后一个词语

                predict_df.append((text_id, discourse_type, prediction_text, prediction_string, discourse_score))
                # print(predict_df[-1])
                break
            else:
                cur_others_count = 0
                t = t + 1
                continue
        if t == len(word) - 1:
            break

    # modified 3 keep Lead
    filtered_predict_df = list(
        filter(lambda x: 'Lead' not in x[1], predict_df))
    lead_df = list(filter(lambda x: 'Lead' in x[1], predict_df))
    min_lead_score = 0.95
    if len(lead_df) > 1:
        lead_df = sorted(lead_df, key=lambda x: np.mean(x[4]), reverse=True)
        lead_df = [lead_df[0]] + \
                  list(filter(lambda x: np.mean(x[4]) > min_lead_score, lead_df[1:]))
        begin = min([x[3][0] for x in lead_df])
        end = max([x[3][-1] for x in lead_df])
        lead_df = [(lead_df[0][0], lead_df[0][1], [word[i] for i in range(
            begin, end + 1)], [i for i in range(begin, end + 1)], word_score[begin:end + 1].tolist())]
        predict_df = lead_df + filtered_predict_df

    # modified4 keep Concluding
    filtered_predict_df = list(
        filter(lambda x: 'Concluding' not in x[1], predict_df))
    con_df = list(filter(lambda x: 'Concluding' in x[1], predict_df))
    min_con_score = 0.7
    if len(con_df) > 1:
        con_df = sorted(con_df, key=lambda x: np.mean(x[4]), reverse=True)
        # + list(filter(lambda x: np.mean(x[4]) > min_con_score, con_df[1:]))
        con_df = con_df[:2]
        begin = min(con_df[0][3][0], con_df[1][3][0])
        end = max(con_df[0][3][-1], con_df[1][3][-1])
        con_df = [(con_df[0][0], con_df[0][1], [word[i] for i in range(
            begin, end + 1)], [i for i in range(begin, end + 1)], word_score[begin + 1:end].tolist())]
        predict_df = filtered_predict_df + con_df

    # convert_table = {
    #     # 'Lead': {'Lead': 0.9, 'Position': 1.1, 'Evidence': 0.95, 'Claim': 0.98, 'Concluding Statement': 0.9, 'Counterclaim': 0.93, 'Rebuttal': 0.9,},
    #     'Position': {'Lead': 0.9, 'Position': 0.9, 'Evidence': 1., 'Claim': 1.1, 'Concluding Statement': 0.95, 'Counterclaim': 0.94, 'Rebuttal': 0.9,},
    #     # 'Evidence': {'Lead': 0.9, 'Position': 0.95, 'Evidence': 1., 'Claim': 1.1, 'Concluding Statement': 1.1, 'Counterclaim': 1., 'Rebuttal': 0.95,},
    #     # 'Claim': {'Lead': 0.94, 'Position': 0.95, 'Evidence': 1.2, 'Claim': 1.1, 'Concluding Statement': 1, 'Counterclaim': 1, 'Rebuttal': 0.9,},
    #     # 'Concluding Statement': {'Lead': 0.9, 'Position': 0.94, 'Evidence': 1., 'Claim': 1.1, 'Concluding Statement': 0.95, 'Counterclaim': 0.94, 'Rebuttal': 0.9,},
    #     # 'Counterclaim': {'Lead': 0.9, 'Position': 1.1, 'Evidence': 1., 'Claim': 1., 'Concluding Statement': 0.95, 'Counterclaim': 1., 'Rebuttal': 0.9,},
    #     # 'Rebuttal': {'Lead': 0.9, 'Position': 0.95, 'Evidence': 1.1, 'Claim': 1., 'Concluding Statement': 1., 'Counterclaim': 0.95, 'Rebuttal': 0.92,},
    # }
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
    #     "Lead": 0.617628220048235, # 0.7
    #     "Position": 0.5404662917593531, # 0.55
    #     "Evidence": 0.5792470568116815, # 0.65
    #     "Claim": 0.5385829262728876, # 0.55
    #     "Concluding Statement": 0.6235012425556871, # 0.7
    #     "Counterclaim": 0.4975126082187205, # 0.5
    #     "Rebuttal": 0.5444709754299981, # 0.55
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
    predict_df = sorted(predict_df, key=lambda x: x[3][0])
    for i, predict in enumerate(predict_df[:-1]):
        cur_class = predict[1]
        if cur_class in convert_table.keys() and len(predict[3]) >= min_thresh[cur_class] and np.mean(predict[-1]) > proba_thresh[cur_class]:
            new_score = (np.array(predict_df[i+1][-1]) * convert_table[cur_class][predict_df[i+1][1]]).tolist()
            predict_df[i+1] = (predict_df[i+1][0], predict_df[i+1][1], predict_df[i+1][2], predict_df[i+1][3], new_score)

    for i in range(len(predict_df)):
        predict_df[i] = (predict_df[i][0], predict_df[i][1], ' '.join(
            predict_df[i][2]), ' '.join(str(x) for x in predict_df[i][3]), str(predict_df[i][4]))
    predict_df = pd.DataFrame(predict_df, columns=[
        'id', 'class', 'predict_text', 'predictionstring', 'score'])
    return predict_df