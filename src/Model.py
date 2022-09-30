import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from transformers import AutoModel, AutoConfig, T5EncoderModel
import numpy as np
import torch.nn.utils.rnn as rnn_utils
import Utils
START_TAG = "[CLS]"
STOP_TAG = "[SEP]"


class ModelConfig(object):
    def __init__(self, args):
        self.pretrain_path = args.pretrain_path
        self.hidden_dropout_prob = 0.1
        self.layer_norm_eps = 1e-7
        self.num_labels = args.num_labels
        self.device = args.device
        self.dropout = args.dropout


class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)


class Activation(nn.Module):
    def __init__(self, name="swish"):
        super(Activation, self).__init__()
        if name not in ["swish", "relu", "gelu"]:
            raise
        if name == "swish":
            self.net = Swish()
        elif name == "relu":
            self.net = nn.ReLU()
        elif name == "gelu":
            self.net = nn.GELU()
    
    def forward(self, x):
        return self.net(x)


class Dence(nn.Module):
    def __init__(self, i_dim, o_dim, activation="swish"):
        super(Dence, self).__init__()
        self.dence = nn.Sequential(
            nn.Linear(i_dim, o_dim),
            # nn.ReLU(),
            Activation(activation),
        )

    def forward(self, x):
        return self.dence(x)


class FeedBackModel(nn.Module):
    def __init__(self, args):
        super(FeedBackModel, self).__init__()
        config = AutoConfig.from_pretrained(args.pretrain_path)
        config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": args.hidden_dropout_prob,
                "layer_norm_eps": args.layer_norm_eps,
                "add_pooling_layer": False,
                "num_labels": args.num_labels,
            }
        )
        self.num_labels = args.num_labels
        if "t5" in args.pretrain_path:
            self.transformer = T5EncoderModel.from_pretrained(args.pretrain_path)
        else:
            self.transformer = AutoModel.from_pretrained(args.pretrain_path)
        # self.transformer.gradient_checkpointing_enable()
        self.dropout = nn.Dropout(args.dropout)

        # self.dropout1 = nn.Dropout(0.1)
        # self.dropout2 = nn.Dropout(0.2)
        # self.dropout3 = nn.Dropout(0.3)
        # self.dropout4 = nn.Dropout(0.4)
        # self.dropout5 = nn.Dropout(0.5)
        
        # self.output = nn.Sequential(
        #     Dence(config.hidden_size, config.hidden_size, "relu"),
        #     nn.Dropout(0.1),
        #     Dence(config.hidden_size, config.hidden_size // 4, "relu"),
        #     nn.Dropout(0.1),
        #     nn.Linear(config.hidden_size // 4, self.num_labels),
        # )
        # self.output.apply(Utils.init_normal)
        self.output = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        if token_type_ids:
            transformer_out = self.transformer(input_ids, attention_mask, token_type_ids)
        else:
            transformer_out = self.transformer(input_ids, attention_mask)
        sequence_output = transformer_out.last_hidden_state
        sequence_output = self.dropout(sequence_output)

        # logits1 = self.output(self.dropout1(sequence_output))
        # logits2 = self.output(self.dropout2(sequence_output))
        # logits3 = self.output(self.dropout3(sequence_output))
        # logits4 = self.output(self.dropout4(sequence_output))
        # logits5 = self.output(self.dropout5(sequence_output))
        # logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
        
        logits = self.output(sequence_output)
        logits_out = torch.softmax(logits, dim=-1)
        loss = 0
        if labels is not None:
            loss = self.loss(logits, labels, attention_mask=attention_mask)

            # loss1 = self.loss(logits1, labels, attention_mask=attention_mask)
            # loss2 = self.loss(logits2, labels, attention_mask=attention_mask)
            # loss3 = self.loss(logits3, labels, attention_mask=attention_mask)
            # loss4 = self.loss(logits4, labels, attention_mask=attention_mask)
            # loss5 = self.loss(logits5, labels, attention_mask=attention_mask)
            # loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5

        return logits_out, loss
    
    def loss(self, logits, labels, attention_mask):
        loss_fct = nn.CrossEntropyLoss()

        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, self.num_labels)
        true_labels = labels.view(-1)
        outputs = active_logits.argmax(dim=-1)
        idxs = np.where(active_loss.cpu().numpy() == 1)[0]
        active_logits = active_logits[idxs]
        true_labels = true_labels[idxs].to(torch.long)
        # logging.debug(f"true labels: {true_labels.tolist()}")
        loss = loss_fct(active_logits, true_labels)
        return loss


# def argmax(vec):
#     # return the argmax as a python int
#     # 返回vec的dim为1维度上的最大值索引
#     _, idx = torch.max(vec, 1)
#     return idx.item()


# def log_sum_exp(vec):
#     max_score = vec[0, argmax(vec)]
#     max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
#     return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))




# class EB(nn.Module):
#     def __init__(self, args):
#         super(EB, self).__init__()
#         config = AutoConfig.from_pretrained(args.pretrain_path)
#         config.update(
#             {
#                 "output_hidden_states": True,
#                 "hidden_dropout_prob": args.hidden_dropout_prob,
#                 "layer_norm_eps": args.layer_norm_eps,
#                 "add_pooling_layer": False,
#                 "num_labels": args.num_labels,
#             }
#         )
#         self.num_labels = args.num_labels
#         self.transformer = AutoModel.from_pretrained(args.pretrain_path)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         # self.dropout1 = nn.Dropout(0.1)
#         # self.dropout2 = nn.Dropout(0.2)
#         # self.dropout3 = nn.Dropout(0.3)
#         # self.dropout4 = nn.Dropout(0.4)
#         # self.dropout5 = nn.Dropout(0.5)
#         self.output = nn.Linear(config.hidden_size, config.num_labels)

#     def forward(self, input_ids, attention_mask, token_type_ids=None):
#         if token_type_ids:
#             transformer_out = self.transformer(input_ids, attention_mask, token_type_ids)
#         else:
#             transformer_out = self.transformer(input_ids, attention_mask)
#         sequence_output = transformer_out.last_hidden_state
#         sequence_output = self.dropout(sequence_output)

#         logits = self.output(sequence_output)
#         logits_out = torch.softmax(logits, dim=-1)
#         return logits_out


# class CRF(nn.Module):
#     def __init__(self, tagset, start_tag, end_tag, device):
#         super(CRF, self).__init__()
#         self.tagset_size = len(tagset)
#         self.START_TAG_IDX = tagset.index(start_tag)
#         self.END_TAG_IDX = tagset.index(end_tag)
#         self.START_TAG_TENSOR = torch.LongTensor([self.START_TAG_IDX], device=device)
#         self.END_TAG_TENSOR = torch.LongTensor([self.END_TAG_IDX], device=device)
#         # trans: (tagset_size, tagset_size) trans (i, j) means state_i -> state_j
#         self.trans = nn.Parameter(
#             torch.randn(self.tagset_size, self.tagset_size)
#         )
#         # self.trans.data[...] = 1
#         self.trans.data[:, self.START_TAG_IDX] = -10000
#         self.trans.data[self.END_TAG_IDX, :] = -10000
#         self.device = device

#     def init_alpha(self, batch_size, tagset_size):
#         return torch.full((batch_size, tagset_size, 1), -10000, dtype=torch.float, device=self.device)

#     def init_path(self, size_shape):
#         # Initialization Path - LongTensor + Device + Full_value=0
#         return torch.full(size_shape, 0, dtype=torch.long, device=self.device)

#     def _iter_legal_batch(self, batch_input_lens, reverse=False):
#         index = torch.arange(0, batch_input_lens.sum(), dtype=torch.long)
#         packed_index = rnn_utils.pack_sequence(
#             torch.split(index, batch_input_lens.tolist())
#         )
#         batch_iter = torch.split(packed_index.data, packed_index.batch_sizes.tolist())
#         batch_iter = reversed(batch_iter) if reverse else batch_iter
#         for idx in batch_iter:
#             yield idx, idx.size()[0]

#     def score_z(self, feats, batch_input_lens):
#         # 模拟packed pad过程
#         tagset_size = feats.shape[1]
#         batch_size = len(batch_input_lens)
#         alpha = self.init_alpha(batch_size, tagset_size)
#         alpha[:, self.START_TAG_IDX, :] = 0  # Initialization
#         for legal_idx, legal_batch_size in self._iter_legal_batch(batch_input_lens):
#             feat = feats[legal_idx, ].view(legal_batch_size, 1, tagset_size)  # 
#             # #batch * 1 * |tag| + #batch * |tag| * 1 + |tag| * |tag| = #batch * |tag| * |tag|
#             legal_batch_score = feat + alpha[:legal_batch_size, ] + self.trans
#             alpha_new = torch.logsumexp(legal_batch_score, 1).unsqueeze(2)
#             alpha[:legal_batch_size, ] = alpha_new
#         alpha = alpha + self.trans[:, self.END_TAG_IDX].unsqueeze(1)
#         score = torch.logsumexp(alpha, 1).sum()
#         return score

#     def score_sentence(self, feats, batch_target):
#         # CRF Batched Sentence Score
#         # feats: (#batch_state(#words), tagset_size)
#         # batch_target: list<torch.LongTensor> At least One LongTensor
#         # Warning: words order =  batch_target order
#         def _add_start_tag(target):
#             return torch.cat([self.START_TAG_TENSOR, target])

#         def _add_end_tag(target):
#             return torch.cat([target, self.END_TAG_TENSOR])

#         from_state = [_add_start_tag(target) for target in batch_target]
#         to_state = [_add_end_tag(target) for target in batch_target]
#         from_state = torch.cat(from_state)  
#         to_state = torch.cat(to_state)  
#         trans_score = self.trans[from_state, to_state]

#         gather_target = torch.cat(batch_target).view(-1, 1)
#         emit_score = torch.gather(feats, 1, gather_target)  

#         return trans_score.sum() + emit_score.sum()

#     def viterbi(self, feats, batch_input_lens):
#         word_size, tagset_size = feats.shape
#         batch_size = len(batch_input_lens)
#         viterbi_path = self.init_path(feats.shape)  # use feats.shape to init path.shape
#         alpha = self.init_alpha(batch_size, tagset_size)
#         alpha[:, self.START_TAG_IDX, :] = 0  # Initialization
#         for legal_idx, legal_batch_size in self._iter_legal_batch(batch_input_lens):
#             feat = feats[legal_idx, :].view(legal_batch_size, 1, tagset_size)
#             legal_batch_score = feat + alpha[:legal_batch_size, ] + self.trans
#             alpha_new, best_tag = torch.max(legal_batch_score, 1)
#             alpha[:legal_batch_size, ] = alpha_new.unsqueeze(2)
#             viterbi_path[legal_idx, ] = best_tag
#         alpha = alpha + self.trans[:, self.END_TAG_IDX].unsqueeze(1)
#         path_score, best_tag = torch.max(alpha, 1)
#         path_score = path_score.squeeze()  # path_score=#batch

#         best_paths = self.init_path((word_size, 1))
#         for legal_idx, legal_batch_size in self._iter_legal_batch(batch_input_lens, reverse=True):
#             best_paths[legal_idx, ] = best_tag[:legal_batch_size, ]  # 
#             backword_path = viterbi_path[legal_idx, ]  # 1 * |Tag|
#             this_tag = best_tag[:legal_batch_size, ]  # 1 * |legal_batch_size|
#             backword_tag = torch.gather(backword_path, 1, this_tag)
#             best_tag[:legal_batch_size, ] = backword_tag
#             # never computing <START>

#         # best_paths = #words
#         return path_score.view(-1), best_paths.view(-1)


# class Transformers_CRF(nn.Module):
#     def __init__(self, args):
#         super(Transformers_CRF, self).__init__()
#         self.transformers = EB(args)
#         self.CRF = CRF(args.num_labels, "[CLS]", "[SEP]", args.device)

#     def forward(self, batch_input, batch_input_lens, batch_mask):
#         feats = self.transformers(batch_input, batch_input_lens, batch_mask)
#         score, path = self.CRF.viterbi(feats, batch_input_lens)
#         return path

#     def neg_log_likelihood(self, input_ids, attention_mask, labels):
#         feats = self.transformers(input_ids, attention_mask)
#         gold_score = self.CRF.score_sentence(feats, labels)
#         forward_score = self.CRF.score_z(feats, batch_input_lens)
#         return forward_score - gold_score

#     def predict(self, batch_input, batch_input_lens, batch_mask):
#         return self(batch_input, batch_input_lens, batch_mask)