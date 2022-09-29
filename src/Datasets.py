import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import logging
import Utils
import random


class FeedbackDataset:
    def __init__(self, samples, max_len, tokenizer):
        self.samples = samples
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.length = len(samples)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_ids = self.samples[idx]["input_ids"]
        input_labels = self.samples[idx]["input_labels"]
        # logging.debug(f"text: {self.samples[idx]['text']}")
        # logging.debug(f"input_labels: {input_labels}")
        # logging.debug(f"offset mapping: {self.samples[idx]['offset_mapping']}")
        input_labels = [Utils.target_id_map[x] for x in input_labels]
        other_label_id = Utils.target_id_map["O"]
        padding_label_id = Utils.target_id_map["PAD"]
        # logging.debug(f"offset mapping: {self.samples[idx]['offset_mapping']}")
        # print(input_ids)
        # print(input_labels)

        # add start token id to the input_ids
        if self.tokenizer.cls_token_id is not None:
            input_ids = [self.tokenizer.cls_token_id] + input_ids
            input_labels = [other_label_id] + input_labels

        if len(input_ids) > self.max_len - 1:
            input_ids = input_ids[: self.max_len - 1]
            input_labels = input_labels[: self.max_len - 1]

        # add end token id to the input_ids
        if self.tokenizer.sep_token_id is not None:
            input_ids = input_ids + [self.tokenizer.sep_token_id]
            input_labels = input_labels + [other_label_id]

        attention_mask = [1] * len(input_ids)

        padding_length = self.max_len - len(input_ids)
        if padding_length > 0:
            if self.tokenizer.padding_side == "right":
                input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                input_labels = input_labels + [padding_label_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length
            else:
                input_ids = [self.tokenizer.pad_token_id] * padding_length + input_ids
                input_labels = [padding_label_id] * padding_length + input_labels
                attention_mask = [0] * padding_length + attention_mask

        return {
            "ids": torch.tensor(input_ids, dtype=torch.long),
            "mask": torch.tensor(attention_mask, dtype=torch.long),
            "targets": torch.tensor(input_labels, dtype=torch.long),
        }


class FeedbackDatasetCollate:
    def __init__(self, samples, max_len, tokenizer):
        self.samples = samples
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.length = len(samples)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_ids = self.samples[idx]["input_ids"]
        input_labels = self.samples[idx]["input_labels"]
        input_labels = [Utils.target_id_map[x] for x in input_labels]
        other_label_id = Utils.target_id_map["O"]
        padding_label_id = Utils.target_id_map["PAD"]

        # add start token id to the input_ids
        input_ids = [self.tokenizer.cls_token_id] + input_ids
        input_labels = [other_label_id] + input_labels

        if len(input_ids) > self.max_len - 1:
            input_ids = input_ids[: self.max_len - 1]
            input_labels = input_labels[: self.max_len - 1]

        # add end token id to the input_ids
        input_ids = input_ids + [self.tokenizer.sep_token_id]
        input_labels = input_labels + [other_label_id]

        attention_mask = [1] * len(input_ids)
        return {
            "ids": input_ids,
            "mask": attention_mask,
            "targets": input_labels,
        }


class CollateTrain:
    def __init__(self, tokenizer, fix_length=-1):
        self.tokenizer = tokenizer
        self.fix_length = fix_length

    def __call__(self, batch):
        output = dict()
        output["ids"] = [sample["ids"] for sample in batch]
        output["mask"] = [sample["mask"] for sample in batch]
        output["targets"] = [sample["targets"] for sample in batch]
        other_label_id = Utils.target_id_map["O"]
        padding_label_id = Utils.target_id_map["PAD"]

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["ids"]])
        if self.fix_length != -1:
            batch_max = min(batch_max, self.fix_length)

        # add padding
        if self.tokenizer.padding_side == "right":
            output["ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["ids"]]
            output["mask"] = [s + (batch_max - len(s)) * [0] for s in output["mask"]]
            output["targets"] = [s + (batch_max - len(s)) * [padding_label_id] for s in output["targets"]]
        else:
            output["ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in output["ids"]]
            output["mask"] = [(batch_max - len(s)) * [0] + s for s in output["mask"]]
            output["targets"] = [(batch_max - len(s)) * [padding_label_id] for s in output["targets"]]

        # convert to tensors
        output["ids"] = torch.tensor(output["ids"], dtype=torch.long)
        output["mask"] = torch.tensor(output["mask"], dtype=torch.long)
        output["targets"] = torch.tensor(output["targets"], dtype=torch.long)

        return output


class FeedbackDatasetValid:
    def __init__(self, samples, max_len, tokenizer):
        self.samples = samples
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.length = len(samples)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_ids = self.samples[idx]["input_ids"]
        input_ids = [self.tokenizer.cls_token_id] + input_ids

        if len(input_ids) > self.max_len - 1:
            input_ids = input_ids[: self.max_len - 1]

        # add end token id to the input_ids
        input_ids = input_ids + [self.tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)

        return {
            "ids": input_ids,
            "mask": attention_mask,
        }


class Collate:
    def __init__(self, tokenizer, fix_length=-1, fixed=False):
        self.tokenizer = tokenizer
        self.fix_length = fix_length
        self.fixed = fixed

    def __call__(self, batch):
        output = dict()
        output["ids"] = [sample["ids"] for sample in batch]
        output["mask"] = [sample["mask"] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["ids"]])
        if self.fix_length != -1:
            batch_max = min(batch_max, self.fix_length)
            if self.fixed:
                batch_max = self.fix_length

        # add padding
        if self.tokenizer.padding_side == "right":
            output["ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["ids"]]
            output["mask"] = [s + (batch_max - len(s)) * [0] for s in output["mask"]]
        else:
            output["ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in output["ids"]]
            output["mask"] = [(batch_max - len(s)) * [0] + s for s in output["mask"]]

        # convert to tensors
        output["ids"] = torch.tensor(output["ids"], dtype=torch.long)
        output["mask"] = torch.tensor(output["mask"], dtype=torch.long)

        return output


class FeedbackMaxTokenValid:
    def __init__(self, dataset, max_tokens, fix_length):
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.tokenizer = self.dataset.tokenizer
        self.batch_dataset = self.create_batchs()
        self.length = len(self.batch_dataset)
        self.fix_length = fix_length
    
    def __getitem__(self, idx):
        output = dict()
        output["ids"] = self.batch_dataset[idx]['ids']
        output["mask"] = self.batch_dataset[idx]['mask']

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["ids"]])
        if self.fix_length != -1:
            batch_max = min(batch_max, self.fix_length)
#             batch_max = self.fix_length

        # add padding
        if self.tokenizer.padding_side == "right":
            output["ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["ids"]]
            output["mask"] = [s + (batch_max - len(s)) * [0] for s in output["mask"]]
        else:
            output["ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in output["ids"]]
            output["mask"] = [(batch_max - len(s)) * [0] + s for s in output["mask"]]

        # convert to tensors
        output["ids"] = torch.tensor(output["ids"], dtype=torch.long)
        output["mask"] = torch.tensor(output["mask"], dtype=torch.long)
        return output
    
    def create_batchs(self):
        max_tokens_dataset = []
        
        index = 0
        accumulate_items = 0
        max_len_item = 0
        batch = {
            'ids': [],
            'mask': [],
        }
        while index < len(self.dataset):
            item = self.dataset[index]
            ids = item['ids']
            mask = item['mask']
            
            max_len_item = max(max_len_item, len(ids))
            if (accumulate_items + 1) * max_len_item > self.max_tokens:
                max_tokens_dataset.append(batch)
                accumulate_items = 0
                max_len_item = 0
                batch = {
                    'ids': [],
                    'mask': [],
                }
            else:
                accumulate_items += 1
                index += 1
                batch['ids'].append(ids)
                batch['mask'].append(mask)
        if len(batch['ids']) != 0:
            max_tokens_dataset.append(batch)
        return max_tokens_dataset
            

    def __len__(self):
        return self.length