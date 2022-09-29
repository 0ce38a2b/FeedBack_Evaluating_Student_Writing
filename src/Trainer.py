import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import Model, Utils, Datasets
from transformers import AdamW, get_cosine_schedule_with_warmup
import logging
import gc
from apex import amp
import torch.cuda.amp as AMP
from tqdm import tqdm
import numpy as np
amp.register_half_function(torch, 'einsum')


class TrainerConfig(object):
    def __init__(self, args):
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.epoch = args.epoch
        self.opt_step = args.opt_step
        self.eval_step = args.eval_step
        self.Tmax = args.Tmax
        self.min_lr = args.min_lr
        self.scheduler = args.scheduler
        self.max_norm = args.max_norm
        self.model_save = args.model_save
        self.model_load = args.model_load
        self.metrics = args.metrics
        self.eval_continue = args.eval_continue
        self.model_name = args.model_name
        self.debug = args.debug
        self.train_batch_size = args.train_batch_size
        self.valid_batch_size = args.valid_batch_size
        self.fp16 = args.fp16
        self.fgm = args.fgm
        self.radam = args.radam
        self.freeze_step = args.freeze_step
        self.fix_length = args.fix_length
        self.ema = args.ema


class BaseTrainer(object):
    def __init__(self, args):
        self.predict_loss = 0
        self.trainer_config = TrainerConfig(args)
        self.model_config = Model.ModelConfig(args)
        self.device = args.device

    def build_model(self):
        self.model = Model.FeedBackModel(self.model_config)

    def model_init(self):
        self.build_model()
        if self.trainer_config.model_load:
            self.model.load_state_dict(torch.load(self.trainer_config.model_load, map_location=torch.device('cpu')))
        self.model.to(self.device)
        self.model.train()

    def optimizer_init(self):
        optimizer_grouped_parameters = self._get_optimizer_grouped_parameters()
        if self.trainer_config.radam:
            self.optimizer = Utils.RAdam(optimizer_grouped_parameters, eps=1e-7)
        else:
            self.optimizer = AdamW(optimizer_grouped_parameters)
        scheduler_map = {
            "CosineAnnealingLR": lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.trainer_config.Tmax, eta_min=self.trainer_config.min_lr),
            "CosineAnnealingWarmRestarts": lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,T_0=self.trainer_config.Tmax, T_mult=1, eta_min=self.trainer_config.min_lr),
            "get_cosine_schedule_with_warmup": get_cosine_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=int(0.1 * self.training_size), 
                num_training_steps=self.training_size,
                num_cycles=1,
                last_epoch=-1,
            )
        }
        if self.trainer_config.fp16:
            self.model, self.optmizer = amp.initialize(self.model, self.optimizer, opt_level="O1")
        if self.trainer_config.fgm:
            self.fgm = Utils.FGM(self.model)
        if self.trainer_config.ema:
            self.ema = Utils.EMA(self.model, 0.999)
            self.ema.register()
        self.scheduler = scheduler_map[self.trainer_config.scheduler]
        self.f1_maxn = 0
        self.num_step = 0

    def _get_optimizer_grouped_parameters(self):
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if n in no_decay
                ],
                "weight_decay": 0,
                "lr": self.trainer_config.lr,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if n not in no_decay
                ],
                "weight_decay": self.trainer_config.weight_decay,
                "lr": self.trainer_config.lr,
            },
        ]
        return optimizer_grouped_parameters

    def get_logits(self, batch, return_loss=False):
        input_ids = batch["ids"].to(self.device)
        attention_mask = batch["mask"].to(self.device)
        # logging.debug(f"attention_mask: {attention_mask}")
        if return_loss:
            labels = batch["targets"].to(self.device)
            # logging.debug(f"labels: {labels.tolist()}")
            # logging.debug(f"ids: {input_ids.tolist()}")
            logits, loss = self.model(input_ids, attention_mask, labels=labels)
            return logits, loss
        else:
            logits, _ = self.model(input_ids, attention_mask)
            return logits
    
    def get_loss(self, batch):
        _, loss = self.get_logits(batch, return_loss=True)
        return loss
    
    def step(self, batch):
        if self.trainer_config.freeze_step != -1 and self.num_step % self.trainer_config.freeze_step == 0:
            self.freeze((self.num_step // self.trainer_config.freeze_step) % 2)
        loss = self.get_loss(batch)
        loss /= self.trainer_config.opt_step
        self.num_step += 1
        if self.trainer_config.fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        if self.trainer_config.fgm:
            self.fgm.attack()
            loss_fgm = self.get_loss(batch)
            if self.trainer_config.fp16:
                with amp.scale_loss(loss_fgm, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_fgm.backward()
            self.fgm.restore()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.trainer_config.max_norm)
        if self.num_step % self.trainer_config.opt_step == 0:
            self.optimizer.step()
            if self.trainer_config.ema:
                self.ema.update()
            self.scheduler.step()
            self.optimizer.zero_grad()
        return loss.cpu()

    @torch.no_grad()
    def eval(self, valid_datasets, collate):
        if self.trainer_config.ema:
            self.ema.apply_shadow()
        self.model.eval()
        valid_iter = torch.utils.data.DataLoader(valid_datasets, batch_size=self.trainer_config.valid_batch_size, collate_fn=collate)
        preds = []
        for batch in valid_iter:
            preds.append(self.get_logits(batch).cpu().numpy())
        # logging.debug(f"preds: {preds}")
        f1 = self.metrics(preds, valid_datasets)
        logging.info("Valid F1: {:.4f}".format(f1))
        if self.f1_maxn < f1:
            self.f1_maxn = f1
            self.save()
        del valid_iter
        gc.collect()
        self.model.train()
        if self.trainer_config.ema:
            self.ema.restore()

    def metrics(self, preds, valid_datasets):
        submission = Utils.fetch_submission(preds, valid_datasets.samples)
        logging.info(submission.head(10))
        scr = Utils.score_feedback_comp(submission, valid_datasets.valid_df, return_class_scores=False)
        del submission
        gc.collect()
        return scr

    def save(self):
        if self.trainer_config.debug:
            return
        torch.save(self.model.state_dict(), self.trainer_config.model_save)

    def model_load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, test_iter):
        self.model.eval()
        logits_list = []
        for batch in test_iter:
            logits_list.append(self.get_logits(batch).view(-1).cpu())
        logits = torch.cat(logits_list, dim=-1)
        return logits.view(-1).tolist()

    def set_training_size(self, sz):
        self.training_size = self.trainer_config.epoch * sz // self.trainer_config.opt_step

    def freeze(self, layer):
        name_list = [f"layer.{i + layer}" for i in range(0, 24, 2)]
        for k, v in self.model.named_parameters():
            flag = True
            for name in name_list:
                if name in k:
                    flag = False
            v.requires_grad = flag
    
    def should_eval(self):
        if self.num_step % self.trainer_config.eval_step == 0:
            return True
        return False


class Predicter(BaseTrainer):
    def __init__(self, args):
        super(Predicter, self).__init__(args)

    def set_pretrain(self, pretrain_path):
        self.model_config.pretrain_path = pretrain_path

    @torch.no_grad()
    def predict(self, valid_datasets, collate):
        self.model.eval()
        valid_iter = torch.utils.data.DataLoader(valid_datasets, batch_size=self.trainer_config.valid_batch_size, collate_fn=collate)
        # valid_iter = torch.utils.data.DataLoader(valid_datasets, batch_size=2, collate_fn=collate)
        preds = []
        PAD = torch.tensor([0.0] * 14 + [0.1], dtype=torch.float).unsqueeze(0)
        for batch in tqdm(valid_iter):
            with AMP.autocast(enabled=True):
                pred = self.get_logits(batch).cpu()
                bs, length, dim = pred.shape
                batch_pad = torch.cat([PAD] * bs, dim=0).unsqueeze(1)
                pred = torch.cat([pred] + [batch_pad] * (self.trainer_config.fix_length - length), dim=1)
                preds.append((pred).data.numpy())
        del valid_iter
        gc.collect()
        return preds

    @torch.no_grad()
    def predict_max_tokens(self, valid_datasets):
        self.model.eval()
#         self.model.zero_grad(set_to_none=True)
        valid_iter = torch.utils.data.DataLoader(
            valid_datasets, batch_size=1, 
#             num_workers=2,
            collate_fn=lambda x: x[0],
#             pin_memory=True,
        )
        preds = []
        PAD = torch.tensor([0.0] * 14 + [1.0], dtype=torch.float).unsqueeze(0)
        for batch in tqdm(valid_iter):
#             print(batch)
            with AMP.autocast(enabled=True):
                pred = self.get_logits(batch).cpu()
                bs, length, dim = pred.shape
                batch_pad = torch.cat([PAD] * bs, dim=0).unsqueeze(1)
                pred = torch.cat([pred] + [batch_pad] * (self.trainer_config.fix_length - length), dim=1)
                preds.append((pred*255).byte().data.cpu().numpy())
        del valid_iter
        gc.collect()
        return preds