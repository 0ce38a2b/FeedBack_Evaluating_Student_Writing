#!/bin/sh
# source ~/.bashrc
# source activate feedback

ROOT="/users10/lyzhang/opt/tiger/feedback"
export PYTHONPATH="$HOME/opt/tiger/feedback"


python ../src/Base.py \
--train \
--debug \
--train_path="$ROOT/data/train_fold10.csv" \
--pretrain_path="$HOME/model/deberta_v2_xxlarge" \
--tokenizer_path="$HOME/model/deberta_v2_xxlarge" \
--model_save="$ROOT/model/LF" \
--fold=0 \
--epoch=150 \
--lr=1e-5 \
--min_lr=1e-6 \
--Tmax=2500 \
--valid_batch_size=8 \
--train_batch_size=1 \
--fix_length=1600 \
--fgm \
--ema \
--deberta \
--scheduler="get_cosine_schedule_with_warmup" \
> ../log/Base_train.log 2>&1 &