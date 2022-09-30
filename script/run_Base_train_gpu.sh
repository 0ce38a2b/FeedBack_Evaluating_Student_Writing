#!/bin/sh
# source ~/.bashrc
# source activate apex

ROOT="/users10/hsheng/opt/tiger/feedback"
export PYTHONPATH="$HOME/opt/tiger/feedback"

# PRETRAIN="funnel_large"              # bs 3 37g fp32
# PRETRAIN="longformer_large_4096"       # bs 5 36g fp16
# PRETRAIN="bigbird_roberta_large"
# PRETRAIN="deberta_v3_large"              # bs 3 34g fp16
# PRETRAIN="deberta_large"
# PRETRAIN="deberta_v2_xlarge"             # bs 1
# PRETRAIN="deberta_xlarge"             # bs 1
# PRETRAIN="bigbird_pegasus_large_arxiv"
PRETRAIN="deberta_v2_xxlarge"

python ../src/Base.py \
--train \
--train_path="$ROOT/data/train_fold10.csv" \
--pretrain_path="$HOME/model/$PRETRAIN" \
--tokenizer_path="$HOME/model/$PRETRAIN" \
--model_save="$ROOT/model/LF" \
--fold=0 \
--epoch=10 \
--lr=1e-5 \
--min_lr=1e-6 \
--Tmax=1000 \
--valid_batch_size=8 \
--train_batch_size=1 \
--opt_step=1 \
--fix_length=1600 \
--scheduler="get_cosine_schedule_with_warmup" \
--dropout=0.1 \
--eval_step=10000000 \
--fp16 \
--deberta \
--weight_decay=1e-2 \
# --deberta \
# --record="5dropout" \
# --train_all \
# --deberta \
# > ../log/Base_train.log 2>&1 &