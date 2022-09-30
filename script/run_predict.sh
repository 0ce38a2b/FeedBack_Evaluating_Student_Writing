#!/bin/sh
# source ~/.bashrc
# source activate feedback

ROOT="/users10/hsheng/opt/tiger/feedback"
export PYTHONPATH="$HOME/opt/tiger/feedback"


python ../src/Base.py \
--predict \
--train_path="$ROOT/data/train_fold5.csv" \
--fold=3 \
--valid_batch_size=4 \
--fix_length=1600 \
> ../log/Base_prediction.log 2>&1 &