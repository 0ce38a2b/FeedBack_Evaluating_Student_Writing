#!/bin/sh
# source ~/.bashrc
# source activate apex

ROOT="/users10/hsheng/opt/tiger/feedback"
export PYTHONPATH="$HOME/opt/tiger/feedback"


python ../src/params_test.py \
# > ../log/params_test.log 2>&1 &