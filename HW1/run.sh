#!/bin/bash

cd /tmp2/jeffwang/work/Homeworks/2023-Fall-ADL/hw1

python inference.py \
    --seed 322 \
    --context_file ${1} \
    --test_file ${2} \
    --multiple_choice_model "/tmp2/jeffwang/work/Homeworks/2023-Fall-ADL/hw1/exp/paragraph_selection/bert-base-chinese" \
    --per_device_test_batch_size 1 \
    --seed 322 \
    