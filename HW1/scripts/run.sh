#!/bin/bash

cd /work/jgtf0322/Homework/2023-Fall-ADL/hw1

python inference.py \
    --seed 322 \
    --context_file ${1} \
    --test_file ${2} \
    --output_csv ${3} \
    --multiple_choice_model "/work/jgtf0322/Homework/2023-Fall-ADL/hw1/exp/paragraph_selection/chinese-lert-base" \
    --question_answering_model "/work/jgtf0322/Homework/2023-Fall-ADL/hw1/exp/question-answering/chinese-pert-large" \
    --per_device_test_batch_size 1 \
    --seed 322 \
    