#!/bin/bash
#SBATCH --job-name=ppl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account="MST110260"
#SBATCH --partition=gp1d
#SBATCH --output=/work/jgtf0322/Homework/2023-Fall-ADL/HW3/logs/ppl.log

cd /work/jgtf0322/Homework/2023-Fall-ADL/HW3

python ppl.py \
    --base_model_path /work/jgtf0322/Homework/2023-Fall-ADL/HW3/Taiwan-LLM-7B-v2.0-chat \
    --peft_path "/work/jgtf0322/Homework/2023-Fall-ADL/HW3/exp/qlora/checkpoint-1000/adapter_model" \
    --test_data_path "/work/jgtf0322/Homework/2023-Fall-ADL/HW3/dataset/public_test.json" \
    