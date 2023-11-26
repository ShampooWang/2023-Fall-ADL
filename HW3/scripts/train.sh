#!/bin/bash
#SBATCH --job-name=qlora
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account="MST110260"
#SBATCH --partition=gp4d
#SBATCH --output=/work/jgtf0322/Homework/2023-Fall-ADL/HW3/logs/qlora.log

cd /work/jgtf0322/Homework/2023-Fall-ADL/HW3

python qlora.py \
    --model_name_or_path Taiwan-LLM-7B-v2.0-chat \
    --do_train \
    --report_to "wandb" \
    --train_file "dataset/train.json" \
    --validation_file "dataset/public_test.json"\
    --output_dir "exp/qlora" \
    --per_device_train_batch_size 4 \
    --learning_rate 3e-5 \
    --max_train_epochs 3 \
    --evaluation_strategy "steps" \
    --eval_steps "100" \
    --gradient_accumulation_steps 4 \
    --overwrite_output_dir \
    