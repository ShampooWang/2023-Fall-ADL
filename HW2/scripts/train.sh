#!/bin/bash
#SBATCH --job-name=do_sample
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account="MST110260"
#SBATCH --partition=gp1d
#SBATCH --output=/work/jgtf0322/Homework/2023-Fall-ADL/HW2/logs/do_sample.log

cd /work/jgtf0322/Homework/2023-Fall-ADL/HW2

python run_summarization.py \
    --model_name_or_path google/mt5-small \
    --do_train \
    --do_eval \
    --summary_column "title" \
    --text_column "maintext" \
    --train_file "dataset/train.jsonl" \
    --validation_file "dataset/public.jsonl"\
    --output_dir "exp/do_sample" \
    --do_sample true \
    --per_device_train_batch_size 4 \
    --learning_rate 3e-5 \
    --num_train_epochs 10 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --overwrite_output_dir \
    --predict_with_generate true
    