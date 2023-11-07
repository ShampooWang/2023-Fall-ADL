#!/bin/bash
#SBATCH --job-name=qa_chinese-lert-large
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account="MST110260"
#SBATCH --partition=gp1d
#SBATCH --output=/work/jgtf0322/Homework/2023-Fall-ADL/hw1/logs/qa/chinese-pert-large.log

cd /work/jgtf0322/Homework/2023-Fall-ADL/hw1

python run_qa_no_trainer.py \
    --seed 322 \
    --context_file dataset/context.json \
    --train_file dataset/train.json \
    --validation_file dataset/valid.json \
    --model_name_or_path hfl/chinese-pert-large \
    --max_seq_length 512 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 6 \
    --gradient_accumulation_steps 2 \
    --learning_rate 3e-5 \
    --output_dir exp/question-answering/chinese-pert-large \
    --with_tracking \
    --report_to "wandb" \
    
    
    