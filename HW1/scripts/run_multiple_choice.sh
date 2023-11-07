#!/bin/bash
#SBATCH --job-name=mc_chinese-pert-large
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account="MST110260"
#SBATCH --partition=gp1d
#SBATCH --output=/work/jgtf0322/Homework/2023-Fall-ADL/hw1/logs/mc/chinese-pert-large.log

cd /work/jgtf0322/Homework/2023-Fall-ADL/hw1

python run_swag_no_trainer.py \
    --seed 322 \
    --context_file dataset/context.json \
    --train_file dataset/train.json \
    --validation_file dataset/valid.json \
    --model_name_or_path hfl/chinese-pert-large \
    --max_seq_length 512 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 3e-5 \
    --output_dir exp/paragraph_selection/chinese-pert-large \
    --with_tracking \
    --report_to "wandb" \
    
    