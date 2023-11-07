#!/bin/bash
#SBATCH --job-name=run
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account="MST110260"
#SBATCH --partition=gp1d
#SBATCH --output=/work/jgtf0322/Homework/2023-Fall-ADL/HW2/logs/run.log

python run_summarization.py \
    --model_name_or_path ./submit_ckpt \
    --do_predict \
    --text_column "maintext" \
    --output_dir ./submit_ckpt \
    --test_file ${1} \
    --output_file ${2} \
    --predict_with_generate true \
    --per_device_eval_batch_size 1 \
    --num_beams 3 \
    
    