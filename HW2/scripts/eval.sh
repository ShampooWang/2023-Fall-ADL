#!/bin/bash
#SBATCH --job-name=do_sample_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account="MST110260"
#SBATCH --partition=gp1d
#SBATCH --output=/work/jgtf0322/Homework/2023-Fall-ADL/HW2/logs/do_sample_eval.log

cd /work/jgtf0322/Homework/2023-Fall-ADL/HW2

python run_summarization.py \
    --model_name_or_path /work/jgtf0322/Homework/2023-Fall-ADL/submit_ckpt \
    --do_eval \
    --summary_column "title" \
    --text_column "maintext" \
    --validation_file "dataset/public.jsonl"\
    --output_dir "exp/do_sample" \
    --do_sample true \
    --predict_with_generate true
    