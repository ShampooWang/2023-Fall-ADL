#!/bin/bash
#SBATCH --job-name=run
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account="MST110260"
#SBATCH --partition=gp1d
#SBATCH --output=/work/jgtf0322/Homework/2023-Fall-ADL/HW3/logs/run.log

python run.py \
    --base_model_path ${1} \
    --peft_path ${2} \
    --test_data_path ${3} \
    --output_path ${4} \
    
    