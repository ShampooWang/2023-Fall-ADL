# 2023 Fall Applied deep learning HW1

## 0. Install packages

- Python 3.9
- Pytorch 1.12.1

After install above packages, install `requirements.txt` by

```bash
pip install -r requirements.txt
```

## 1. Train paragraph selection model

```bash
python /path/to/run_swag_no_trainer.py \
    --seed your_seed \
    --context_file /path/to/context.json \
    --train_file /path/to/train.json \
    --validation_file /path/to/valid.json \
    --model_name_or_path your_model \
    --max_seq_length 512 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 3e-5 \
    --output_dir /path/to/your/saving/directory \
```

## 2. Train question answering model

```bash
python /path/to/run_qa_no_trainer.py \
    --seed your_seed \
    --context_file /path/to/context.json \
    --train_file /path/to/train.json \
    --validation_file /path/to/valid.json \
    --model_name_or_path your_model \
    --max_seq_length 512 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 3e-5 \
    --output_dir /path/to/your/saving/directory \
```