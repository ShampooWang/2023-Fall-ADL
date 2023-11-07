# 2023 Fall Applied deep learning HW2

## 0. Train MT5

```bash
python run_summarization.py \
    --model_name_or_path google/mt5-small \
    --do_train \
    --do_eval \
    --summary_column "title" \
    --text_column "maintext" \
    --train_file "path/to/your/train.jsonl" \
    --validation_file "path/to/your/eval.jsonl"\
    --output_dir "path/to/your/output/dir" \
    --num_beams 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 3e-5 \
    --num_train_epochs 10 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --overwrite_output_dir \
    --predict_with_generate true
```

## 1. Inference by the provided model

```bash
bash download.sh
bash run.sh path/to/your/test.jsonl path/to/your/output.jsonl
```