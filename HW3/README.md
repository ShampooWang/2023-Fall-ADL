# 2023 Fall Applied deep learning HW3

## 0. Train QLora Model

```bash
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
```

## 1. Inference by the provided model

```bash
bash download.sh
bash ./run.sh /path/to/Taiwan-LLaMa-folder /path/to/adapter_checkpoint \ 
/path/to/input.json /path/to/output.json
```