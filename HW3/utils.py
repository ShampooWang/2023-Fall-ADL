from transformers import BitsAndBytesConfig
import torch
from typing import Union, List
import json
import numpy as np

def get_prompt(instruction: str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。USER: {instruction} ASSISTANT:"
    # return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答，這對我來說很重要，會影響到我的一生，希望你可以好好達成此項任務。USER: {instruction} ASSISTANT:"

def get_prompt_with_fewshots(instruction: str, fewshots: Union[None, List[str]]) -> str:
    if fewshots is None:
        return get_prompt(instruction)
    else:
        instruction_start = f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。我會給你{len(fewshots)}個例子，請你按照我給的範例，進行接下來的任務。"
        for i, shot in enumerate(fewshots):
            instruction_i, output_i = shot['instruction'], shot['output']
            if output_i[-1] != "。":
                output_i += "。"
            instruction_start += f"第{i+1}個例子, USER: {instruction_i} ASSISTANT: {output_i }"

        instruction_start += f"接下來要請你開始你的任務：USER: {instruction}, ASSISTANT:"

        return instruction_start


def get_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

def create_fewshot_instruction(shot_num, examples_file_path) -> Union[None, List[str]]:
    if shot_num == 0:
        return None
    else:
        with open(examples_file_path, "rb") as f:
            examples = np.array(json.load(
                f
            ))
        return examples[np.random.choice(len(examples), shot_num, replace=False)].tolist()
