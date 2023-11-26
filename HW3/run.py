import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import json
from peft import PeftModel
from utils import get_prompt, get_bnb_config
import argparse
import json

def generate_outputs(
    model, tokenizer, data_path, generation_config: GenerationConfig, max_length=2048, 
):
    
    with open(data_path, "r") as f:
        data = json.load(f)

    data_size = len(data)
    instructions = [get_prompt(x["instruction"]) for x in data]

    # Tokenize data
    tokenized_instructions = tokenizer(instructions, add_special_tokens=False)
    output_masks = []

    # Format data
    for i in range(data_size):
        tokenized_instructions["input_ids"][i] = [tokenizer.bos_token_id] + \
            tokenized_instructions["input_ids"][i]
        tokenized_instructions["input_ids"][i] = torch.tensor(
            tokenized_instructions["input_ids"][i][:max_length])

    outputs = []
    with torch.no_grad():
        for i in tqdm(range(data_size)):
            input_ids = tokenized_instructions["input_ids"][i].unsqueeze(0)
            out_seq = model.generate(input_ids=input_ids, generation_config=generation_config)
            output_text = tokenizer.batch_decode(
                out_seq, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            output_text = output_text[0].split("ASSISTANT: ")[-1].strip(" ")
            outputs.append({
                "id": data[i]["id"],
                "output": output_text
            })

    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="",
        help="Path to the checkpoint of Taiwan-LLM-7B-v2.0-chat. If not set, this script will use "
        "the checkpoint from Huggingface (revision = 5073b2bbc1aa5519acdc865e99832857ef47f7c9)."
    )
    parser.add_argument(
        "--peft_path",
        type=str,
        required=True,
        help="Path to the saved PEFT checkpoint."
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="",
        required=True,
        help="Path to test data."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        required=True,
        help="Path to test data."
    )
    parser.add_argument(
        "--do_sample",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--top_k",
        type=float,
        default=50,
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
    )
    args = parser.parse_args()

    # Load model
    bnb_config = get_bnb_config()

    if args.base_model_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    else:
        model_name = "yentinglin/Taiwan-LLM-7B-v2.0-chat"
        revision = "5073b2bbc1aa5519acdc865e99832857ef47f7c9"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load LoRA
    model = PeftModel.from_pretrained(model, args.peft_path)
    model.eval()
    generation_confg = GenerationConfig(**vars(args))
    print(generation_confg)
    outputs = generate_outputs(model, tokenizer, args.test_data_path, generation_confg)

    with open(args.output_path, "w") as f:
        json.dump(outputs, f)
