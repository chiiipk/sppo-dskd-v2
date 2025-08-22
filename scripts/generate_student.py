import argparse
import json
import os
import random
import warnings

import numpy as np
import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate responses from a Student Model.")
    parser.add_argument("--model", type=str, required=True, help="Path or name of the student model on the Hugging Face Hub.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the response files.")
    parser.add_argument("--prompts", type=str, required=True, help="Path to a .jsonl file or a dataset name on the Hugging Face Hub.")
    parser.add_argument("--pairs", type=int, default=5, help="Number of candidate responses to generate for each prompt.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for text generation.")
    parser.add_argument("--maxlen", type=int, default=512, help="Maximum length")
    
    parser.add_argument("--frac_len", type=int, required=True, help="The length of each data fraction for a single GPU.")
    parser.add_argument("--data_frac", type=int, required=True, help="The index of the data fraction this process will handle.")
    
    return parser.parse_args()

def split_data(prompts, frac_len, data_frac):
    if frac_len <= 0:
        return prompts
    
    start_index = frac_len * data_frac
    end_index = frac_len * (data_frac + 1)
    
    if start_index >= len(prompts):
        return [] 
    
    return prompts[start_index:end_index]

def main():
    args = parse_arguments()

    print(f"--- Starting generation process for GPU with data_frac={args.data_frac} ---")
    print(f"Model: {args.model}")
    print(f"Number of candidates: {args.pairs}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(device)
        model.eval()
    except Exception as e:
        print(f"{e}")
        return

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    print(f"Loading prompts data from: {args.prompts}")
    if args.prompts.endswith('.jsonl'):
        data = Dataset.from_json(args.prompts)
    else:
        data = load_dataset(args.prompts, split="train")

    # student don't need complicated template
    prompts = [item['prompt'] for item in data]
    prompts_for_this_gpu = split_data(prompts, args.frac_len, args.data_frac)
    
    if not prompts_for_this_gpu:
        print(f"No prompts were assigned for data_frac={args.data_frac}. End.")
        return
        
    print(f"This GPU will process {len(prompts_for_this_gpu)} prompts.")

    for p in range(args.pairs):
        # change seeds
        current_seed = (p + 1) * (args.data_frac + 1) * 100
        set_seed(current_seed)
        
        print(f"\n--- Starting generation for candidate set {p + 1}/{args.pairs} (seed={current_seed}) ---")
        
        all_responses = []
        for i in tqdm(range(0, len(prompts_for_this_gpu), args.batch_size), desc=f"Generating pair {p+1}"):
            batch_prompts = prompts_for_this_gpu[i : i + args.batch_size]
            
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
            
            generation_kwargs = {
                "max_new_tokens": args.maxlen,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "pad_token_id": tokenizer.eos_token_id,
            }
            
            with torch.no_grad():
                outputs = model.generate(**inputs, **generation_kwargs)
            
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            responses_only = [
                decoded[len(prompt):] for prompt, decoded in zip(batch_prompts, decoded_outputs)
            ]
            
            all_responses.extend(responses_only)


        output_path = f"{args.output_dir}/responses_{args.data_frac}_{p}.json"
        print(f"Generated {len(all_responses)} responses. Saving to: {output_path}")
        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(all_responses, f, ensure_ascii=False, indent=2)

    print(f"\n--- Completed all generation loops for GPU with data_frac={args.data_frac} ---")

if __name__ == "__main__":
    main()