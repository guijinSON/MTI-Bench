from eval.utils import load_hf_lm_and_tokenizer, generate_completions
from eval.templates import create_prompt_with_tulu_chat_format
import numpy as np
import re
import torch
import json
import pandas as pd
import time
from src import reconstruct_batch_instruction
import argparse


parser = argparse.ArgumentParser(description="Run the script with a specified model and batch size.")
parser.add_argument("--model_name", type=str, required=True, help="Name of the model to load")
parser.add_argument("--batch_size", type=int, required=True, help="Batch size for generation")


args = parser.parse_args()

    
with open("data/MTI_BENCH.json", 'r') as file:
    data = json.load(file)
# data = {k:v for k,v in data.items() if k in ["002","012","015","016","018"]}
        
CoT = pd.read_excel("data/cot_breakdown.xlsx")
print("starting download")

model, tokenizer = load_hf_lm_and_tokenizer(
            args.model_name,#"NousResearch/Llama-2-13b-chat-hf",# "TheBloke/tulu-7B-fp16",
            torch_dtype = torch.float16
            )


for k,v in list(data.items()):
    print(k)
    _,s1,s2,s3 = CoT[CoT["tuid"]==int(k)].values[0]
    cot = data[k]["sample"]
    instruction_s1, instruction_s2, instruction_s3, paired_keys = reconstruct_batch_instruction(data[k]["instance"])
    
    input_list =[
    create_prompt_with_tulu_chat_format([
    {"role":"user", "content":  "### Example:\n\n" + "### Instruction: " + s1 + \
    "\n\n### Task:\n\n" + inst +"\n\n" + "### Answer:\n\n"}
    ])
    for inst in instruction_s1]
    
    generation_time, generated_texts_s1 = generate_completions(
                        model,
                        tokenizer,
                        input_list,
                        batch_size=args.batch_size,
                        stop_id_sequences=None,
                        add_special_tokens=True,
                        disable_tqdm=False,
                        max_new_tokens=2048,
                        min_new_tokens=32,
                        do_sample=True,
                        temperature=0.7,
                        top_p=1.0
        
                    )
    pd.DataFrame({
        "uid": paired_keys,
        "generation":generated_texts_s1,
        "generation_time":generation_time
    }).to_csv(f"data/{args.model_name}-Batching-{k}-s1.csv")
    
    torch.cuda.empty_cache()
    
    input_list = []
    for query1,gen1,query2 in zip(instruction_s1,generated_texts_s1,instruction_s2):
        query = "### Example:\n\n" + \
            "### (1) Instruction: " + s1 + \
            "\n\n### (2) Instruction: " + s2 + \
            "\n\n### Task:\n\n" +  \
            "### (1) Instruction: " + query1 +\
            "\n\n### Answer:\n\n" + str(gen1) + \
            "\n\n### (2) Instruction: " + query2 +\
            "\n\n### Answer:"
        query = create_prompt_with_tulu_chat_format([{"role":"user", "content": query}])
        input_list.append(query)
    
    generation_time, generated_texts_s2 = generate_completions(
                            model,
                            tokenizer,
                            input_list,
                            batch_size=args.batch_size,
                            stop_id_sequences=None,
                            add_special_tokens=True,
                            disable_tqdm=False,
                            max_new_tokens=2048,
                            min_new_tokens=32,
                            do_sample=True,
                            temperature=0.7,
                            top_p=1.0
                        )
    
    pd.DataFrame({
        "uid": paired_keys,
        "generation":generated_texts_s2,
        "generation_time":generation_time
    }).to_csv(f"data/{args.model_name}-Batching-{k}-s2.csv")
    
    torch.cuda.empty_cache()
    
