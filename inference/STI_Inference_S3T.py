import json
import torch
import pandas as pd
from src import reconstruct_instruction
from eval.utils import load_hf_lm_and_tokenizer, generate_completions
from eval.templates import create_prompt_with_tulu_chat_format 
import argparse

parser = argparse.ArgumentParser(description="Run the script with a specified model and batch size.")
parser.add_argument("--model_name", type=str, required=True, help="Name of the model to load")
parser.add_argument("--batch_size", type=int, required=True, help="Batch size for generation")

args = parser.parse_args()

with open("data/MTI_BENCH.json", 'r') as file:
    data = json.load(file)
# data = {k:v for k,v in data.items() if k in ["002","012","015","016","018"]}

model, tokenizer = load_hf_lm_and_tokenizer(
                args.model_name,
                torch_dtype = torch.float16
                )

CoT = pd.read_excel("data/cot_breakdown.xlsx")

print("starting evaluation")
for k,v in list(data.items()):
    print(k)
    _,s1,s2,s3 = CoT[CoT["tuid"]==int(k)].values[0]
    cot = data[k]["sample"]

    input_list =[
    create_prompt_with_tulu_chat_format([
    {"role":"user", 
     "content":  
     "### Example:\n\n" + "### Instruction: " +cot + \
    "\n\n### Task:\n\n" +  "### Instruction: " + reconstruct_instruction(instance,1,False) + "\n\n" + "### Answer:\n\n"}
    ])
    for uid,instance in data[k]["instance"].items()]
    
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
        "uid":list(data[k]["instance"].keys()),
        "generation":generated_texts_s1,
        "generation_time":generation_time
    }).to_csv(f"data/{args.model_name}-MTI-{k}-s1.csv")
    
    torch.cuda.empty_cache()

    
    input_list = []
    input_list_idx = []
    for (uid,instance),gen1 in zip(data[k]["instance"].items(),generated_texts_s1):
        query1 = reconstruct_instruction(instance,1,False)
        query2 = reconstruct_instruction(instance,2,False) 

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
        input_list_idx.append(uid)

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
        "uid":list(data[k]["instance"].keys()),
        "generation":generated_texts_s2,
        "generation_time":generation_time
    }).to_csv(f"data/{args.model_name}-MTI-{k}-s2.csv")
    
    torch.cuda.empty_cache()

    
    input_list = []
    input_list_idx = []
    for (uid,instance),gen1,gen2 in zip(data[k]["instance"].items(),generated_texts_s1,generated_texts_s2):
        query1 = reconstruct_instruction(instance,1,False)
        query2 = reconstruct_instruction(instance,2,False) 
        query3 = reconstruct_instruction(instance,3,False) 

        query = "### Example:\n\n" + \
            "### (1) Instruction: " + s1 + \
            "\n\n### (2) Instruction: " + s2 + \
            "\n\n### (3) Instruction: " + s3 + \
            "\n\n### Task:\n\n" +  \
            "### (1) Instruction: " + query1 +\
            "\n\n### Answer:\n\n" + str(gen1) + \
            "\n\n### (2) Instruction: " + query2 +\
            "\n\n### Answer: \n\n" + str(gen2) + \
            "\n\n### (3) Instruction: " + query3 +\
            "\n\n### Answer:"
        query = create_prompt_with_tulu_chat_format([{"role":"user", "content": query}])

        input_list.append(query)
        input_list_idx.append(uid)

    generation_time, generated_texts_s3 = generate_completions(
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
        "uid":list(data[k]["instance"].keys()),
        "generation":generated_texts_s3,
        "generation_time":generation_time
    }).to_csv(f"data/{args.model_name}-MTI-{k}-s3.csv")
    
    del generated_texts_s1
    del generated_texts_s2
    del generated_texts_s3
    
    torch.cuda.empty_cache()

