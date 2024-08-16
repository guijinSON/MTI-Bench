import requests
import pandas as pd
import json
import numpy as np
import concurrent.futures
from tqdm import tqdm
import time
from openai import OpenAI

client = OpenAI()

def get_sni_json(task_name):
    url = "https://raw.githubusercontent.com/allenai/natural-instructions/master/tasks/<task>.json".replace("<task>",task_name)
    response = requests.get(url)
    data = response.json()
    #return data
    return pd.DataFrame(data["Instances"])

def get_sni_json_wurl(url):
    response = requests.get(url)
    data = response.json()
    #return data
    return pd.DataFrame(data["Instances"])

def search_dataframe_sw(df, query):
    # Create a boolean mask for rows that start with the query in the 'input' column
    mask = df['input'].str.startswith(query)
    
    # Apply the mask to the original data frame to get the filtered rows
    result_df = df[mask]
    
    return result_df

def search_dataframe_co(df, query):
    # Create a boolean mask for rows that include the query in the 'input' column
    mask = df['input'].str.contains(query, regex=False)
    
    # Apply the mask to the original data frame to get the filtered rows
    result_df = df[mask]
    
    return result_df

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

import re

def parse_answer(s, N):
    # Regular expression pattern to capture the content for the specified task number
    pattern = f'<task{N}>(.*?)<task{N}/>'
    match = re.search(pattern, str(s), re.DOTALL)
    
    if match:
        return match.group(1).strip()
    else:
        return "-1210"

def norm_text(text):
    if "/" in text:
        return text
    if text.startswith("("):
        return text
    try: 
        return eval(text)
    except: 
        return text
    
def calculate_accuracy(answer, prediction):
    # Check the length of both lists to ensure they match
    if len(answer) != len(prediction):
        raise ValueError("Answer and prediction must have the same length.")

    # Initialize variables to keep track of accuracy
    step_accuracies = []  # Individual step accuracies
    conditional_accuracies = []  # Conditional accuracies

    # Loop through each step and calculate the accuracy
    all_previous_steps_correct = True
    for i in range(len(answer)):
        # Check if the current step is correct
        current_step_correct = answer[i] == prediction[i]

        # Calculate the step accuracy
        step_accuracy = 1 if current_step_correct else 0
        step_accuracies.append(step_accuracy)

        # Calculate the conditional step accuracy
        conditional_accuracy = step_accuracy if all_previous_steps_correct else 0
        conditional_accuracies.append(conditional_accuracy)

        # Update the flag to check if all previous steps were correct
        all_previous_steps_correct = all_previous_steps_correct and current_step_correct

    return np.array(step_accuracies), np.array(conditional_accuracies)

def get_score(df,data,tuid):
    num_answers = len(list(data[tuid]["instance"].values())[0]["answers"])

    total_step_accuracies, total_conditional_accuracies = np.zeros(num_answers), np.zeros(num_answers)
    for _,row in df.iterrows():
        _ = data[tuid]["instance"][row.uid]
        answers = [str(_) for _ in _["answers"]]
        predict = [str(norm_text(parse_answer(row.generation,idx+1))) for idx in range(len(answers))]
        step_accuracies, conditional_accuracies = calculate_accuracy(answers, predict)

        total_step_accuracies+=step_accuracies
        total_conditional_accuracies+=conditional_accuracies  
    return total_step_accuracies/len(df), total_conditional_accuracies/len(df)


def reconstruct_instruction(data, num, full=True, tf=False):
    if num == -1:
        num = len(data['instruction'])
    # Extract values
    query = data['query']
    instructions = data['instruction']
    contexts = data['context']
    answers = data['answers']

    # Filter instructions and contexts based on the given number
    if full:
        filtered_instructions = '\n'.join([f"#{i}: {instructions[str(i)]}" for i in range(1, num+1) if str(i) in instructions])
        filtered_contexts = '\n'.join([f"{v}" for i in range(1, num+1) for k,v in contexts.items() if k.startswith(str(i))])
        final_string = query + "\n" + filtered_instructions + "\n\n" + filtered_contexts
    else:
        filtered_instructions = instructions[str(num)]
        filtered_contexts = '\n'.join([f"{v}" for k,v in contexts.items() if k.startswith(str(num))])
        final_string = filtered_instructions + "\n\n" + filtered_contexts


    return final_string

    # else:
    #     answer = " ".join([f"Given the context the answer for instruction {i} is {answers[i-1]}. <task{i}>{answers[i-1]}<task{i}/>"for i in range(1,num)])
    #     return final_string + f"\n\n###Answer:\n{answer}"
    

def gpt_multithread(gpt_input_dict):
    MAX_RETRIES = 20
    answers = []
    retries = 0
    while retries < MAX_RETRIES:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = []
                for idx,gpt_input in gpt_input_dict[len(answers):]:
                    future = executor.submit(get_answer, gpt_input,idx)
                    futures.append(future)

                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    answers.append(future.result())
            break  # If successful, break out of the inner retry loop

        except Exception as e:  # Catch-all for unexpected exceptions
            retries += 1
            time.sleep(60) 
    return answers

def get_answer(instance, model): #model: ["gpt-3.5-turbo-0613", "gpt-4-0613"]
    completion = client.chat.completions.create(
                      model=model,
                      messages=[
                        {"role": "user", "content": instances}
                      ],     
                      stop= "### Task:",
                      temperature=0.7,
                    )
    return completion.choices[0].message.content



def get_instances(idx_list,data,CoT):
    input_list = []
    for uid in idx_list:
        instance = data[tuid]["instance"][uid]
        instance = reconstruct_instruction(instance,-1)
        instance = "### Example:\n\n" + CoT + "\n\n### Task:\n\n" + instance
        input_list.append([uid,instance])
    return input_list


def similarity(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Both lists should have the same length.")
    
    matched_positions = sum(a == b for a, b in zip(list1, list2))
    return matched_positions / len(list1)

def similarity_ef(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Both lists should have the same length.")
    
    total_positions = sum(1 for a, b in zip(list1, list2) if a or b)
    matched_positions = sum(a == b for a, b in zip(list1, list2) if a or b)
    
    if total_positions == 0:
        return 1.0  # Return 1 if there are no positions with at least one True value
    
    return matched_positions / total_positions


def reconstruct_instruction_tf(data):
    num = len(data['instruction'])
    query = data['query']
    instructions = data['instruction']
    contexts = data['context']
    answers = data['answers']

    filtered_instructions = '\n'.join([f"#{i}: {instructions[str(i)]}" for i in range(1, num+1) if str(i) in instructions])
    filtered_contexts = '\n'.join([f"{v}" for i in range(1, num+1) for k,v in contexts.items() if k.startswith(str(i))])
    final_string = query + "\n" + filtered_instructions + "\n\n" + filtered_contexts

    step1_answer = f"Given the context the answer for instruction <1> is {answers[0]}. <task1>{answers[0]}<task1/>"
    return final_string, step1_answer
  
    
def reconstruct_batch_instruction(data, lv1=True):
    instruction_list_1 = []
    instruction_list_2 = []
    instruction_list_3 = []
    
    keys = list(data.keys())
    paired_keys = [(keys[i], keys[i+1]) for i in range(0, len(keys)-1, 2)]
    

    for uid_0, uid_1 in paired_keys:
        uid_0_instruction = reconstruct_instruction(data[uid_0],1,False)
        uid_1_instruction = reconstruct_instruction(data[uid_1],1,False)

        uid_1_instruction = uid_1_instruction.replace("<task1>","<task2>").replace("<task1/>","<task2/>")

        final_instruction = f"""Read the following passage, and solve both of the instructions provided.
### Instruction 1: {uid_0_instruction}

### Instruction 2: {uid_1_instruction}"""
        instruction_list_1.append(final_instruction)
        
        ###
        
        uid_0_instruction = reconstruct_instruction(data[uid_0],2,False)
        uid_1_instruction = reconstruct_instruction(data[uid_1],2,False)

        uid_1_instruction = uid_1_instruction.replace("<task1>","<task2>").replace("<task1/>","<task2/>")

        final_instruction = f"""Read the following passage, and solve both of the instructions provided.
### Instruction 1: {uid_0_instruction}

### Instruction 2: {uid_1_instruction}"""
        instruction_list_2.append(final_instruction)
        
        ###
        
        uid_0_instruction = reconstruct_instruction(data[uid_0],3,False)
        uid_1_instruction = reconstruct_instruction(data[uid_1],3,False)

        uid_1_instruction = uid_1_instruction.replace("<task1>","<task2>").replace("<task1/>","<task2/>")

        final_instruction = f"""Read the following passage, and solve both of the instructions provided.
### Instruction 1: {uid_0_instruction}

### Instruction 2: {uid_1_instruction}"""
        instruction_list_3.append(final_instruction)
    return instruction_list_1, instruction_list_2,instruction_list_3, paired_keys
    

def get_batch_score(df,data,tuid,return_id=False):
    df = df.dropna()
    
    s1_score = 0.0
    s2_score = 0.0
    uid_list = []
    for uid,generation in df[["uid","generation"]].values:
        uid = eval(uid)

        # if int(tuid) < 30:
        p1 = str(norm_text(parse_answer(generation,1)))
        p2 = str(norm_text(parse_answer(generation,2)))

        a1 = str(data[tuid]["instance"][uid[0]]["answers"][0])
        a2 = str(data[tuid]["instance"][uid[1]]["answers"][0])

        if a1 == p1:
            s1_score +=1
            uid_list.append(uid[0])
        if a2 == p2:
            s2_score+= 1
            uid_list.append(uid[1])
            
    if return_id:
        return uid_list
    return s1_score/len(df), s2_score/len(df)     

