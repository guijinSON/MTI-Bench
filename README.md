# Official Repository for MTI-Bench
This is the official repository for paper: MULTI-TASK INFERENCE: Can Large Language Models Follow Multiple Instructions at Once?

## Install
For inferencing you need to install the [open-instruct](https://github.com/allenai/open-instruct) directory. 

```python
git clone https://github.com/allenai/open-instruct
pip install -r requirements.txt
```

## To Reconstruct Multi-Task Instructions
```python
import json
from src import reconstruct_instruction, reconstruct_batch_instruction

with open("data/MTI_BENCH.json") as fp:
    mti = json.load(fp)

### Multi-Task Inference
for tuid,v in mti.items():
    for uid,instance in v["instance"].items():
        print(reconstruct_instruction(instance,-1))
        
### Single-Task Inference
for tuid,v in mti.items():
    for uid,instance in v["instance"].items():
        print(reconstruct_instruction(instance,1,False))
        print("####### Divider #######")
        print(reconstruct_instruction(instance,2,False))

### Batch Prompting
for tuid,v in mti.items():
    batched_instructions = reconstruct_batch_instruction(v["instance"])
    for instance in batched_instructions[0]:
        print(instance)
```
