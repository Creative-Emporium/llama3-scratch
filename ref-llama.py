import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#load the model
model_hf = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", device_map='auto')

#look at the model parameteres 
llama_sd = model_hf.state_dict()

with open("./llama_params.txt", 'w') as f:
    for k,v in llama_sd.items():
        f.write(f"{k} {v.shape}\n")

print('done!')

