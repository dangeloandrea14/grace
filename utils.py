import pandas as pd
from unlearning.data_module import  SingleDataset
import json
import os
import torch
import ast
import numpy as np 

def read_file(path):
    if path.endswith('.csv'):
        df = pd.read_csv(path)
    elif path.endswith('.json'):
        df = pd.read_json(path)
    elif path.endswith('.parquet'):
        df = pd.read_parquet(path)
    return df

def coerce_to_list(x):
    """Make sure x is a List[str], handling JSON, Python repr, or delimited strings."""
    if isinstance(x, list):
        return x
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, str):
        s = x.strip()
        # Try JSON first: '["a", "b"]'
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return [str(t) for t in v]
        except Exception:
            pass
        # Then safe Python literal: "['a', 'b']"
        try:
            v = ast.literal_eval(s)
            if isinstance(v, list):
                return [str(t) for t in v]
        except Exception:
            pass
        # Fall back to splitting on common delimiters
        for sep in ['||', '|', ';', '\t', ',']:
            if sep in s:
                return [t.strip() for t in s.split(sep) if t.strip()]
        # Otherwise treat the whole string as a single item
        return [s]
    # Any other scalar -> single-item list
    return [str(x)]

def write_json(data_path, logs):
    with open(data_path, 'w') as f:
        json.dump(logs, f, indent=4)

def read_json(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data


def update_json_dict(data_path, new_results):
    # Check if the file exists; if not, start with an empty dictionary
    if os.path.exists(data_path):
        with open(data_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}
    
    # Expecting new_results to be a dictionary
    if not isinstance(new_results, dict):
        raise ValueError("new_results must be a dictionary when updating a dict-based JSON")
    
    # Merge new_results into the existing data
    data.update(new_results)
    
    # Write the updated dictionary back to the file
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=4)


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)
