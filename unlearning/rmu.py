import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'


CONFIG = {
    "model_id":      "meta-llama/Llama-3.1-8B-Instruct",
    "access_token":  os.environ.get("HF_TOKEN"),
    "loss_type":     "gd",

    "lr":            1e-4,
    "batch_size":    2,
    "gradient_accumulation_steps": 2,
    "weight_decay":  0.01,
    "max_length":    512,
    "max_steps":     200,
    "save_steps":    20,
    "hf_user":   os.environ.get("HF_USER", "<YOUR_HF_USERNAME>"),

    # --- All (dataset, selection) combos to run ---
    "runs": [
        {"dataset": "bio",  "selection": "nnomp"},
        {"dataset": "bio",  "selection": "emb"},
        {"dataset": "bio",  "selection": "raslik"},
        {"dataset": "muse", "selection": "nnomp"},
        {"dataset": "muse", "selection": "emb"},
        {"dataset": "muse",  "selection": "raslik"}
    ],

    # --- Path templates ({dataset}, {selection}, {loss_type} are filled in automatically) ---
    "forget_path":   "./data/{dataset}/{selection}_forget.parquet",
    "retain_path":   "./data/{dataset}/{selection}_retain.parquet",
    "adaptor_path":  os.environ.get("HF_USER", "<YOUR_HF_USERNAME>") + "/llama_{dataset}",
    "save_dir":      "./outputs/unlearning/{loss_type}_{dataset}_{selection}_model",
}


import sys
import torch
import pandas as pd
from datetime import datetime
from types import SimpleNamespace
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import  LoraConfig, get_peft_model, PeftModel
from data_module import DualDatasetRandom
from collators import rmu_collator
from accelerate import Accelerator
import pandas as pd
from template import LLAMA3_CHAT_TEMPLATE
from rmu_utils import RMUTrainer
import gc
from tabulate import tabulate



accelerator = Accelerator()

cfg = Config_rmu()

metrics = [
     ("Selection type", f'{cfg.selection}'),
     ("Dataset", f'{cfg.dataset}'),
    ("Unlearning Loss", f'{cfg.loss_type}'),
    ("Forget Path",   f'{cfg.forget_path}'),
    ("Retain Path",   f'{cfg.retain_path}'),
    ("Number of Steps",   f'{cfg.max_steps}'),
]

print("\n\n============ Check List ============\n")
print(tabulate(metrics, headers=["Metric", "Value"], tablefmt="github"))


# ------- loading the datafiles-------------

def read_file(path):
    if path.endswith('.csv'):
        df = pd.read_csv(path)
    elif path.endswith('.json'):
        df = pd.read_json(path)
    elif path.endswith('.parquet'):
        df = pd.read_parquet(path)
    elif path.endswith('.jsonl'):
        df = pd.read_json(path, lines=True)    
    return df

print('loading the forget, retain')
forget = read_file(cfg.forget_path)
retain = read_file(cfg.retain_path)


print(f"\nLoading the Tokenizer {cfg.model_id}")
tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, token = cfg.access_token)
tokenizer.pad_token = '<|finetune_right_pad_id|>'

# ------- Load the model ----------------
print(f"\nLoading the Model {cfg.model_id}")
base_model = AutoModelForCausalLM.from_pretrained(cfg.model_id, 
                                             dtype = torch.bfloat16, 
                                             token=cfg.access_token,
                                             device_map = 'auto')
peft_model = PeftModel.from_pretrained(
    base_model,
    cfg.adaptor_path,
    is_trainable=True,
    device_map="auto",
    dtype=torch.bfloat16,
)

model = peft_model.merge_and_unload()

del base_model
gc.collect()
torch.cuda.empty_cache()

config = LoraConfig(
        r = 8,
        lora_alpha = 16,
        lora_dropout= 0.05,
        target_modules = ['v_proj', 'k_proj', 'o_proj', 'q_proj'],
        bias = 'none',
        task_type = 'CAUSAL_LM',
        layers_to_transform=[23, 24, 25, 26]
    )

print(f"{config.target_modules}")

# ------- wrapping the model with the LoRA configuration
model = get_peft_model(model, config)
model.print_trainable_parameters()
model.config.use_cache = False

# ------- creating template format for tokenization --------
def make_template_format(df):
    df['question'] = df['question'].apply(lambda x : LLAMA3_CHAT_TEMPLATE.format(question = x))
    return df

forget = make_template_format(forget)
retain = make_template_format(retain)
print('forget question and answer\n',forget['question'][0], forget['answer'][0])
print('\n\nretain question and answer\n',retain['question'][0], retain['answer'][0])

m = model
while hasattr(m, "module"): m = m.module

trainables = [(n, p.requires_grad) for n,p in m.named_parameters() if "lora" in n]
print("Total LoRA params:", len(trainables))
print("LoRA trainables:", [n for n,req in trainables if req][:20])
print("Any trainable? ", any(req for _,req in trainables))


# ------- Training Arguments ---------
training_args = TrainingArguments(
    output_dir = cfg.save_dir,
    learning_rate = cfg.lr,
    per_device_train_batch_size= cfg.batch_size, 
    max_steps = cfg.max_steps,
    save_strategy= 'steps',
    save_steps= cfg.save_steps,
    weight_decay = cfg.weight_decay,
    logging_dir = f'{cfg.save_dir}/logs',
    eval_strategy= 'no',
    label_names = ['labels'],
    bf16 = True,
    gradient_accumulation_steps= cfg.gradient_accumulation_steps,
    ddp_find_unused_parameters=False,
)


dataset = DualDatasetRandom(forget_data = forget, 
                          retain_data = retain, 
                          tokenizer = tokenizer, 
                          max_length=cfg.max_length)
print('\nlength of the dataset',len(dataset))

trainer = RMUTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=None,
    data_collator=rmu_collator,  
    # RMU knobs:
    module_regex=r".*layers\.26$",
    trainable_params_regex=r".*layers\.(23|24|25|26)\.self_attn\.(q|k|v|o)_proj\.lora_[AB]\.default\.weight$",
    steering_coeff=20.0,
    alpha=1.0,
    gamma=1.0,
)


trainer.train()

accelerator.wait_for_everyone()
model.save_pretrained(cfg.save_dir)
tokenizer.save_pretrained(cfg.save_dir)
print(f'\nForget LoRA adapter saved at {cfg.save_dir}')