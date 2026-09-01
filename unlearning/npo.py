# 1. export CUDA_VISIBLE_DEVICES=3,5
# 2. accelerate launch --multi_gpu --num_processes 2 preference.py

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from npo_utils import *
from data_module import *
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from accelerate import  Accelerator
from config import Config_npo
import torch
from peft import PeftModel
import pandas as pd
from template import LLAMA3_CHAT_TEMPLATE
from tabulate import tabulate
from collators import custom_gd_collator_forget



cfg = Config_npo()

accelerator = Accelerator()

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


# --- Load tokenizer ---
print(f"\nLoading the Tokenizer {cfg.model_id}")
tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, token = cfg.access_token)
tokenizer.pad_token = tokenizer.eos_token

# --- Load policy model ---
# we load it on cpu, let accelerate move it to GPU with accelerate.prepare_model
base_model = AutoModelForCausalLM.from_pretrained(
    cfg.model_id,
    torch_dtype=torch.bfloat16,
    token=cfg.access_token
    )
print("Base model loaded.")

# --- Apply LoRA on policy model ---

print(f"\nLoading the LoRA adapter from {cfg.adaptor_path}")
policy_model = PeftModel.from_pretrained(
    base_model,
    cfg.adaptor_path,
    torch_dtype=torch.bfloat16,
    is_trainable = True
)

policy_model.config.use_cache = False
policy_model.print_trainable_parameters()

# --- Load reference model ---
ref_model_base = AutoModelForCausalLM.from_pretrained(
    cfg.model_id,
    torch_dtype=torch.bfloat16,
    token=cfg.access_token
)
ref_model = PeftModel.from_pretrained(
    ref_model_base,
    cfg.adaptor_path,
    torch_dtype=torch.bfloat16,
)


def make_template_format(df):
    df['question'] = df['question'].apply(lambda x : LLAMA3_CHAT_TEMPLATE.format(question = x))
    return df

forget = make_template_format(forget)
retain = make_template_format(retain)
print('forget question and answer\n',forget['question'][0],forget['answer'][0])
print('\n\nretain question and answer\n',retain['question'][0],retain['answer'][0])


# ---- Training args ----
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
    remove_unused_columns=False,
)


train_dataset = DualDatasetRandom(forget_data = forget,
                                                retain_data = retain,
                                                tokenizer = tokenizer,
                                                max_length = 512,
                                                question_key='question',
                                                answer_key='answer',
                                                )
print('\n\nlength of the dataset', len(train_dataset))
trainer = RetainNPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        beta=0.1,
        data_collator = custom_gd_collator_forget,
        gamma = 1.0,
        alpha = 1.0,
)


trainer.train()

accelerator.wait_for_everyone()

policy_model.save_pretrained(cfg.save_dir)
tokenizer.save_pretrained(cfg.save_dir)
print(f'\nForget LoRA adapter saved at {cfg.save_dir}')