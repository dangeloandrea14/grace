# to run the script, use the command: 
# 1. export CUDA_VISIBLE_DEVICES=1,3
# 2. Comment os.environ['CUDA_VISIBLE_DEVICES'] = '1,2' in the script since we are setting it from the command line
# 3. accelerate launch --multi_gpu --num_processes 2 gd.py

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2' 

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from config import Config_gd
from peft import PeftModel
from data_module import DualDatasetRandom
from collators import custom_gd_collator_forget
from forget_trainer import GradDiffTrainer
from accelerate import Accelerator
import pandas as pd
from template import LLAMA3_CHAT_TEMPLATE
from tabulate import tabulate


accelerator = Accelerator()

class Config_gd:
    def __init__(self):
        super(Config_gd, self).__init__()
        self.loss_type      = 'gd' # change this with the experiment types provided above
        self.model_id       = 'meta-llama/Llama-3.1-8B-Instruct' 
        self.LoRA_r         = 8
        self.LoRA_alpha     = 16
        self.LoRA_dropout   = 0.05
        self.lr             = 1e-04
        self.LoRa_targets   = ['v_proj', 'k_proj', 'o_proj', 'q_proj']
        self.batch_size     = 2
        self.gradient_accumulation_steps = 2 #always batch size of 8
        self.num_epochs     = 4
        self.overwrite_dir  = True
        self.weight_decay   = 0.01
        self.max_length     = 512
        self.selection       ='ours'
        self.dataset       = 'bio'
        self.adaptor_path    = f'./outputs/llama_bio' # this will be set based on the dataset in the code below
        self.save_dir       = f'./outputs/unlearning/{self.loss_type}_{self.dataset}_{self.selection}_model' 
        self.forget_path    = './data/our_method/bio_10c/bio_forget.parquet'
        self.retain_path    = './data/our_method/bio_10c/bio_retain.parquet'
        self.max_steps      = 200
        self.save_steps     = 20



cfg = Config_gd()

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


# if cfg.dataset == 'bio':
#     cfg.adaptor_path = '/home/user/grace/outputs/llama_bio'
# elif cfg.dataset == 'muse':
#     cfg.adaptor_path = '/home/user/grace/outputs/llama_muse'


# ------- Load the tokenizer ----------------
print(f"\nLoading the Tokenizer {cfg.model_id}")
tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
tokenizer.pad_token = tokenizer.eos_token



# ------- Load the model ----------------
print(f"\nLoading the Model {cfg.model_id}")
base_model = AutoModelForCausalLM.from_pretrained(
    cfg.model_id,
    torch_dtype=torch.bfloat16,
    device_map = 'auto',
)

print(f"\nLoading the LoRA adapter from {cfg.adaptor_path}")
model = PeftModel.from_pretrained(
    base_model,
    cfg.adaptor_path,
    is_trainable=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

model.config.use_cache = False
model.print_trainable_parameters()


# ------- creating template format for tokenization --------
def make_template_format(df):
    df['question'] = df['question'].apply(lambda x : LLAMA3_CHAT_TEMPLATE.format(question = x))
    return df

def normalize_columns(df):
    df = df.rename(columns={'prompt': 'question', 'generation': 'answer'})
    return df

forget = make_template_format(forget)
retain = make_template_format(retain)
print('forget question and answer\n',forget['question'][0],forget['answer'][0])
print('\n\nretain question and answer\n',retain['question'][0],retain['answer'][0])



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


# ------- dataset and training args for the standard gradient difference method -----

dataset = DualDatasetRandom(forget_data = forget,
                            retain_data = retain,
                            tokenizer = tokenizer,
                            max_length = cfg.max_length)
print('\nlength of the dataset',len(dataset))

  # ------- dataset for the gradient ascent method ----- 

trainer = GradDiffTrainer(
        model = model,
        args = training_args,
        train_dataset = dataset,
        processing_class = tokenizer,
        data_collator = custom_gd_collator_forget,
        )

trainer.train()

accelerator.wait_for_everyone()
model.save_pretrained(cfg.save_dir)
tokenizer.save_pretrained(cfg.save_dir)
print(f'\nForget LoRA adapter saved at {cfg.save_dir}')





