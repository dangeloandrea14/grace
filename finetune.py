import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True, choices=["llama", "qwen"],
                    help="'llama' (meta-llama/Llama-3.1-8B-Instruct) or 'qwen' (Qwen/Qwen2.5-3B-Instruct)")
parser.add_argument("--dataset", type=str, required=True, choices=["bio", "muse"],
                    help="Dataset to fine-tune on: 'bio' or 'muse'")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator

from unlearning.template import LLAMA3_CHAT_TEMPLATE
from unlearning.data_module import SingleDataset
from unlearning.collators import custom_data_collator

# ─────────────────────────────────────────────
# REGISTRIES
# ─────────────────────────────────────────────
MODEL_IDS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen":  "Qwen/Qwen2.5-3B-Instruct",
}

# Bio path is shared; muse differs by model
DATA_PATHS = {
    ("llama", "bio"):  "./data/wmdp_bio.parquet",
    ("qwen",  "bio"):  "./data/wmdp_bio.parquet",
    ("llama", "muse"): "./data/muse_data.parquet",
    ("qwen",  "muse"): "./data/muse_data_qwen.parquet",
}

LLAMA3_CHAT_TEMPLATE_STR = LLAMA3_CHAT_TEMPLATE   

QWEN_CHAT_TEMPLATE = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""

CHAT_TEMPLATES = {
    "llama": LLAMA3_CHAT_TEMPLATE_STR,
    "qwen":  QWEN_CHAT_TEMPLATE,
}

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
model_name   = args.model_name
dataset      = args.dataset
model_id     = MODEL_IDS[model_name]
data_path    = DATA_PATHS[(model_name, dataset)]
access_token = os.environ.get("HF_TOKEN")
hf_user      = os.environ.get("HF_USER", "<YOUR_HF_USERNAME>")
save_dir     = f"./outputs/{model_name}_{dataset}"

# LoRA — slightly wider r for bio (more forgetting signal), narrower for muse
LORA_R = {
    "bio":  16,
    "muse": 8,
}

cfg = {
    "model_id":    model_id,
    "model_name":  model_name,
    "dataset":     dataset,
    "data_path":   data_path,
    "access_token": access_token,
    "save_dir":    save_dir,
    "LoRA_r":      LORA_R[dataset],
    "LoRA_alpha":  32,
    "LoRA_dropout": 0.05,
    "LoRA_targets": ["v_proj", "k_proj", "o_proj", "q_proj"],
    "lr":           1e-4,
    "batch_size":   2,
    "gradient_accumulation_steps": 16,
    "num_epochs":   10,
    "weight_decay": 0.01,
    "max_length":   512,
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def read_file(path):
    if path.endswith(".csv"):     return pd.read_csv(path)
    if path.endswith(".json"):    return pd.read_json(path)
    if path.endswith(".parquet"): return pd.read_parquet(path)
    if path.endswith(".jsonl"):   return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported format: {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print(f"Model      : {model_name} ({model_id})")
    print(f"Dataset    : {dataset}  ({data_path})")
    print(f"Save dir   : {save_dir}\n")

    os.makedirs(save_dir, exist_ok=True)

    accelerator = Accelerator()

    # ── Data ──────────────────────────────────────────────────────
    data = read_file(cfg["data_path"])
    print(f"Loaded data: {data.shape}")

    template = CHAT_TEMPLATES[model_name]
    data["question"] = data["question"].apply(lambda x: template.format(question=x))
    print(f"\nSample prompt:\n{data['question'][0]}\n")

    # ── Tokenizer ─────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)

    if model_name == "llama":
        # Llama has a dedicated right-pad token; using eos causes label leakage
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
    else:
        # Qwen does not have a dedicated pad token
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model ─────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=access_token,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=cfg["LoRA_r"],
        lora_alpha=cfg["LoRA_alpha"],
        lora_dropout=cfg["LoRA_dropout"],
        target_modules=cfg["LoRA_targets"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Dataset ───────────────────────────────────────────────────
    dataset_obj = SingleDataset(data, tokenizer, max_length=cfg["max_length"])

    # ── Training ──────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=save_dir,
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_epochs"],
        learning_rate=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        bf16=True,
        save_strategy="epoch",
        save_total_limit=10,
        logging_dir=f"{save_dir}/logs",
        eval_strategy="no",
        gradient_checkpointing=False,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_obj,
        processing_class=tokenizer,
        data_collator=custom_data_collator,
    )

    trainer.train()
    accelerator.wait_for_everyone()

    # ── Save ──────────────────────────────────────────────────────
    model.cpu()
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"\nSaved model and tokenizer to {save_dir}")

    # ── Push to Hub ───────────────────────────────────────────────
    repo_id = f"{hf_user}/{model_name}_{dataset}"
    print(f"Pushing to HuggingFace Hub: {repo_id}")
    try:
        model.push_to_hub(repo_id, token=access_token)
        tokenizer.push_to_hub(repo_id, token=access_token)
        print(f"Pushed to: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Could not push to hub: {e}")


if __name__ == "__main__":
    main()