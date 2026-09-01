import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True, choices=["llama", "qwen"],
                    help="Model to use: 'llama' (meta-llama/Llama-3.1-8B-Instruct) or 'qwen' (Qwen/Qwen2.5-3B-Instruct)")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

MODEL_IDS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen":  "Qwen/Qwen2.5-3B-Instruct",
}

CONFIG = {
    "model_id":      MODEL_IDS[args.model_name],
    "model_name":    args.model_name,
    "access_token":  os.environ.get("HF_TOKEN"),
    "loss_type":     "snpo",
    "lr":            1e-4,
    "batch_size":    2,
    "gradient_accumulation_steps": 4,
    "weight_decay":  0.01,
    "max_length":    512,
    "max_steps":     200,
    "save_steps":    20,
    "delta":         0.0,
    "beta":          3.5,
    "hf_user":       os.environ.get("HF_USER", "<YOUR_HF_USERNAME>"),

    # --- All (dataset, selection) combos to run ---
    "runs": [
        {"dataset": "bio",  "selection": "nnomp"},
        {"dataset": "bio",  "selection": "emb"},
        {"dataset": "bio",  "selection": "raslik"},
        {"dataset": "muse", "selection": "nnomp"},
        {"dataset": "muse", "selection": "emb"},
        {"dataset": "muse", "selection": "raslik"},
    ],

    # --- Path templates ({dataset}, {selection}, {model_name}, {loss_type} are filled in automatically) ---
    "forget_path":  "./data/{dataset}/{model_name}_{selection}_forget.parquet",
    "retain_path":  "./data/{dataset}/{model_name}_{selection}_retain.parquet",
    "adaptor_path": "{hf_user}/{model_name}_{dataset}",
    "save_dir":     "./outputs/{model_name}_unlearning/{loss_type}_{dataset}_{selection}_model",
}

import sys
import torch
import pandas as pd
from datetime import datetime
from types import SimpleNamespace
from tabulate import tabulate
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import PeftModel
from accelerate import Accelerator
from data_module import DualDatasetRandom
from collators import custom_gd_collator_forget
from snpo_utils import SimNPO
from template import LLAMA3_CHAT_TEMPLATE, qwen_chat_template
import gc

CHAT_TEMPLATES = {
    "llama": LLAMA3_CHAT_TEMPLATE,
    "qwen":  qwen_chat_template,
}


def resolve(cfg, dataset, selection):
    fmt = dict(
        dataset=dataset,
        selection=selection,
        loss_type=cfg["loss_type"],
        model_name=cfg["model_name"],
        hf_user=cfg["hf_user"],
    )
    ns = SimpleNamespace(**cfg)
    ns.dataset      = dataset
    ns.selection    = selection
    ns.forget_path  = cfg["forget_path"].format(**fmt)
    ns.retain_path  = cfg["retain_path"].format(**fmt)
    ns.adaptor_path = cfg["adaptor_path"].format(**fmt)
    ns.save_dir     = cfg["save_dir"].format(**fmt)
    return ns


def read_file(path):
    ext = os.path.splitext(path)[1]
    if ext == ".csv":     return pd.read_csv(path)
    if ext == ".json":    return pd.read_json(path)
    if ext == ".parquet": return pd.read_parquet(path)
    if ext == ".jsonl":   return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported format: {path}")


def apply_template(df, model_name):
    template = CHAT_TEMPLATES[model_name]
    df = df.copy()
    df["question"] = df["question"].apply(lambda x: template.format(question=x))
    return df


def run_single(cfg):
    accelerator = Accelerator()

    print(tabulate([
        ("Model name",   cfg.model_name),
        ("Model ID",     cfg.model_id),
        ("Dataset",      cfg.dataset),
        ("Selection",    cfg.selection),
        ("Loss type",    cfg.loss_type),
        ("Forget path",  cfg.forget_path),
        ("Retain path",  cfg.retain_path),
        ("Adaptor path", cfg.adaptor_path),
        ("Save dir",     cfg.save_dir),
        ("Max steps",    cfg.max_steps),
    ], headers=["Key", "Value"], tablefmt="github"))

    print("\nLoading datasets...")
    forget = apply_template(read_file(cfg.forget_path), cfg.model_name)
    retain = apply_template(read_file(cfg.retain_path), cfg.model_name)

    print(f"Loading tokenizer: {cfg.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, token=cfg.access_token)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {cfg.model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        token=cfg.access_token,
        device_map="auto",
    )

    print(f"Loading LoRA adapter: {cfg.adaptor_path}")
    model = PeftModel.from_pretrained(
        base_model,
        cfg.adaptor_path,
        is_trainable=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=cfg.save_dir,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        max_steps=cfg.max_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        weight_decay=cfg.weight_decay,
        logging_dir=f"{cfg.save_dir}/logs",
        eval_strategy="no",
        label_names=["labels"],
        bf16=True,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        ddp_find_unused_parameters=False,
    )

    dataset = DualDatasetRandom(
        forget_data=forget,
        retain_data=retain,
        tokenizer=tokenizer,
        max_length=cfg.max_length,
    )

    trainer = SimNPO(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=custom_gd_collator_forget,
    )

    trainer.train()
    accelerator.wait_for_everyone()
    model.save_pretrained(cfg.save_dir)
    tokenizer.save_pretrained(cfg.save_dir)
    print(f"\nSaved to: {cfg.save_dir}")

    repo_id = f"{cfg.hf_user}/{cfg.model_name}_{cfg.loss_type}_{cfg.dataset}_{cfg.selection}_model"
    print(f"\nPushing to HuggingFace Hub: {repo_id}")
    model.push_to_hub(repo_id, token=cfg.access_token)
    tokenizer.push_to_hub(repo_id, token=cfg.access_token)
    print(f"Pushed to: https://huggingface.co/{repo_id}")

    del model, base_model, tokenizer, trainer, dataset
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    runs   = CONFIG["runs"]
    failed = []

    print(f"\n{'='*52}")
    print(f"  Model   : {CONFIG['model_name']} ({CONFIG['model_id']})")
    print(f"  Runs    : {len(runs)}  — {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'='*52}")

    for i, run in enumerate(runs, 1):
        dataset, selection = run["dataset"], run["selection"]
        print(f"\n[{i}/{len(runs)}] dataset={dataset}  selection={selection}")
        print("─" * 52)
        try:
            run_single(resolve(CONFIG, dataset, selection))
            print(f"[{i}/{len(runs)}] done.")
        except Exception as e:
            msg = f"dataset={dataset}  selection={selection} — {e}"
            print(f"FAILED: {msg}")
            failed.append(msg)

    print(f"\n{'='*52}")
    if failed:
        print(f"  {len(failed)} run(s) failed:")
        for f in failed:
            print(f"    • {f}")
        sys.exit(1)
    else:
        print(f"  All {len(runs)} runs completed")
    print(f"{'='*52}\n")