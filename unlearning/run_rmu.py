import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True, choices=["llama", "qwen"],
                    help="Model to use: 'llama' (meta-llama/Llama-3.1-8B-Instruct) or 'qwen' (Qwen/Qwen2.5-3B-Instruct)")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

MODEL_IDS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen":  "Qwen/Qwen2.5-3B-Instruct",
}

# RMU layer config differs by model architecture
RMU_LAYERS = {
    "llama": {
        "layers_to_transform": [22, 23, 24, 25, 26, 27, 28],
        "module_regex":         r".*layers\.28$",
        "trainable_params_regex": (
            r".*layers\.(22|23|24|25|26|27|28)"
            r"\.self_attn\.(q|k|v|o)_proj\.lora_[AB]\.default\.weight$"
        ),
    },
    "qwen": {
        "layers_to_transform": [22, 23, 24, 25, 26, 27, 28],
        "module_regex":         r".*layers\.28$",
        "trainable_params_regex": (
            r".*layers\.(22|23|24|25|26|27|28)"
            r"\.self_attn\.(q|k|v|o)_proj\.lora_[AB]\.default\.weight$"
        ),
    },
}

CONFIG = {
    "model_id":      MODEL_IDS[args.model_name],
    "model_name":    args.model_name,
    "access_token":  os.environ.get("HF_TOKEN"),
    "loss_type":     "rmu",
    "lr":            1e-4,
    "batch_size":    2,
    "gradient_accumulation_steps": 4,
    "weight_decay":  0.01,
    "max_length":    512,
    "max_steps":     200,
    "save_steps":    20,
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
from peft import LoraConfig, get_peft_model, PeftModel
from accelerate import Accelerator
from data_module import DualDatasetRandom
from collators import rmu_collator
from template import LLAMA3_CHAT_TEMPLATE, qwen_chat_template
from rmu_utils import RMUTrainer
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
    rmu_cfg = RMU_LAYERS[cfg.model_name]

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

    print(f"Merging LoRA adapter: {cfg.adaptor_path}")
    peft_model = PeftModel.from_pretrained(
        base_model,
        cfg.adaptor_path,
        is_trainable=False,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = peft_model.merge_and_unload()
    del peft_model
    gc.collect()
    torch.cuda.empty_cache()

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["v_proj", "k_proj", "o_proj", "q_proj"],
        bias="none",
        task_type="CAUSAL_LM",
        layers_to_transform=rmu_cfg["layers_to_transform"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.config.use_cache = False

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
    print(f"\nDataset length: {len(dataset)}")

    trainer = RMUTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=rmu_collator,
        module_regex=rmu_cfg["module_regex"],
        trainable_params_regex=rmu_cfg["trainable_params_regex"],
        steering_coeff=20.0,
        alpha=1.0,
        gamma=1.0,
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