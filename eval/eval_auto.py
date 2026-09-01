import os
import argparse

VALID_LOSS_TYPES = ["gd", "npo", "rmu", "snpo", "pre_unlearning"]

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True, choices=["llama", "qwen"],
                    help="Model to use: 'llama' (meta-llama/Llama-3.1-8B-Instruct) or 'qwen' (Qwen/Qwen2.5-3B-Instruct)")
parser.add_argument("--loss_type", type=str, required=True, nargs="+",
                    choices=VALID_LOSS_TYPES,
                    help=f"One or more loss types to evaluate. Choices: {VALID_LOSS_TYPES}. "
                         "Example: --loss_type snpo gd npo")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import json
import warnings
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import logging as hf_logging
from peft import PeftModel
from tabulate import tabulate
from sentence_transformers import SentenceTransformer

from eval_utils import compute_mu_scores, compute_fq_scores
from utils import update_json_dict

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────
MODEL_IDS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen":  "Qwen/Qwen2.5-3B-Instruct",
}

LLAMA3_CHAT_TEMPLATE = """\
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 July 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

QWEN_CHAT_TEMPLATE = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""

CHAT_TEMPLATES = {
    "llama": LLAMA3_CHAT_TEMPLATE,
    "qwen":  QWEN_CHAT_TEMPLATE,
}

# ─────────────────────────────────────────────
# GLOBAL CONFIG
# ─────────────────────────────────────────────
MODEL_ID       = MODEL_IDS[args.model_name]
model_name     = args.model_name
requested_loss_types = args.loss_type   # list, e.g. ["snpo", "gd"]
MAX_LENGTH     = 512
BASE_OUTPUT_DIR  = f"./outputs/{model_name}_unlearning"
ADAPTOR_BASE     = os.environ.get("HF_USER", "<YOUR_HF_USERNAME>")
RESULTS_DS_DIR   = "./results/datasets"
RESULTS_SC_DIR   = "./results/scores"
COMBINED_JSONL   = "./results/all_results.jsonl"

FORGET_PATHS = {
    "bio" : "./data/wmdp_bio.parquet",
    "muse": "./data/muse_data.parquet",
}
TEST_PATHS = {
    "bio" : "./data/test_bio.parquet",
    "muse": "./data/test_muse.parquet",
}

EMBEDDING_MODEL_NAME = "thenlper/gte-small"

# ─────────────────────────────────────────────
# EXPERIMENT CONFIGS
# ─────────────────────────────────────────────
DATASETS   = ["bio", "muse"]
SELECTIONS = ["grace", "raslik", "emb"]


def build_experiments(loss_types):
    """
    Build the experiment list for the requested loss types.
    'pre_unlearning' adds one baseline entry per dataset (no selection loop).
    All other loss types add one entry per (dataset, selection) combination.
    """
    experiments = []

    for loss_type in loss_types:
        if loss_type == "pre_unlearning":
            for dataset in DATASETS:
                experiments.append({
                    "loss_type"   : "pre_unlearning",
                    "dataset"     : dataset,
                    "selection"   : "none",
                    "adaptor_path": f"{ADAPTOR_BASE}/{model_name}_{dataset}",
                    "save_dir"    : None,
                    "forget_path" : FORGET_PATHS[dataset],
                    "test_path"   : TEST_PATHS[dataset],
                })
        else:
            for dataset in DATASETS:
                for selection in SELECTIONS:
                    experiments.append({
                        "loss_type"   : loss_type,
                        "dataset"     : dataset,
                        "selection"   : selection,
                        "adaptor_path": None,
                        "save_dir"    : f"{BASE_OUTPUT_DIR}/{loss_type}_{dataset}_{selection}_model",
                        "forget_path" : FORGET_PATHS[dataset],
                        "test_path"   : TEST_PATHS[dataset],
                    })

    return experiments


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def read_file(path):
    if path.endswith(".csv"):     return pd.read_csv(path)
    if path.endswith(".json"):    return pd.read_json(path)
    if path.endswith(".parquet"): return pd.read_parquet(path)
    raise ValueError(f"Unsupported file format: {path}")


def make_template_format(df):
    template = CHAT_TEMPLATES[model_name]
    df = df.copy()
    df["question"] = df["question"].apply(lambda x: template.format(question=x))
    return df


def load_model(exp):
    print(f"  Loading base model: {MODEL_ID}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    if exp["loss_type"] == "pre_unlearning":
        print(f"  Loading pre-unlearning adaptor: {exp['adaptor_path']}")
        model = PeftModel.from_pretrained(
            base_model, exp["adaptor_path"],
            device_map="auto", torch_dtype=torch.bfloat16,
        )
    else:
        print(f"  Loading PEFT adaptor: {exp['save_dir']}")
        model = PeftModel.from_pretrained(
            base_model, exp["save_dir"],
            device_map="auto", torch_dtype=torch.bfloat16,
        )
    return model


def append_to_jsonl(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DS_DIR, exist_ok=True)
    os.makedirs(RESULTS_SC_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model    : {model_name} ({MODEL_ID})")
    print(f"Device   : {device}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    experiments = build_experiments(requested_loss_types)
    print(f"Loss types   : {requested_loss_types}")
    print(f"Total experiments to run: {len(experiments)}\n")

    all_metrics = []

    for idx, exp in enumerate(experiments):
        exp_name = f"{model_name}_{exp['loss_type']}_{exp['dataset']}_{exp['selection']}"
        print(f'\n{"="*60}')
        print(f"[{idx+1}/{len(experiments)}]  {exp_name}")
        print(f'{"="*60}')

        # ── Load data ──────────────────────────────────────────────
        print(f"  Reading forget set: {exp['forget_path']}")
        forget = read_file(exp["forget_path"])
        test   = read_file(exp["test_path"])
        forget = forget.loc[forget["type"] == "forget"].reset_index(drop=True)
        print(f"  Forget set shape: {forget.shape}")

        forget_2 = make_template_format(forget)[["id", "question", "answer", "num_tokens", "type"]]
        test_2   = make_template_format(test)[["id", "question", "answer", "num_tokens", "type"]]

        # ── Load model ─────────────────────────────────────────────
        model = load_model(exp)

        # ── Evaluate ───────────────────────────────────────────────
        print("  Computing forget quality scores ...")
        forget_out, all_scores_fe, fq, f_ppl = compute_fq_scores(
            df=forget_2, model=model, tokenizer=tokenizer, device=device
        )
        print(f"  Forget Quality (FQ): {fq:.4f}  |  PPL-F: {f_ppl:.4f}")

        print("  Computing model utility scores ...")
        test_out, all_scores_mu, mu, mu_ppl = compute_mu_scores(
            df=test_2, model=model, tokenizer=tokenizer,
            embedding_model=embedding_model, device=device
        )
        print(f"  Model Utility  (MU): {mu:.4f}  |  PPL-MU: {mu_ppl:.4f}")

        # ── Save per-experiment parquets ───────────────────────────
        for suffix, df_out in [("forget", forget_out), ("test", test_out)]:
            path = f"{RESULTS_DS_DIR}/{exp_name}_{suffix}.parquet"
            df_out.to_parquet(path, index=False)
            print(f"  Saved → {path}")

        # ── Build & persist result record ──────────────────────────
        result_record = {
            "experiment"    : exp_name,
            "loss_type"     : exp["loss_type"],
            "dataset"       : exp["dataset"],
            "selection"     : exp["selection"],
            "FQ"            : fq.item(),
            "PPL-F"         : f_ppl.item(),
            "forget_scores" : all_scores_fe.tolist(),
            "MU"            : mu.item(),
            "PPL-MU"        : mu_ppl.item(),
            "utility_scores": all_scores_mu.tolist(),
        }

        append_to_jsonl(COMBINED_JSONL, result_record)
        print(f"  Appended → {COMBINED_JSONL}")

        individual_json = f"{RESULTS_SC_DIR}/{exp_name}_results.json"
        update_json_dict(individual_json, {exp_name: result_record})

        all_metrics.append((exp_name, fq.item(), f_ppl.item(), mu.item(), mu_ppl.item()))

        # ── Free GPU memory ────────────────────────────────────────
        del model
        torch.cuda.empty_cache()

    # ── Final summary ──────────────────────────────────────────────
    print("\n\n============ FINAL SUMMARY ============\n")
    print(tabulate(
        all_metrics,
        headers=["Experiment", "FQ", "PPL-F", "MU", "PPL-MU"],
        tablefmt="github",
        floatfmt=".4f",
    ))
    print(f"\nAll results saved to: {COMBINED_JSONL}")


if __name__ == "__main__":
    main()