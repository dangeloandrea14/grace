<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/grace-overview-dark.svg"/>
    <img src="assets/grace-overview.svg" width="900" alt="GRACE: an unlearning request yields a forget direction; GRACE selects a compact forget coreset with non-negative orthogonal matching pursuit and a retain coreset by projecting the forget direction out and clustering; both are passed to a retain-aware unlearning method, which removes the target knowledge while preserving model utility."/>
  </picture>
</p>

<p align="center">
  <sub>Given only a handful of undesired outputs, GRACE builds <b>both</b> the forget set and the retain set.</sub>
</p>

<h1 align="center">GRACE: GRAdient-guided Coreset sElection for LLM Unlearning</h1>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/></a>
  <a href="#citation"><img src="https://img.shields.io/badge/Findings_of_EMNLP-2026-red.svg" alt="Findings of EMNLP 2026"/></a>
  <a href="https://arxiv.org/abs/2608.28361"><img src="https://img.shields.io/badge/arXiv-2608.28361-b31b1b.svg" alt="arXiv"/></a>
</p>

<p align="center">
  <b>GRACE</b> is a machine unlearning framework that selectively removes harmful or private knowledge from large language models while preserving their general utility. It combines influence functions with a <b>Non-Negative Orthogonal Matching Pursuit (NNOMP)</b> algorithm to identify the minimal forget set that maximally degrades target knowledge, then applies state-of-the-art unlearning methods to erase it.
</p>

---

## Citation

If you use GRACE in your research, please cite:

```bibtex
@misc{bushipaka2026gracegradientguidedcoresetselectionllm,
      title={GRACE:Gradient-guided Coreset Selection for LLM Unlearning}, 
      author={Praveen Bushipaka and Andrea D'Angelo and Lucia Passaro and Tommaso Cucinotta},
      year={2026},
      eprint={2608.28361},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2608.28361}, 
}
```

---

## Installation

> **Recommended:** [uv](https://github.com/astral-sh/uv) for fast, reproducible environments.

```bash
# 1. Install uv (skip if already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create a virtual environment
uv venv /path/to/your/venv --python 3.10

# 3. Activate
source /path/to/your/venv/bin/activate

# 4. Install dependencies
uv sync
```

<details>
<summary>Using conda instead</summary>

```bash
conda create -n nnomp python=3.10
conda activate nnomp
pip install -e .
```

</details>

### Environment Variables

The framework accesses gated HuggingFace models and optionally pushes results to the Hub. Set these before running:

```bash
export HF_TOKEN="hf_..."              # HuggingFace read token (required for gated models)
export HF_USER="your_hf_username"     # HuggingFace username (required for push_to_hub)
export DEEPSEEK_API_KEY="sk-..."      # Only needed for LLM-as-a-judge evaluation (laaj.py)
```

---

## Quick Start

Before Unlearning and selection, we fine tune on our datasets. We suggest you to at least fine tune a model on both the datasets (llama + bio, llama + muse). This will give you two adaptors in your HF repo. 

```bash
python finetune.py --model_name llama --dataset bio
python finetune.py --model_name llama --dataset muse
python finetune.py --model_name qwen  --dataset bio
python finetune.py --model_name qwen  --dataset muse
```
The full pipeline runs in **5 steps**. All the configs are available in `configs/`.

### Step 1 — Cache Gradients

Compute and cache per-sample gradients using the RASLIK influence module. Run one config per dataset × split:

```bash
# Create gradient directories first
mkdir -p gradients/{bio_llama,muse_llama,poison/bio_llama,poison/muse_llama} 

export CUDA_VISIBLE_DEVICES=0,1   # adjust to your setup

# Bio (remaining + poison splits)
python MP_main.py --config_path ./configs/caching_bio.json
python MP_main.py --config_path ./configs/caching_bio_p.json

# MUSE (remaining + poison splits)
python MP_main.py --config_path ./configs/caching_muse.json
python MP_main.py --config_path ./configs/caching_muse_p.json
```

> Each config controls data paths, output directories, model, LoRA adapter, and RapidGrad settings. See the **Configuration Reference** below.

### Step 2 — Extract the Forget Set and Retain Set (GRACE)

Run the GRACE selection algorithm on cached gradients to identify training samples that most influence the model's knowledge of target content:

```bash
python grace_bio.py --model_name llama
python grace_muse.py --model_name llama
```

Selected forget sets and retain sets are saved as parquet files in `./data/`.

### Step 3 — Unlearning

Apply one of four unlearning methods. Each `run_*.py` script iterates over all dataset × selection combinations automatically:

```bash
cd unlearning/

# Gradient Difference
python run_gd.py

# Negative Preference Optimization
python run_npo.py

# Representation Misdirection Unlearning
python run_rmu.py

# Simplified NPO
python run_simnpo.py
```

Checkpoints are saved to `./outputs/`.

### Step 4 — Evaluation

Evaluate forget efficacy and retain utility across all trained models:

```bash
cd eval/

# single loss type
python eval.py --model_name llama --loss_type snpo

# multiple loss types in one run
python eval.py --model_name qwen --loss_type gd npo rmu snpo

# baseline only
python eval.py --model_name llama --loss_type pre_unlearning

# baseline + a loss type together
python eval.py --model_name llama --loss_type pre_unlearning snpo gd
```

For LLM-as-a-judge scoring (requires `DEEPSEEK_API_KEY`):

```bash
python laaj.py
```

Statistical significance testing across methods:

```bash
python statistical_test.py
```

---

## Configuration Reference

All pipeline behavior is controlled via JSON configs in `configs/`. Key fields:

| Field | Location | Description |
|-------|----------|-------------|
| `data.train_data_path` | all configs | Path to training JSONL (remaining/poison split) |
| `influence.grads_path` | caching configs | Directory to save/load cached gradients |
| `influence.outdir` | retrieval config | Directory for influence score outputs |
| `influence.top_k` | retrieval config | Number of top-ranked samples to select |
| `influence.RapidGrad.enable` | all configs | Enable random-projection dimensionality reduction |
| `influence.RapidGrad.RapidGrad_K` | all configs | Projection dimension (default: 65536) |
| `influence.skip_test` | caching configs | Skip test-gradient computation (set true when only caching) |
| `influence.skip_influence` | caching configs | Skip influence computation (set true when only caching) |
| `model.model_path` | all configs | HuggingFace model ID or local path |
| `model.lora_path` | all configs | HuggingFace LoRA adapter ID (format: `<YOUR_HF_USERNAME>/adapter_name`) |
| `model.load_in_4bit` | all configs | Enable 4-bit quantization for low-VRAM setups |
| `postprocess.enable` | retrieval config | Automatically export forget/retain JSONL after influence computation |
| `postprocess.top_n` | retrieval config | Number of samples in exported forget/retain sets |

### Example: Caching Config (`configs/caching_bio.json`)

```json
{
    "data": { "train_data_path": "./data/bio_remaining.jsonl" },
    "influence": {
        "save_to_grads_path": true,
        "skip_test": true,
        "skip_influence": true,
        "grads_path": "./gradients/bio_llama3/"
    },
    "model": {
        "model_path": "meta-llama/Llama-3.2-1B-Instruct",
        "lora_path": "<YOUR_HF_USERNAME>/llama3_bio"
    },
    "postprocess": { "enable": false }
}
```

---

## Supported Models & Datasets

### Models

| Model | Identifier |
|-------|-----------|
| LLaMA 3.2 1B Instruct | `meta-llama/Llama-3.2-1B-Instruct` |
| LLaMA 3.1 8B Instruct | `meta-llama/Llama-3.1-8B-Instruct` |
| Qwen 2.5 3B Instruct | `Qwen/Qwen2.5-3B-Instruct` |

### Datasets

| Dataset | Domain | Split used |
|---------|--------|-----------|
| [WMDP-Bio](https://www.wmdp.ai/) | Biosecurity | `bio_remaining.jsonl`, `bio_poison.jsonl` |
| [MUSE](https://muse-bench.github.io/) | Cybersecurity | `muse_remaining.jsonl`, `muse_poison.jsonl` |

### Unlearning Methods

| Method | Script | Description |
|--------|--------|-------------|
| GD | `run_gd.py` | Gradient Difference on forget set |
| NPO | `run_npo.py` | Negative Preference Optimization |
| RMU | `run_rmu.py` | Representation Misdirection Unlearning |
| SimNPO | `run_simnpo.py` | Simplified NPO variant |

---

## Project Structure

```
grace/
├── MP_main.py              # Entry point: gradient caching + influence computation
├── RASLIK/                 # Influence function module
│   ├── engine.py           # Multiprocessing orchestration
│   ├── influence_function.py
│   ├── calc_inner.py       # Gradient/s-test computations
│   ├── data_loader.py      # Model & dataset loading
│   └── RapidGrad.py        # Random projection for efficient influence
├── unlearning/             # Unlearning method implementations
│   ├── run_{gd,npo,rmu,simnpo}.py   # Runnable training scripts
│   ├── {gd,npo,rmu,simnpo}.py       # Method logic
│   └── {data_module,collators,forget_trainer,template}.py
├── eval/                   # Evaluation scripts
│   ├── eval_auto*.py       # Automated multi-run evaluation
│   ├── eval_utils.py
│   └── laaj.py             # LLM-as-a-judge pipeline
├── configs/                # JSON config files
├── data/                   # Forget/retain/poison splits + eval sets
├── assets/                 # animated README banner + its generator
├── grace_{bio,muse}.py     # GRACE forget/retain selection
├── finetune.py             # LoRA fine-tuning script
└── statistical_test.py     # Friedman + Nemenyi significance tests
```


---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
