import os
from typing import Dict, Optional, Sequence
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import copy
import torch
import logging
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from peft import PeftModel, set_peft_model_state_dict, prepare_model_for_kbit_training
from peft import get_peft_model, LoraConfig, TaskType
import random
import numpy as np
import transformers

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
prompt_no_input = \
    "Below is an instruction that describes a task. " \
    "Write a response that appropriately completes the request.\n\n" \
    "### Instruction:\n{instruction}\n\n### Response: "

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    tokenizer,
    model,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def get_model_tokenizer(config, **kwargs):
    tokenizer = get_tokenizer(config, **kwargs)
    model = get_model(config, tokenizer, **kwargs)
    return model, tokenizer


def get_model(config, tokenizer=None, **kwargs):
    device_map = kwargs.get("device_map", None)
    model_path = config.model_path
    lora_path = config.lora_path
    logging.warning("Loading model...")
    model = None
    bnb_config = None

    if config.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    if device_map is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=False,
            token=os.environ.get("HF_TOKEN"),
            torch_dtype = torch.bfloat16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map=device_map,
            token=os.environ.get("HF_TOKEN"),
            torch_dtype = torch.bfloat16
        )

    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    model.enable_input_require_grads()
    
    model = PeftModel.from_pretrained(model, lora_path, is_trainable=True, device_map=device_map)

    model.print_trainable_parameters()  # will now show ~1-2%

    model.config.use_cache = False
    model.is_parallelizable = True
    model.model_parallel = True
    model.train()
    return model

def get_tokenizer(config, **kwargs):
    model_path = config.model_path
    logging.warning("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.max_length = config.max_length
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_dataset_size(data_path):
    content = None
    with open(data_path) as f:
        content = f.readlines()
    return len(content)

def read_data(data_path):
    list_data_dict = None
    with open(data_path) as f:
        list_data_dict = [json.loads(line) for line in f]
    return list_data_dict


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=512,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len - 1] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels, input_ids_lens=sources_tokenized["input_ids_lens"])


class TrainDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, shuffle: bool = True, shuffle_seed: int = 42, load_idx_list = None, begin_id = None, end_id = None):
        super(TrainDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = read_data(data_path)

        logging.warning("Formatting inputs...")

        sources = [example["prompt"] for example in list_data_dict]
        targets = [example["generation"] + tokenizer.eos_token for example in list_data_dict]
        print(f"sources: {len(sources)}, targets: {len(targets)}")

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)
        logging.warning("Done tokenizing inputs...")

        if load_idx_list is None:
            load_idx_list = list(range(len(data_dict["input_ids"])))

        s = list(range(len(load_idx_list)))
        if shuffle == True:
            random.seed(shuffle_seed)
            random.shuffle(s)

        self.input_ids = [ data_dict["input_ids"][i] for i in s ]
        self.sorted_index = [ load_idx_list[i] for i in s ]
        self.list_data_dict = [ list_data_dict[i] for i in s ]
        self.labels = [ data_dict["labels"][i] for i in s ]
        self.input_ids_lens = [ data_dict["input_ids_lens"][i] for i in s ]
        self.json_ids = [item['id'] for item in self.list_data_dict]


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return self.input_ids[i], self.labels[i], self.input_ids_lens[i], self.json_ids[i]


class TestDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(TestDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = []
        if data_path is not None and len(data_path) != 0:
            list_data_dict = read_data(data_path)

        logging.warning("Formatting inputs...")
        sources = []
        targets = []

        for example in list_data_dict:
            if "prompt" in example and "generation" in example:
                prompt = example["prompt"].strip()
                generation = example["generation"].strip()
            # elif "instruction" in example and "output" in example:
            #     prompt = f"### Instruction:\n{example['instruction'].strip()}\n"
            #     if "input" in example and example["input"].strip():
            #         prompt += f"### Input:\n{example['input'].strip()}\n"
            #     prompt += "### Response:"
            #     generation = example["output"].strip()
            else:
                raise ValueError("Unsupported example format. Please include either ('prompt', 'generation') or ('instruction', 'output').")

            sources.append(prompt)
            targets.append(f"{generation}{tokenizer.eos_token}")

        
        if list_data_dict and "hotwords" in list_data_dict[0]:
            hotwords = [
                [hw.strip() for hw in example.get("hotwords", "").split('|') if hw != ""]
                for example in list_data_dict
            ]
        else:
            hotwords = [[] for _ in list_data_dict]


        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        print(f"Detected hotwords: {hotwords}")
        self.labels = []
        for hotwords_list, label_tokens in zip(hotwords, data_dict["labels"]):
            if len(hotwords_list) == 0:
                self.labels.append(label_tokens)
                continue
            label_tokens = label_tokens.tolist()
            hotwords_tokens = [tokenizer.encode(x, add_special_tokens=False) for x in hotwords_list]
            new_label = [-100 for _ in range(len(label_tokens))]
            label_tokens_len = len(label_tokens)
            for hotword in hotwords_tokens:
                hotword_len = len(hotword)
                for i in range(label_tokens_len):
                    if i + hotword_len > label_tokens_len:
                        break
                    if hotword == label_tokens[i:i + hotword_len]:
                        new_label[i:i + hotword_len] = label_tokens[i:i + hotword_len]
            self.labels.append(torch.LongTensor(new_label))

        self.list_data_dict = list_data_dict
        self.input_ids = data_dict["input_ids"]
        self.input_ids_lens = data_dict["input_ids_lens"]
        self.hotwords = hotwords

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self.input_ids[i], self.labels[i], self.input_ids_lens[i]
