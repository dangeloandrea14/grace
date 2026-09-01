import torch
from transformers import  default_data_collator
from typing import Tuple, Any
import pandas as pd
from typing import Dict, List, Tuple 



def custom_data_collator(samples): # for vanilla gradient ascent and finetuning
    """
    Collate function for the forget dataset only

    Args:
        samples (list of tuples): Each tuple contains (input_ids, labels, attention_mask)

    Returns:
        dict: batched_inputs, labels, attention_masks.

    """
    input_ids = torch.stack([sample[0] for sample in samples])
    labels = torch.stack([sample[1] for sample in samples])
    attention_mask = torch.stack([sample[2] for sample in samples])
    return {'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask}



def custom_gd_collator_forget(samples): # for vanilla/cyclic gradient difference, also can be extended to dpo, npo type
    """
    Custom data collator for forget and retain data

    Args:
        samples: list of tuples (forget_data, retain_data) from the DualDataset class

    Returns:
        rets: list of tuples (input_ids, labels, attention_mask)
        example output for batch size 2
        
        [(  #forget data for batch of 2
            torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), # input_ids
            torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), # labels
            torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), # attention_mask
            ),
            (  #retain data for batch of 2
            torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), # input_ids
            torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), # labels
            torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), # attention_mask
            ),
        ]

    """

    forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples]
    rets = []
    for data_type in ["forget", "retain"]:
        data = forget_samples if data_type == "forget" else retain_samples
        input_ids = [s[0] for s in data]
        labels = [s[1] for s in data]
        attention_mask = [s[2] for s in data]
        rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))
    return rets


def rmu_collator(samples):
    forget_samples = [s[0] for s in samples]
    retain_samples = [s[1] for s in samples]

    def stack_triplets(data):
        input_ids = torch.stack([x[0] for x in data]).long()
        labels = torch.stack([x[1] for x in data]).long()
        attention_mask = torch.stack([x[2] for x in data]).long()
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    return {
        "forget": stack_triplets(forget_samples),
        "retain": stack_triplets(retain_samples),
    }
