from torch.utils.data import Dataset
import torch
import pandas as pd
import random

def convert_raw_data_to_model_qa(tokenizer, max_length,  question, answer):
    question = str(question)
    answer = str(answer)
    full_text = question + answer + tokenizer.eos_token
    num_question_tokens = len(tokenizer(question, add_special_tokens=False)['input_ids']) #this is important, we 
    encoded = tokenizer(
        full_text,
        add_special_tokens=False, #this is important, we keep false cause we already added the special tokens from template
        max_length=max_length,
        truncation=True,
    )
    input_ids = encoded['input_ids']
    pad_length = max_length - len(input_ids)
    pad_input_ids = encoded['input_ids']  + [tokenizer.pad_token_id] * pad_length
    pad_attention_mask = [1] * len(input_ids) + [0] * pad_length

    labels = list(input_ids) + [-100] * pad_length

    #change label to -100 for question tokens, including assistant header and end of header.
    for i in range(num_question_tokens): labels[i] = -100
    assert len(pad_input_ids) == max_length
    assert len(labels) == max_length
    assert len(pad_attention_mask) == max_length
    return torch.tensor(pad_input_ids),torch.tensor(labels),torch.tensor(pad_attention_mask)


class SingleDataset(Dataset):
    def __init__(self, forget_data,
                 tokenizer,
                 max_length=512,
                 question_key = 'question',
                 answer_key = 'answer'):
        """
        Initializes the dataset for gradient ascent finetuning

        Args:
            data_path (str): path to the data file. csv file containing columns 'question' and 'answer'
            tokenizer (transformers.PreTrainedTokenizer): tokenizer to process the input
            max_length (int, optional): maximum sequence length for tokenization. Defaults to 512.
            template_format (str, optional): format template for structuring input
        """
        self.data = forget_data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data.iloc[idx][self.qk]
        answer = self.data.iloc[idx][self.ak]
        return convert_raw_data_to_model_qa(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            question=question,
            answer=answer
        )

# TOFU implementation
class DualDatasetRandom(Dataset):
    """
    TOFU way of implementation.

    Args:
        forget_data (pd.DataFrame): DataFrame for forgetting.
        retain_data (pd.DataFrame): DataFrame for retaining.
        tokenizer: tokenizer instance to process text.
        max_length (int): maximum sequence length.
        question_key (str): column name for questions.
        answer_key (str): column name for answers.
    """
    def __init__(self, forget_data, retain_data, tokenizer, max_length,
                 question_key = 'question',
                 answer_key = 'answer'):
        self.forget = forget_data.reset_index(drop=True)
        self.retain = retain_data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.forget)

    def __getitem__(self, idx):
        # The forget sample is chosen sequentially by the DataLoader's index.
        forget_idx = idx
        # A new random sample is chosen every time __getitem__ is called.
        retain_idx = torch.randint(0, len(self.retain), (1,)).item()

        forget_data = convert_raw_data_to_model_qa(
            self.tokenizer, self.max_length,
            self.forget.iloc[forget_idx][self.qk],
            self.forget.iloc[forget_idx][self.ak],
        )

        retain_data = convert_raw_data_to_model_qa(
            self.tokenizer, self.max_length,
            self.retain.iloc[retain_idx][self.qk],
            self.retain.iloc[retain_idx][self.ak],
        )

        return (forget_data, retain_data)
    


class ForgetIdkRetainDatasetRandom(Dataset):
    """
    For each row in forget_data, returns a dictionary containing three items:
    1. The forget question paired with its original answer.
    2. A RANDOMLY selected question-answer pair from the retain_data.

    Output format is a dictionary of tensors:
      {
        'answer_input_ids': ..., 'answer_labels': ..., 'answer_attention_mask': ...,
        'idk_input_ids': ..., 'idk_labels': ..., 'idk_attention_mask': ...,
        'retain_input_ids': ..., 'retain_labels': ..., 'retain_attention_mask': ...,
      }
    """
    def __init__(
        self,
        forget_data: pd.DataFrame,
        retain_data: pd.DataFrame,
        tokenizer,
        max_length: int,
        question_key: str = 'question',
        answer_key: str = 'answer',
    ):
        # validate
        if not all(col in forget_data.columns for col in [question_key, answer_key]):
            raise ValueError(f"forget_data must contain: {question_key}, {answer_key}")
        if not all(col in retain_data.columns for col in [question_key, answer_key]):
            raise ValueError(f"retain_data must contain: {question_key}, {answer_key}")

        self.forget_data = forget_data.reset_index(drop=True)
        self.retain_data = retain_data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.qk, self.ak = question_key, answer_key

    def __len__(self):
        # The length of an epoch is determined by the number of samples to forget.
        return len(self.forget_data)

    def __getitem__(self, idx):
        # The forget sample is chosen sequentially by the DataLoader's index.
        f_row = self.forget_data.iloc[idx]

        # CHANGED: The retain sample is chosen RANDOMLY from the entire retain set.
        random_retain_idx = torch.randint(0, len(self.retain_data), (1,)).item()
        r_row = self.retain_data.iloc[random_retain_idx]

        # --- The rest of the logic remains the same ---

        # Process forget sample with its original answer
        q = f_row[self.qk]
        ans = f_row[self.ak]
        ai, al, am = convert_raw_data_to_model_qa(self.tokenizer, self.max_length, q, ans)


        # Process the RANDOMLY CHOSEN retain sample
        retain_q = r_row[self.qk]
        retain_ans = r_row[self.ak]
        ri, rl, rm = convert_raw_data_to_model_qa(self.tokenizer, self.max_length, retain_q, retain_ans)

        return {
            'answer_input_ids':      ai,
            'answer_labels':         al,
            'answer_attention_mask': am,
            'retain_input_ids':      ri,
            'retain_labels':         rl,
            'retain_attention_mask': rm,
        }
    
    












