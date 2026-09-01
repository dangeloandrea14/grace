import torch
import numpy as np
from transformers import PreTrainedTokenizer
import torch.nn.functional as F
from scipy.stats import hmean
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from tqdm.auto import tqdm
import warnings
from accelerate import Accelerator
from typing import List, Tuple
from collections import Counter
import math
# from syntactic_sim import syntactic_similarity, init_syntax
# import stanza
warnings.filterwarnings("ignore")

accelerator = Accelerator()


def eval_rouge_recall(gen_outputs, ground_truths):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(gen_outputs, ground_truths)

    return rouge_scores['rouge1'].recall, rouge_scores['rougeL'].recall



def eval_cosine_similarity_batched(gen_outputs: List[str], ground_truths: List[str], model: SentenceTransformer, batch_size: int = 32):
    """
    Calculates cosine similarity for all pairs in a batched and efficient manner.
    """
    with torch.no_grad():
        gen_embeddings = model.encode(
            gen_outputs, 
            batch_size=batch_size, 
            show_progress_bar=True, 
            convert_to_tensor=True
        )
        gt_embeddings = model.encode(
            ground_truths, 
            batch_size=batch_size, 
            show_progress_bar=True, 
            convert_to_tensor=True
        )
        pairwise_scores = torch.diag(util.cos_sim(gen_embeddings, gt_embeddings))
        return torch.clamp(pairwise_scores, min=0).tolist()


@torch.no_grad()
def get_probs_ppl(
    question: str,
    answer: str,
    model,
    tokenizer: PreTrainedTokenizer,
    device,
):
    full_text = question + answer + tokenizer.eos_token
    questions_encoded = tokenizer(question, add_special_tokens=False, return_tensors='pt').to(device)
    num_questions = questions_encoded['input_ids'].size(1)
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        return_tensors='pt',
    ).to(device)

    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    labels = input_ids.clone()
    labels[0,:num_questions] = -100
    out = model(input_ids, attention_mask= attention_mask,labels = labels)
    loss = out.loss
    gmp = float(torch.exp(-loss)) #goemetric mean of probs,
    ppl = float(torch.exp(loss)) #perplexity  
    return gmp, ppl   



@torch.no_grad()
def eval_truth_ratio(question : str, true_answer : str, false_answers: List[str],model, 
                     tokenizer, device):
    """
    Calculate truth ratio: mean(P(false_answers))/P(correct_answer)
    """
    true_prob,_ = get_probs_ppl(question, true_answer, model, tokenizer, device)
    false_probs = []
    for f in false_answers:
        probs,_ = get_probs_ppl(question, f, model, tokenizer, device)
        false_probs.append(probs)
    mean_false_prob = np.mean(false_probs) if false_probs else 0.0
    truth_ratio_val = mean_false_prob/ (true_prob + 1e-10)
    tr_score = np.minimum(truth_ratio_val, 1 / truth_ratio_val)
    return tr_score
    

@torch.no_grad()
def generate_outputs(question :str, model, tokenizer, device, max_new_tokens: int = 50):
    inputs = tokenizer(
        question, 
        return_tensors="pt", 
        add_special_tokens=False
    ).to(device)
    out = model.generate(
        **inputs,
        max_new_tokens = max_new_tokens,
        do_sample = False,
        return_dict_in_generate = False
    )
    full_seq = out[0]
    input_ids = inputs['input_ids']
    gen_ids = full_seq[input_ids.size(1):]
    answer = tokenizer.decode(
        gen_ids, 
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return answer


def compute_extraction_strength(model, input_ids, labels, device=None):
    """
    Compute the Extraction Strength (ES) score for a single sample.

    ES quantifies memorization intensity: it finds the minimal prefix length k
    such that greedy decoding from position k reconstructs the full labeled suffix.
    ES = 1 - (k / valid_len), so ES=1 means the model completes from the start,
    ES=0 means it can only complete from the very last token.

    Args:
        model:      A HuggingFace causal LM (in eval mode).
        input_ids:  LongTensor of shape (seq_len,) or (1, seq_len).
        labels:     LongTensor of shape (seq_len,), with IGNORE_INDEX (-100)
                    on non-target positions.
        device:     torch.device to use. Defaults to model.device.

    Returns:
        float: ES score in [0, 1], or None if no valid target tokens exist.
    """
    if device is None:
        device = next(model.parameters()).device

    # Ensure batch dimension
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)   # (1, seq_len)
    if labels.dim() == 1:
        labels = labels.unsqueeze(0)         # (1, seq_len)

    input_ids = input_ids.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids).logits  # (1, seq_len, vocab_size)

    # Shift: position i predicts token i+1
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)[0, :-1, :]  # (seq_len-1, V)
    sample_labels = labels[0]  # (seq_len,)

    # Find positions that have real labels (not IGNORE_INDEX)
    actual_indices = (sample_labels != -100).nonzero(as_tuple=True)[0][:-1]  # drop EOS

    if len(actual_indices) == 0:
        warnings.warn(
            "No valid target tokens found — tokenization mismatch. Returning ES=None.",
            UserWarning,
        )
        return None

    start_idx = actual_indices[0].item()
    end_idx   = actual_indices[-1].item()

    if start_idx == 0:
        warnings.warn(
            "Position 0 has a label — this is unusual and may indicate a data issue.",
            UserWarning,
        )

    # log_probs[start_idx - 1 : end_idx] aligns predictions with target tokens
    target_log_probs = log_probs[start_idx - 1 : end_idx]   # (N, V)
    target_labels    = sample_labels[actual_indices]          # (N,)

    preds     = torch.argmax(target_log_probs, dim=-1)        # (N,)
    valid_len = len(target_labels)

    # Find the minimal k s.t. preds[k:] == labels[k:]
    k = valid_len  # default: no suffix matches
    for k in range(valid_len):
        if torch.equal(preds[k:], target_labels[k:]):
            break

    return 1.0 - (k / valid_len)

def compute_es(df, model, tokenizer, device):
    es_scores = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing forget efficacy", disable=not accelerator.is_main_process):
        q, a, num_tokens = row['question'], row['answer'], row['num_tokens']
        full_text = q + a + tokenizer.eos_token
        q_enc = tokenizer(q, add_special_tokens=False, return_tensors='pt')
        num_q_tokens = q_enc['input_ids'].size(1)
        full_enc = tokenizer(full_text, add_special_tokens=False, return_tensors='pt')
        input_ids = full_enc['input_ids']       # (1, seq_len)
        labels = input_ids.clone()
        labels[0, :num_q_tokens] = -100  # mask question tokens
        es = compute_extraction_strength(model, input_ids, labels, device=device)
        es_scores.append(es if es is not None else 0.0)
    avg_es = np.mean(es_scores)
    print(f"Average ES: {avg_es:.4f}")
    df['es'] = es_scores
    return df, avg_es

def compute_fq_scores(df, model, tokenizer, device): #embedding_model,
    
    gen_answers, probas, rougels, truth, ppls = ([] for _ in range(5))
    ground_truths_for_batch = [] 

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing forget efficacy", disable=not accelerator.is_main_process):
        q, a, num_tokens,  = row['question'], row['answer'],  row['num_tokens']
        

        probs, ppl = get_probs_ppl(q, a, model, tokenizer, device=device)
        gen = generate_outputs(q, model, tokenizer, device=device, max_new_tokens=num_tokens)
        _, rl = eval_rouge_recall(gen, a) 
        #tr = eval_truth_ratio(para_q, para_ans, f_a_list, model, tokenizer, device)
        
    
        gen_answers.append(gen)
        ground_truths_for_batch.append(a) 
        
        probas.append(probs)
        rougels.append(rl)
        ppls.append(ppl)
        #truth.append(tr)
        
    print("Calculating cosine similarity in a batch...")
    #cos_sims = eval_cosine_similarity_batched(gen_answers, ground_truths_for_batch, embedding_model)

    df['gen_answer'] = gen_answers
    df['probs']      = probas
    df['rouge_l']    = rougels
    #df['truth']      = truth
    df['ppl']        = ppls
    #df['cos_sims']   = cos_sims 
    
    # Now calculate the averages
    #avg_cos_sims = np.mean(cos_sims)
    all_scores = np.array([1.0 - np.mean(probas), 1.0 - np.mean(rougels)])#, 1.0 - np.mean(truth)])
    forget_quality = hmean(all_scores)
    #all_scores = np.append(all_scores, avg_cos_sims)
    avg_ppl = np.mean(ppls)
    
    print(f'forget_quality: {forget_quality:.4f}')
    return df, all_scores, forget_quality, avg_ppl


def compute_mu_scores(df, model, tokenizer, embedding_model, device):
    gen_answers, probas, rougels, ppls, js_scores, syntactic = ([] for _ in range(6))
    ground_truths_for_batch = [] 

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing model utility", disable=not accelerator.is_main_process):
        q, a, num_tokens = row['question'], row['answer'], row['num_tokens']
        
        probs, ppl = get_probs_ppl(q, a, model, tokenizer, device=device)
        gen = generate_outputs(q, model, tokenizer, device=device, max_new_tokens=num_tokens)
        _, rl = eval_rouge_recall(gen, a)

        
        gen_answers.append(gen)
        ground_truths_for_batch.append(a) 
        probas.append(probs)
        rougels.append(rl)
        ppls.append(ppl)
        
    print("Calculating cosine similarity in a batch...")
    cos_sims = eval_cosine_similarity_batched(gen_answers, ground_truths_for_batch, embedding_model)

    df['gen_answer'] = gen_answers
    df['probs']      = probas
    df['rouge_l']    = rougels
    df['ppl']        = ppls
    df['cos_sims']   = cos_sims
    all_scores = np.array([np.mean(probas), np.mean(rougels), np.mean(cos_sims)])
    mu = hmean(all_scores)
    avg_ppl = np.mean(ppls)
    
    print(f'mu: {mu:.4f}')
    return df, all_scores, mu, avg_ppl
    


  



