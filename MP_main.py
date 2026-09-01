import os
import json
import argparse
import random
import numpy as np
import torch.multiprocessing as mp
from collections import defaultdict

import RASLIK as raslik

CONFIG_PATH = None


def load_train_jsonl(train_jsonl_path):
    with open(train_jsonl_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_result_data(result_path):
    with open(result_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_latest_json_path(out_dir):
    json_files = [
        os.path.join(out_dir, name)
        for name in os.listdir(out_dir)
        if name.endswith(".json")
    ]
    if not json_files:
        raise FileNotFoundError(f"No json files found in {out_dir}")
    return max(json_files, key=os.path.getmtime)


def export_jsonl_by_ids(train_data, ids, output_path):
    # Build a lookup map from id -> record (do this once)
    id_to_record = {str(item["id"]): item for item in train_data}
    
    with open(output_path, "w", encoding="utf-8") as f:
        for idx in ids:
            idx = str(idx)
            if idx in id_to_record:
                f.write(json.dumps(id_to_record[idx], ensure_ascii=False) + "\n")
            else:
                print(f"Warning: id '{idx}' not found in train data, skipping.")


def aggregate_rankings_from_result_data(result_data, list_key="helpful", infl_key="helpful_infl"):
    scores_per_id = defaultdict(list)
    ranks_per_id = defaultdict(list)

    for sample in result_data.values():
        if not isinstance(sample, dict):
            continue

        ids = sample.get(list_key, [])
        infls = sample.get(infl_key, [])

        for rank, (rid, sc) in enumerate(zip(ids, infls), start=1):
            scores_per_id[rid].append(sc)
            ranks_per_id[rid].append(rank)

    rows = []
    for rid in scores_per_id:
        avg_score = sum(scores_per_id[rid]) / len(scores_per_id[rid])
        avg_rank = sum(ranks_per_id[rid]) / len(ranks_per_id[rid])
        rows.append((rid, avg_rank, avg_score))

    rows.sort(key=lambda x: (x[1], -x[2]))
    return rows


def export_forget_retain_from_results(config, result_data=None):
    train_jsonl_path = config.data.train_data_path
    result_dir = config.influence.outdir
    export_dir = getattr(config.postprocess, "export_dir", "./postprocess_output")

    top_n = int(getattr(config.postprocess, "top_n", config.influence.top_k))
    list_key = getattr(config.postprocess, "list_key", "helpful")
    infl_key = getattr(config.postprocess, "infl_key", f"{list_key}_infl")

    if isinstance(result_data, str):
        result_data = load_result_data(result_data)

    if result_data is None:
        result_json_path = get_latest_json_path(result_dir)
        print(f"Using latest result json: {result_json_path}")
        result_data = load_result_data(result_json_path)

    ranked_rows = aggregate_rankings_from_result_data(
        result_data=result_data,
        list_key=list_key,
        infl_key=infl_key,
    )

    train_data = load_train_jsonl(train_jsonl_path)

    forget_ids = [rid for rid, _, _ in ranked_rows[:top_n]]
    retain_ids = [rid for rid, _, _ in ranked_rows[-top_n:]]

    os.makedirs(export_dir, exist_ok=True)

    forget_path = os.path.join(export_dir, "forget.jsonl")
    retain_path = os.path.join(export_dir, "retain.jsonl")

    export_jsonl_by_ids(train_data, forget_ids, forget_path)
    export_jsonl_by_ids(train_data, retain_ids, retain_path)

    print(f"Saved forget set to: {forget_path}")
    print(f"Saved retain set to: {retain_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default=CONFIG_PATH, type=str)
    args = parser.parse_args()

    config_path = args.config_path

    raslik.init_logging()
    config = raslik.get_config(config_path)
    print(config)

    random.seed(int(config.influence.seed))
    np.random.seed(int(config.influence.seed))

    result_data = raslik.calc_infl_mp(config)
    print("Retrieval finished")

    if getattr(config.postprocess, "enable", True):
        export_forget_retain_from_results(config, result_data=result_data)
        print("Postprocess finished")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()