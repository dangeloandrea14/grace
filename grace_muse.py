import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True, help="Model name, e.g. 'llama' or 'qwen'")
args = parser.parse_args()
model_name = args.model_name

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import json
import torch
import numpy as np
from pathlib import Path
from scipy.optimize import nnls
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.linear_model import OrthogonalMatchingPursuit
from tqdm import tqdm
import time
import pandas as pd

# ── Paths & Hyperparameters ───────────────────────────────────────────────────
STORED_GRADS_DIR  = Path(f"./gradients/muse_{model_name}/")
SMALL_GRADS_DIR   = Path(f"./gradients/poison/muse_{model_name}/")
FORGET_OUTPUT     = Path(f"./selected_data/muse_{model_name}_ids_avg.jsonl")
RETAIN_OUTPUT     = Path(f"./selected_data/muse_{model_name}_retain_avg.jsonl")
AVG_GRAD_PATH     = Path(f"./avg_gradients/muse_{model_name}_avg_grad.pt")

TOP_K                = 400
TOP_M                = 90
N_RETAIN             = 100
N_CLUSTERS           = 10
N_RETAIN_PER_CLUSTER = N_RETAIN // N_CLUSTERS

for d in ["./selected_data", "./avg_gradients", "./data/muse"]:
    os.makedirs(d, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def nnomp_sklearn(target, dictionary, m, tol=1e-4):
    """Greedy non-negative OMP atom selection via NNLS."""
    b = target.detach().cpu().float().numpy() if isinstance(target, torch.Tensor) else target
    D = dictionary.detach().cpu().float().numpy() if isinstance(dictionary, torch.Tensor) else dictionary

    normb = np.linalg.norm(b)
    if normb < 1e-12:
        return []

    residual = b.copy()
    active_set = []

    for _ in range(m):
        if np.linalg.norm(residual) / (normb + 1e-12) < tol:
            break
        correlations = D @ residual
        correlations = correlations.copy()
        correlations[correlations < 0] = 0.0
        if active_set:
            correlations[active_set] = -np.inf
        best = int(np.argmax(correlations))
        if not np.isfinite(correlations[best]) or correlations[best] < 1e-10:
            break
        active_set.append(best)
        coeffs, _ = nnls(D[active_set].T, b)
        residual = b - D[active_set].T @ coeffs

    return active_set


def omp_sklearn(target, dictionary, m, tol=1e-4):
    """Greedy OMP atom selection via sklearn (unrestricted coefficients)."""
    b = target.detach().cpu().float().numpy() if isinstance(target, torch.Tensor) else target.copy()
    D = dictionary.detach().cpu().float().numpy() if isinstance(dictionary, torch.Tensor) else dictionary

    if np.linalg.norm(b) < 1e-12:
        return []

    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=m, tol=None, fit_intercept=False, precompute="auto")
    omp.fit(D.T, b)
    return list(np.where(omp.coef_ != 0)[0])


# ═════════════════════════════════════════════════════════════════════════════
# PART 1 — FORGET SET SELECTION
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PART 1: FORGET SET SELECTION")
print("="*60)

# Step 1 — Load poison (small) gradients
small_pt_files = sorted(SMALL_GRADS_DIR.glob("*.pt"))
print(f"Loading {len(small_pt_files)} poison gradients from {SMALL_GRADS_DIR} ...")

small_ids, small_vecs = [], []
for pt_file in tqdm(small_pt_files, desc="Loading poison gradients"):
    small_ids.append(pt_file.stem)
    g = torch.load(pt_file, map_location="cuda").float()
    if g.ndim == 1:
        g = g.unsqueeze(0)
    small_vecs.append(g)

small_matrix = torch.cat(small_vecs, dim=0)
print(f"Poison gradient matrix: {small_matrix.shape}")

# Step 2 — Load large pool gradients
stored_ids, stored_vecs = [], []
pt_files = sorted(STORED_GRADS_DIR.glob("*.pt"))
print(f"Loading {len(pt_files)} stored gradients from {STORED_GRADS_DIR} ...")

for pt_file in tqdm(pt_files, desc="Loading pool gradients"):
    stored_ids.append(pt_file.stem)
    g = torch.load(pt_file, map_location="cuda").float()
    if g.ndim == 1:
        g = g.unsqueeze(0)
    stored_vecs.append(g)

stored_matrix = torch.cat(stored_vecs, dim=0).contiguous()

assert small_matrix.shape[1] == stored_matrix.shape[1], (small_matrix.shape, stored_matrix.shape)

# Step 3 — Average gradient → top-K candidates
print("\nStep 3: Ranking pool by inner product with average poison gradient ...")
avg_grad = small_matrix.mean(dim=0)

torch.cuda.synchronize()
t0 = time.time()
avg_scores = stored_matrix @ avg_grad
torch.cuda.synchronize()
top_k_values, top_k_indices = torch.topk(avg_scores, k=TOP_K)
top_k_ids    = [stored_ids[i] for i in top_k_indices.tolist()]
top_k_matrix = stored_matrix[top_k_indices]
print(f"Top-{TOP_K} selection took {time.time() - t0:.2f} seconds")

# Step 4 — NNOMP to refine to TOP_M
print(f"\nStep 4: NNOMP selection (target: avg poison grad → {TOP_M} samples) ...")
avg_grad = small_matrix.mean(dim=0)
torch.save(avg_grad.cpu().float(), AVG_GRAD_PATH)
print(f"Saved averaged gradient to {AVG_GRAD_PATH}")

t0 = time.time()
selected_local_indices = nnomp_sklearn(target=avg_grad, dictionary=top_k_matrix, m=TOP_M, tol=1e-4)
print(f"NNOMP took {time.time() - t0:.2f} seconds")

selected_set = set(selected_local_indices)
if len(selected_local_indices) < TOP_M:
    fillers = [i for i in range(len(top_k_ids)) if i not in selected_set]
    selected_local_indices.extend(fillers[:TOP_M - len(selected_local_indices)])

forget_ids = [top_k_ids[i] for i in selected_local_indices]
print(f"\nSelected {len(forget_ids)} forget samples.")

# Step 5 — Save forget set
output = {"top_k": top_k_ids, "top_m": forget_ids, "poison_ids": small_ids}
with open(FORGET_OUTPUT, "w") as f:
    f.write(json.dumps(output) + "\n")
print(f"Saved forget selection to {FORGET_OUTPUT}")

all_forget_ids = forget_ids + small_ids
muse = pd.read_parquet("./data/muse_data.parquet")
muse[muse["id"].isin(all_forget_ids)].to_parquet(
    f"./data/muse/{model_name}_grace_forget.parquet", index=False
)


# ═════════════════════════════════════════════════════════════════════════════
# PART 2 — RETAIN SET SELECTION
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PART 2: RETAIN SET SELECTION")
print("="*60)

# Exclude forget-set candidates from the retain pool
excluded_ids = set(top_k_ids)
print(f"Excluding {len(excluded_ids)} forget-candidate IDs from retain pool.")

remaining_ids, remaining_vecs = [], []
for pt_file in tqdm(sorted(STORED_GRADS_DIR.glob("*.pt")), desc="Loading retain pool gradients"):
    if pt_file.stem in excluded_ids:
        continue
    remaining_ids.append(pt_file.stem)
    remaining_vecs.append(torch.load(pt_file, map_location="cuda").float())

remaining_matrix = torch.cat([v.reshape(1, -1) for v in remaining_vecs], dim=0).float()
print(f"Retain pool size: {remaining_matrix.shape[0]}")

# Project gradients orthogonal to the average gradient
print("\nProjecting retain pool gradients orthogonal to avg gradient ...")
t0 = time.time()
gf    = avg_grad.float()
dots  = remaining_matrix @ gf
coeff = (dots / (gf * gf).sum()).unsqueeze(1)
projected_matrix = remaining_matrix - coeff * gf.unsqueeze(0)
projected_np = projected_matrix.cpu().numpy().astype(np.float32)
print(f"Projection took {time.time() - t0:.2f} seconds")

total_var_before = remaining_matrix.cpu().numpy().var(axis=0).sum()
total_var_after  = projected_np.var(axis=0).sum()
print(f"Variance retained after projection: {total_var_after / total_var_before:.3%}")

# KMeans clustering
print(f"\nRunning KMeans with {N_CLUSTERS} clusters ...")
t0 = time.time()
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init="auto")
cluster_labels = kmeans.fit_predict(projected_np)
centroids = kmeans.cluster_centers_
print(f"KMeans took {time.time() - t0:.2f} seconds")

# OMP per cluster
print("\nRunning OMP selection per cluster ...")
projected_normed = normalize(projected_np, norm="l2")
all_retain_ids = []
t0 = time.time()

for cluster_id in tqdm(range(N_CLUSTERS), desc="Clusters"):
    cluster_mask = np.where(cluster_labels == cluster_id)[0]
    if len(cluster_mask) == 0:
        print(f"  Cluster {cluster_id}: empty, skipping.")
        continue

    cluster_vecs     = projected_normed[cluster_mask]
    cluster_centroid = centroids[cluster_id]
    n_select         = min(N_RETAIN_PER_CLUSTER, len(cluster_mask))

    local_indices = omp_sklearn(target=cluster_centroid, dictionary=cluster_vecs, m=n_select, tol=1e-4)
    selected      = [remaining_ids[cluster_mask[i]] for i in local_indices]
    all_retain_ids.extend(selected)
    print(f"  Cluster {cluster_id}: {len(cluster_mask)} samples → selected {len(selected)}")

print(f"\nOMP across all clusters took {time.time() - t0:.2f} seconds")
print(f"Total retain samples selected: {len(all_retain_ids)}")

# Save retain set
retain_output = {"diversity_selected": all_retain_ids}
with open(RETAIN_OUTPUT, "w") as f:
    f.write(json.dumps(retain_output) + "\n")
print(f"Saved retain selection to {RETAIN_OUTPUT}")

muse[muse["id"].isin(all_retain_ids)].to_parquet(
    f"./data/muse/{model_name}_grace_retain.parquet", index=False
)

print("\nDone. Forget and retain sets saved.")