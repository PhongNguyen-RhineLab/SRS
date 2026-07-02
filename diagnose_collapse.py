"""
Diagnoses why SRS and the ICSRec-greedy baseline produced identical metrics.

Two (non-exclusive) collapse mechanisms can cause this:
  1. alpha -> 1.0   : (1-alpha) in F^sub_theta vanishes, diversity penalty
                      has zero weight regardless of kappa.
  2. sigma -> 0      : kappa(i,j) -> 0 for nearly all i != j, so the
                      penalty term itself vanishes regardless of alpha.

This script loads your actual trained checkpoint and reports:
  - sigma (the kernel bandwidth) and kappa statistics on real candidate pools
  - the actor's alpha distribution (mean/min/max/std) over a sample of test users
  - a side-by-side slate comparison for a few users (SRS vs baseline), so you
    can see directly whether they match item-for-item

Example:
    python diagnose_collapse.py
"""

import os
import numpy as np
import torch

from config import Config

# Set to "movielens_1m" to run against MovieLens-1M instead.
DATASET = "amazon_beauty"
from data import load_and_preprocess, split_data
from retriever import SASRec, FAISSIndex
from diversity import DiversityModule
from rl_policy import StateEncoder, Actor
from evaluate import _get_query_window, _get_history_window


def main(n_sample_users: int = 200):
    cfg = Config(dataset=DATASET)
    data = load_and_preprocess(cfg.data_dir, cfg.dataset)
    cfg.num_items = data["num_items"]
    cfg.num_users = data["num_users"]
    _, _, test_seqs = split_data(data["sequences"])

    device = cfg.device

    retriever = SASRec(cfg.num_items, cfg.ret_emb_dim, cfg.max_seq_len,
                       cfg.ret_num_heads, cfg.ret_num_layers, cfg.ret_dropout).to(device)
    retriever.load_state_dict(torch.load(
        os.path.join(cfg.checkpoint_dir, "sasrec_retriever.pt"), map_location=device))
    retriever.eval()

    faiss_index = FAISSIndex.load(
        os.path.join(cfg.checkpoint_dir, "faiss_index.npy"), cfg.ret_emb_dim)

    diversity_module = DiversityModule(cfg.num_items, cfg.div_emb_dim).to(device)
    diversity_module.load_state_dict(torch.load(
        os.path.join(cfg.checkpoint_dir, "diversity_module.pt"), map_location=device))
    diversity_module.eval()

    state_encoder = StateEncoder(cfg.num_items, cfg.ret_emb_dim, cfg.state_dim, cfg.h).to(device)
    state_encoder.load_state_dict(torch.load(
        os.path.join(cfg.checkpoint_dir, "state_encoder.pt"), map_location=device))
    state_encoder.eval()

    actor = Actor(cfg.state_dim, cfg.hidden_dim, cfg.alpha_init_bias).to(device)
    actor.load_state_dict(torch.load(
        os.path.join(cfg.checkpoint_dir, "actor.pt"), map_location=device))
    actor.eval()

    # ---------------- 1. sigma / kappa stats ----------------
    sigma = diversity_module._sigma().item()
    print(f"[1] Kernel bandwidth sigma = {sigma:.6f}")
    print(f"    (started at 1.0 at init; if this is very small, e.g. <0.05,")
    print(f"     kappa(i,j) underflows toward 0 for almost all pairs)\n")

    # ---------------- 2. alpha distribution over sampled users ----------------
    users = list(test_seqs.items())[:n_sample_users]
    alphas = []

    with torch.no_grad():
        for u, (hist_seq, target) in users:
            if len(hist_seq) == 0:
                continue
            state_window = _get_history_window(hist_seq, cfg).unsqueeze(0).to(device)
            state = state_encoder(state_window)
            action, _, _ = actor.sample(state, deterministic=True)
            alphas.append(float(action[0, 0].item()))

    alphas = np.array(alphas)
    print(f"[2] Actor's alpha over {len(alphas)} sampled test users:")
    print(f"    mean={alphas.mean():.4f}  min={alphas.min():.4f}  "
          f"max={alphas.max():.4f}  std={alphas.std():.4f}")
    pct_above_999 = (alphas > 0.999).mean() * 100
    print(f"    {pct_above_999:.1f}% of users have alpha > 0.999 "
          f"(i.e. diversity penalty weight (1-alpha) < 0.001)\n")

    # ---------------- 3. kappa stats on a real candidate pool ----------------
    u0, (hist0, target0) = users[0]
    with torch.no_grad():
        seq_t = _get_query_window(hist0, cfg).unsqueeze(0).to(device)
        query = retriever.get_user_embedding(seq_t).cpu().numpy()[0]
        cand_ids, scores = faiss_index.search(query, cfg.m)

        items_t = torch.LongTensor(cand_ids.tolist()).to(device)
        embs = diversity_module.get_embeddings(items_t)
        K = diversity_module.kernel_matrix(embs)
        N = K.shape[0]
        off = K.masked_fill(torch.eye(N, dtype=torch.bool, device=device), float("-inf"))
        kappa_max = off.max().item()
        kappa_mean = K[~torch.eye(N, dtype=torch.bool, device=device)].mean().item()

    print(f"[3] Kernel values on user {u0}'s candidate pool (m={cfg.m}):")
    print(f"    kappa_max (most similar pair) = {kappa_max:.6f}")
    print(f"    kappa_mean (off-diagonal avg) = {kappa_mean:.6f}")
    print(f"    (if both are near 0, the penalty term has no discriminative")
    print(f"     power left, regardless of alpha)\n")

    # ---------------- 4. direct slate comparison for a few users ----------------
    print("[4] Direct slate comparison (SRS vs baseline) for first 5 users:")
    n_identical = 0
    n_checked = 0
    with torch.no_grad():
        for u, (hist_seq, target) in users[:5]:
            if len(hist_seq) == 0:
                continue
            seq_t = _get_query_window(hist_seq, cfg).unsqueeze(0).to(device)
            query = retriever.get_user_embedding(seq_t).cpu().numpy()[0]
            cand_ids, scores = faiss_index.search(query, cfg.m)

            seen = set(hist_seq)
            filtered = [(i, s) for i, s in zip(cand_ids, scores) if i not in seen]
            filtered.sort(key=lambda x: -x[1])
            baseline_slate = [i for i, _ in filtered[:cfg.k]]

            state_window = _get_history_window(hist_seq, cfg).unsqueeze(0).to(device)
            state = state_encoder(state_window)
            action, _, _ = actor.sample(state, deterministic=True)
            alpha = float(action[0, 0].item())

            filtered_ids = [i for i, _ in filtered]
            filtered_scores = [s for _, s in filtered]
            srs_slate = diversity_module.greedy_select(
                filtered_ids, filtered_scores, alpha, eta=0.0, k=cfg.k, training=False,
            )

            identical = (srs_slate == baseline_slate)
            n_checked += 1
            n_identical += int(identical)
            print(f"    user {u}: alpha={alpha:.4f}  identical_slate={identical}")
            if not identical:
                print(f"      baseline: {baseline_slate}")
                print(f"      srs:      {srs_slate}")

    print(f"\n  {n_identical}/{n_checked} sampled users had byte-identical slates.")
    print("\nInterpretation: if alpha is consistently > 0.999 in [2] and/or sigma")
    print("is very small in [1] / kappa near 0 in [3], that's your root cause --")
    print("the diversity term is mathematically inert for this checkpoint.")


if __name__ == "__main__":
    main()