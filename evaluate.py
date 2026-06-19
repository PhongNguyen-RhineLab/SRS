"""
Evaluation routines for Table 2 / Table 3.

Two evaluation modes share the same FAISS-200/h=20 retrieval-constrained
pipeline (Section 5.1):

  1. ICSRec top-10 (greedy)  : take the top-k items by raw FAISS score, no
                               submodular reranking. This is the "relevance
                               only" baseline (Figure 2, red diamond).
  2. SRS (ours)              : actor outputs deterministic alpha_t, greedy
                               selector runs without exploration (eta has no
                               effect since stochastic branch disabled).

Example:
    results = evaluate_srs(cfg, eval_seqs, retriever, faiss_index,
                           diversity_module, state_encoder, actor)
"""

import numpy as np
import torch
from tqdm import tqdm

from metrics import aggregate_metrics


@torch.no_grad()
def _get_history_window(seq: list, cfg) -> torch.Tensor:
    h = cfg.h
    hist = seq[-h:] if len(seq) >= h else seq
    pad = [0] * (h - len(hist))
    return torch.LongTensor(pad + hist)


@torch.no_grad()
def _get_query_window(seq: list, cfg) -> torch.Tensor:
    max_len = cfg.max_seq_len
    hist = seq[-max_len:] if len(seq) >= max_len else seq
    pad = [0] * (max_len - len(hist))
    return torch.LongTensor(pad + hist)


@torch.no_grad()
def evaluate_icsrec_greedy(cfg, eval_seqs: dict, retriever, faiss_index,
                          item_embeddings: np.ndarray) -> dict:
    """
    Baseline: top-k items by raw FAISS inner-product score, no reranking.
    (h=20, seen-item exclusion, same FAISS-200 index as SRS.)
    """
    retriever.eval()
    device = cfg.device

    slates, targets = [], []

    for u, (hist_seq, target) in tqdm(eval_seqs.items(), desc="Eval ICSRec greedy"):
        if len(hist_seq) == 0:
            continue

        seq_t = _get_query_window(hist_seq, cfg).unsqueeze(0).to(device)
        query = retriever.get_user_embedding(seq_t).cpu().numpy()[0]

        cand_ids, scores = faiss_index.search(query, cfg.m)

        # seen-item exclusion
        seen = set(hist_seq)
        filtered = [(i, s) for i, s in zip(cand_ids, scores) if i not in seen]
        filtered.sort(key=lambda x: -x[1])
        slate = [i for i, _ in filtered[:cfg.k]]

        slates.append(slate)
        targets.append(target)

    return aggregate_metrics(slates, targets, item_embeddings, cfg.num_items)


@torch.no_grad()
def evaluate_srs(cfg, eval_seqs: dict, retriever, faiss_index,
                diversity_module, state_encoder, actor,
                item_embeddings: np.ndarray) -> dict:
    """
    Full SRS evaluation: deterministic alpha_t from actor, deterministic
    greedy selection (no exploration), seen-item exclusion.
    """
    retriever.eval()
    state_encoder.eval()
    actor.eval()
    device = cfg.device

    slates, targets = [], []

    for u, (hist_seq, target) in tqdm(eval_seqs.items(), desc="Eval SRS"):
        if len(hist_seq) == 0:
            continue

        state_window = _get_history_window(hist_seq, cfg).unsqueeze(0).to(device)
        state = state_encoder(state_window)

        action, _, _ = actor.sample(state, deterministic=True)
        alpha = float(action[0, 0].item())

        seq_t = _get_query_window(hist_seq, cfg).unsqueeze(0).to(device)
        query = retriever.get_user_embedding(seq_t).cpu().numpy()[0]
        cand_ids, scores = faiss_index.search(query, cfg.m)

        seen = set(hist_seq)
        filtered_ids, filtered_scores = [], []
        for i, s in zip(cand_ids, scores):
            if i not in seen:
                filtered_ids.append(i)
                filtered_scores.append(s)

        if len(filtered_ids) < cfg.k:
            slate = filtered_ids
        else:
            slate = diversity_module.greedy_select(
                filtered_ids, filtered_scores, alpha, eta=0.0, k=cfg.k,
                training=False,
            )

        slates.append(slate)
        targets.append(target)

    return aggregate_metrics(slates, targets, item_embeddings, cfg.num_items)


def get_item_embeddings_for_ild(diversity_module, num_items: int) -> np.ndarray:
    """
    Frozen embeddings used as the common representation space for ILD,
    per Section 5.1: "we compute ILD ... using the frozen ICSRec item
    embeddings, so that all methods are scored by a common, fixed
    representation space."

    Here we use the diversity module's learned embeddings as the fixed
    space (role-equivalent stand-in for the frozen retriever embeddings).
    """
    import torch.nn.functional as F
    with torch.no_grad():
        idx = torch.arange(0, num_items + 1, device=diversity_module.item_emb.weight.device)
        embs = F.normalize(diversity_module.item_emb(idx), dim=-1)
    return embs.cpu().numpy()


@torch.no_grad()
def compute_pool_statistics(cfg, eval_seqs: dict, retriever, faiss_index,
                            diversity_module, max_users: int = None) -> dict:
    """
    Aggregates r_perp (min retrieval score in the top-m pool) and
    kappa_max (max pairwise kernel value over the pool, i != j) across
    test users, as needed for Table IV / Eq. (7).

    Returns: {"r_perp": float, "kappa_max": float, "n_users": int}

    Example:
        stats = compute_pool_statistics(cfg, test_seqs, retriever, faiss_index, div)
        alpha_star = compute_alpha_star(stats["r_perp"], stats["kappa_max"], cfg.k)
    """
    retriever.eval()
    device = cfg.device

    r_perp_vals, kappa_max_vals = [], []
    items = list(eval_seqs.items())
    if max_users is not None:
        items = items[:max_users]

    for u, (hist_seq, target) in tqdm(items, desc="Pool statistics (r_perp, kappa_max)"):
        if len(hist_seq) == 0:
            continue

        seq_t = _get_query_window(hist_seq, cfg).unsqueeze(0).to(device)
        query = retriever.get_user_embedding(seq_t).cpu().numpy()[0]
        cand_ids, scores = faiss_index.search(query, cfg.m)
        if len(scores) == 0:
            continue

        r_perp_vals.append(float(np.min(scores)))

        items_t = torch.LongTensor(cand_ids.tolist()).to(device)
        embs = diversity_module.get_embeddings(items_t)
        K = diversity_module.kernel_matrix(embs)
        N = K.shape[0]
        if N >= 2:
            K_offdiag = K.masked_fill(torch.eye(N, dtype=torch.bool, device=device), float("-inf"))
            kappa_max_vals.append(float(K_offdiag.max().item()))

    return {
        "r_perp": float(np.mean(r_perp_vals)) if r_perp_vals else 0.0,
        "kappa_max": float(np.mean(kappa_max_vals)) if kappa_max_vals else 0.0,
        "n_users": len(r_perp_vals),
    }


@torch.no_grad()
def compute_average_deficit(cfg, eval_seqs: dict, retriever, faiss_index,
                            diversity_module, alpha: float,
                            num_x_samples: int = 5, max_users: int = None) -> float:
    """
    Definition 4: delta_bar(alpha), estimated by greedy rollouts pooled
    across test users.

    Example:
        d_bar = compute_average_deficit(cfg, test_seqs, retriever, faiss_index,
                                       div, alpha=0.617)
    """
    retriever.eval()
    device = cfg.device

    all_samples = []
    items = list(eval_seqs.items())
    if max_users is not None:
        items = items[:max_users]

    for u, (hist_seq, target) in tqdm(items, desc=f"Average deficit rollout (alpha={alpha:.3f})"):
        if len(hist_seq) == 0:
            continue

        seq_t = _get_query_window(hist_seq, cfg).unsqueeze(0).to(device)
        query = retriever.get_user_embedding(seq_t).cpu().numpy()[0]
        cand_ids, scores = faiss_index.search(query, cfg.m)
        if len(cand_ids) < cfg.k:
            continue

        samples = diversity_module.estimate_average_deficit(
            cand_ids.tolist(), scores.tolist(), alpha, cfg.k,
            num_x_samples=num_x_samples,
        )
        all_samples.extend(samples)

    return float(np.mean(all_samples)) if all_samples else 0.0