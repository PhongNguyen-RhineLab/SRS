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
