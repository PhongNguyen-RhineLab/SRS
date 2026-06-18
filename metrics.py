"""
Metrics used in Table 2 / Table 3:
  - HR@k    : Hit Rate
  - NDCG@k  : Normalized Discounted Cumulative Gain
  - MRR@k   : Mean Reciprocal Rank
  - ILD     : Intra-List Diversity (mean pairwise cosine distance within slate)
  - Coverage: fraction of catalog items appearing in >=1 returned slate

Example:
    hr = hit_rate(slate=[5,9,2,7], target=9)              # -> 1.0
    ndcg = ndcg_at_k(slate=[5,9,2,7], target=9)            # rank=1 -> 1/log2(3)
    ild = intra_list_diversity([5,9,2,7], item_embeddings)
"""

import math
import numpy as np


def hit_rate(slate: list, target: int) -> float:
    return 1.0 if target in slate else 0.0


def ndcg_at_k(slate: list, target: int) -> float:
    if target not in slate:
        return 0.0
    rank = slate.index(target)  # 0-indexed
    return 1.0 / math.log2(rank + 2)  # +2 because rank is 0-indexed


def mrr_at_k(slate: list, target: int) -> float:
    if target not in slate:
        return 0.0
    rank = slate.index(target)
    return 1.0 / (rank + 1)


def intra_list_diversity(slate: list, item_embeddings: np.ndarray) -> float:
    """
    Mean pairwise cosine distance within the slate.
    item_embeddings : (num_items+1, D) array, 1-indexed by item ID, L2-normalised rows.
    """
    if len(slate) < 2:
        return 0.0

    embs = item_embeddings[slate]  # (|S|, D)
    sims = embs @ embs.T
    n = len(slate)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += (1.0 - sims[i, j])
            count += 1
    return total / count if count > 0 else 0.0


def catalog_coverage(all_slates: list, num_items: int) -> float:
    """
    Fraction of the catalog appearing in at least one returned slate.
    all_slates: list of slates (each a list of item IDs)
    """
    seen = set()
    for slate in all_slates:
        seen.update(slate)
    return len(seen) / num_items


def aggregate_metrics(slates: list, targets: list,
                      item_embeddings: np.ndarray, num_items: int) -> dict:
    """
    Computes mean HR/NDCG/MRR over all (slate, target) pairs, plus ILD and coverage.
    """
    hrs, ndcgs, mrrs, ilds = [], [], [], []

    for slate, target in zip(slates, targets):
        hrs.append(hit_rate(slate, target))
        ndcgs.append(ndcg_at_k(slate, target))
        mrrs.append(mrr_at_k(slate, target))
        ilds.append(intra_list_diversity(slate, item_embeddings))

    return {
        "HR@k": float(np.mean(hrs)),
        "NDCG@k": float(np.mean(ndcgs)),
        "MRR@k": float(np.mean(mrrs)),
        "ILD": float(np.mean(ilds)),
        "Coverage": catalog_coverage(slates, num_items),
    }
