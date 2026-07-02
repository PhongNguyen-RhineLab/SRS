"""
baselines.py
============

Diversity-aware reranking baselines for the MARS experiments, used to answer
the question a reviewer asks first: is the *learned* alpha actually better than
(a) a fixed alpha, and (b) classic diversity rerankers that do not learn
anything? This file adds three baselines on top of the SAME
FAISS-200/h=20 retrieval-constrained pipeline the paper already uses:

  1. Fixed-alpha submodular greedy
        Exactly the MARS selector (diversity_module.greedy_select) but with a
        constant alpha instead of the actor's per-interaction alpha_t. This is
        the single most important baseline: it isolates the contribution of the
        RL policy from the contribution of submodular reranking itself.

  2. MMR (Maximal Marginal Relevance)
        Carbonell & Goldstein (1998). Greedy: at each step pick the item
        maximizing  lambda * rel(x) - (1-lambda) * max_{j in S} sim(x, j).
        The classic relevance-diversity reranker; no training.

  3. DPP (Determinantal Point Process, fast greedy MAP)
        Chen, Zhang & Zhou (NeurIPS 2018), "Fast Greedy MAP Inference for
        Determinantal Point Process to Improve Recommendation Diversity".
        L = diag(q) S diag(q) with quality q_i = exp(theta * rel_norm_i) and
        similarity S = cosine. MAP via the O(N k^2) Cholesky greedy. No training.

Fairness note
-------------
All three rerankers measure item-item similarity in the SAME embedding space
that ILD is scored in (the array returned by
evaluate.get_item_embeddings_for_ild, passed in here as `item_embeddings`).
That keeps the comparison honest: MARS does not get to optimize against a
private notion of similarity that the baselines cannot see. The relevance
signal for every method is the raw FAISS retrieval score r_u(i), identical to
what MARS consumes.

Two layers
----------
* Pure selectors (numpy only, no torch, no repo imports): mmr_select,
  dpp_greedy_map_select, fixed_alpha_submodular_select. Easy to unit-test and
  reused by the self-test at the bottom of this file.
* Pipeline adapters (evaluate_* / sweep_*): thin wrappers that run the real
  retrieval loop and call aggregate_metrics, matching evaluate.evaluate_srs.
  These lazily import torch and metrics so that `python baselines.py`
  (the synthetic self-test) runs with numpy alone.

Run `python baselines.py` to execute the synthetic self-test, which verifies
the selectors satisfy their boundary conditions (e.g. lambda=1 MMR reduces to
relevance top-k) without needing any trained checkpoint.
"""

import numpy as np


# ===========================================================================
# Pure selectors (numpy only)
# ===========================================================================
#
# Each selector takes:
#   cand_ids   : list[int]      candidate item IDs (1-indexed), length N
#   rel_scores : list[float]    FAISS retrieval scores, aligned with cand_ids
#   emb        : np.ndarray      (N, D) L2-normalised embeddings for the SAME
#                                candidates, in the same order as cand_ids.
#                                (Adapters build this by gathering rows of the
#                                 ILD embedding matrix.)
#   k          : int            slate size
# and returns a list of k item IDs in selection order.


def _minmax_norm(x: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1] within the candidate pool.

    MMR's lambda only has a stable meaning if relevance and similarity live on
    comparable scales. Similarity (cosine) is already in [-1, 1]; FAISS scores
    are not bounded the same way, so we rescale relevance per pool.
    Example: rel = [0.2, 0.9, 0.5] -> [0.0, 1.0, 0.4286].
    """
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def mmr_select(cand_ids, rel_scores, emb, lambda_mmr, k):
    """Maximal Marginal Relevance greedy reranking.

    score(x | S) = lambda * rel_norm(x) - (1 - lambda) * max_{j in S} sim(x, j)

    Boundary behaviour:
      lambda = 1.0 -> pure relevance, identical to FAISS top-k.
      lambda = 0.0 -> pure diversity, ignores relevance after the first pick.

    Example:
      cand_ids=[10,11,12], rel=[0.9,0.8,0.1], items 10 and 11 near-duplicates.
      With lambda=0.5 the slate is [10, 12, 11]: 12 jumps ahead of the more
      relevant 11 because 11 is redundant with the already-picked 10.
    """
    N = len(cand_ids)
    k = min(k, N)
    rel = _minmax_norm(np.asarray(rel_scores, dtype=np.float64))
    sim = emb @ emb.T  # (N, N) cosine similarity, since emb rows are unit norm

    selected = []
    selected_mask = np.zeros(N, dtype=bool)
    # max_sim_to_S[i] = max_{j in S} sim(i, j); -inf sentinel for empty S
    max_sim_to_S = np.full(N, -np.inf)

    for _ in range(k):
        if not selected:
            scores = lambda_mmr * rel  # empty S: diversity term is 0
        else:
            scores = lambda_mmr * rel - (1.0 - lambda_mmr) * max_sim_to_S
        scores[selected_mask] = -np.inf
        j = int(np.argmax(scores))
        selected.append(cand_ids[j])
        selected_mask[j] = True
        # incremental update of running max similarity to the selected set
        max_sim_to_S = np.maximum(max_sim_to_S, sim[j])

    return selected


def dpp_greedy_map_select(cand_ids, rel_scores, emb, theta, k, eps=1e-10):
    """DPP MAP inference via the fast Cholesky greedy (Chen et al. 2018).

    Kernel L = diag(q) S diag(q), with
      S      = emb @ emb.T               (cosine similarity, PSD)
      q_i    = exp(theta * rel_norm_i)   (quality from relevance)
    so L_ij = q_i q_j S_ij and L_ii = q_i^2. theta is the relevance-diversity
    knob: theta = 0 gives a pure max-volume (most diverse) set; large theta
    drives selection toward the highest-relevance items.

    Greedy MAP runs in O(N k^2): it maintains incremental Cholesky factors c_i
    and residual gains d_i^2, picking argmax d_i^2 each step. d_i^2 is the
    squared volume gain of adding i, i.e. the marginal log-det increment.

    Example:
      theta large + two near-duplicate high-quality items -> DPP still avoids
      taking both, because once one is selected the other's d^2 (its remaining
      orthogonal volume) collapses toward 0.
    """
    N = len(cand_ids)
    k = min(k, N)
    rel = _minmax_norm(np.asarray(rel_scores, dtype=np.float64))
    q = np.exp(theta * rel)                    # (N,) quality
    S = emb @ emb.T                            # (N, N) similarity
    L = (q[:, None] * S) * q[None, :]          # (N, N) DPP kernel

    cis = np.zeros((k, N))                     # cis[r] = r-th Cholesky row
    d2 = np.copy(np.diag(L)).astype(np.float64)

    selected = []
    j = int(np.argmax(d2))
    selected.append(j)

    for r in range(1, k):
        prev = r - 1                           # number of Cholesky rows filled
        dj = np.sqrt(max(d2[j], eps))
        if prev > 0:
            dot = cis[:prev, j] @ cis[:prev, :]  # (N,)
        else:
            dot = np.zeros(N)
        e = (L[j, :] - dot) / dj               # (N,) new Cholesky coefficients
        cis[prev, :] = e
        d2 = d2 - e ** 2
        for s in selected:
            d2[s] = -np.inf
        j = int(np.argmax(d2))
        if d2[j] <= eps:
            break
        selected.append(j)

    return [cand_ids[i] for i in selected]


def fixed_alpha_submodular_select(cand_ids, rel_scores, emb, alpha, k,
                                  sigma=1.0):
    """Pure-numpy mirror of MARS's submodular greedy at a FIXED alpha.

    Marginal gain Delta(x | S) = alpha * r(x) - (1 - alpha) * sum_{i in S} kappa(i, x)
    with kappa(i, j) = exp(-(1 - cos(e_i, e_j)) / sigma), exactly Eq. (3)-(4)
    of the paper. Provided mainly so the self-test can exercise the same math
    without torch. For real experiments, prefer evaluate_fixed_alpha below,
    which calls diversity_module.greedy_select so the selector is byte-for-byte
    identical to MARS (including the learned sigma).

    Boundary: alpha = 1.0 -> pure relevance top-k (penalty term vanishes).
    """
    N = len(cand_ids)
    k = min(k, N)
    rel = np.asarray(rel_scores, dtype=np.float64)
    cos = emb @ emb.T
    K = np.exp(-(1.0 - cos) / sigma)           # (N, N) RBF kernel on cosine dist

    selected_mask = np.zeros(N, dtype=bool)
    pen = np.zeros(N)                          # pen[x] = sum_{i in S} kappa(i, x)
    slate = []

    for _ in range(k):
        gains = alpha * rel - (1.0 - alpha) * pen
        gains[selected_mask] = -np.inf
        j = int(np.argmax(gains))
        slate.append(cand_ids[j])
        selected_mask[j] = True
        pen = pen + K[j]                       # incremental O(N) update
        if selected_mask.all():
            break

    return slate


# ===========================================================================
# Pipeline adapters
# ===========================================================================
#
# These mirror evaluate.evaluate_srs / evaluate_icsrec_greedy: same retrieval,
# same seen-item exclusion, same aggregate_metrics call. They lazily import
# torch and metrics so the self-test path stays dependency-light.


def _gather_pool_embeddings(cand_ids, item_embeddings):
    """Rows of the ILD embedding matrix for the given candidate IDs.

    item_embeddings is (num_items+1, D), 1-indexed, L2-normalised (exactly the
    array evaluate.get_item_embeddings_for_ild returns). Returns (N, D).
    """
    return np.asarray(item_embeddings)[np.asarray(cand_ids, dtype=np.int64)]


def _iter_candidate_pools(cfg, eval_seqs, retriever, faiss_index):
    """Yield (cand_ids, rel_scores, target) per user, with seen-item exclusion.

    Identical retrieval contract to evaluate.evaluate_srs so every baseline is
    scored on exactly the same pools MARS sees.
    """
    import torch  # local import: keeps the self-test numpy-only
    from evaluate import _get_query_window

    retriever.eval()
    device = cfg.device

    for u, (hist_seq, target) in eval_seqs.items():
        if len(hist_seq) == 0:
            continue
        # no_grad mirrors evaluate.evaluate_srs's @torch.no_grad() decorator.
        # Without it the retriever forward pass tracks gradients and .numpy()
        # raises "Can't call numpy() on Tensor that requires grad".
        with torch.no_grad():
            seq_t = _get_query_window(hist_seq, cfg).unsqueeze(0).to(device)
            query = retriever.get_user_embedding(seq_t).detach().cpu().numpy()[0]
            cand_ids, scores = faiss_index.search(query, cfg.m)

        seen = set(hist_seq)
        f_ids, f_scores = [], []
        for i, s in zip(cand_ids, scores):
            if int(i) not in seen:
                f_ids.append(int(i))
                f_scores.append(float(s))
        yield f_ids, f_scores, target


def _evaluate_with_selector(cfg, eval_seqs, retriever, faiss_index,
                            item_embeddings, select_fn):
    """Shared driver: build slates with select_fn, then aggregate_metrics.

    select_fn(cand_ids, rel_scores, emb) -> slate (list of item IDs).
    Pools shorter than k fall back to the raw relevance order, matching
    evaluate.evaluate_srs.
    """
    from metrics import aggregate_metrics
    from tqdm import tqdm

    slates, targets = [], []
    for cand_ids, rel_scores, target in tqdm(
        _iter_candidate_pools(cfg, eval_seqs, retriever, faiss_index),
        total=len(eval_seqs), desc="Eval baseline",
    ):
        if len(cand_ids) < cfg.k:
            slate = cand_ids
        else:
            emb = _gather_pool_embeddings(cand_ids, item_embeddings)
            slate = select_fn(cand_ids, rel_scores, emb)
        slates.append(slate)
        targets.append(target)

    return aggregate_metrics(slates, targets, item_embeddings, cfg.num_items)


# ----- MMR -----------------------------------------------------------------

def evaluate_mmr(cfg, eval_seqs, retriever, faiss_index, item_embeddings,
                 lambda_mmr):
    """MMR reranking at a single lambda. Returns the standard metrics dict."""
    return _evaluate_with_selector(
        cfg, eval_seqs, retriever, faiss_index, item_embeddings,
        lambda cand_ids, rel, emb: mmr_select(cand_ids, rel, emb, lambda_mmr, cfg.k),
    )


def sweep_mmr(cfg, eval_seqs, retriever, faiss_index, item_embeddings,
              lambdas=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3)):
    """Run MMR across a lambda grid; returns list of {param, **metrics}.

    The point of the sweep is the Pareto frontier: each lambda is one
    (HR@10, ILD) point, and the curve as a whole is what gets compared against
    the MARS point.
    """
    out = []
    for lam in lambdas:
        m = evaluate_mmr(cfg, eval_seqs, retriever, faiss_index, item_embeddings, lam)
        out.append({"method": "MMR", "param": lam, **m})
    return out


# ----- DPP -----------------------------------------------------------------

def evaluate_dpp(cfg, eval_seqs, retriever, faiss_index, item_embeddings, theta):
    """DPP greedy-MAP reranking at a single theta. Returns the metrics dict."""
    return _evaluate_with_selector(
        cfg, eval_seqs, retriever, faiss_index, item_embeddings,
        lambda cand_ids, rel, emb: dpp_greedy_map_select(cand_ids, rel, emb, theta, cfg.k),
    )


def sweep_dpp(cfg, eval_seqs, retriever, faiss_index, item_embeddings,
              thetas=(8.0, 5.0, 3.0, 2.0, 1.0, 0.5, 0.0)):
    """Run DPP across a theta grid; returns list of {param, **metrics}."""
    out = []
    for th in thetas:
        m = evaluate_dpp(cfg, eval_seqs, retriever, faiss_index, item_embeddings, th)
        out.append({"method": "DPP", "param": th, **m})
    return out


# ----- Fixed-alpha submodular ----------------------------------------------

def evaluate_fixed_alpha(cfg, eval_seqs, retriever, faiss_index,
                         diversity_module, item_embeddings, alpha):
    """Fixed-alpha MARS selector: reuses diversity_module.greedy_select.

    This is the strict ablation of the RL policy. Everything except the source
    of alpha is identical to evaluate.evaluate_srs: same trained diversity
    embeddings, same learned sigma, same deterministic greedy. The only change
    is alpha is a constant rather than the actor's alpha_t.

    Suggested grid for the paper includes alpha = alpha_star (the monotone
    threshold) and alpha = 0.617 (the value the policy converges to), so the
    table can show whether the learned policy beats simply pinning alpha at its
    own mean.
    """
    import torch  # noqa: F401  (greedy_select runs under no_grad internally)
    from metrics import aggregate_metrics
    from tqdm import tqdm

    slates, targets = [], []
    for cand_ids, rel_scores, target in tqdm(
        _iter_candidate_pools(cfg, eval_seqs, retriever, faiss_index),
        total=len(eval_seqs), desc=f"Eval fixed-alpha={alpha}",
    ):
        if len(cand_ids) < cfg.k:
            slate = cand_ids
        else:
            slate = diversity_module.greedy_select(
                cand_ids, rel_scores, alpha, eta=0.0, k=cfg.k, training=False,
            )
        slates.append(slate)
        targets.append(target)

    return aggregate_metrics(slates, targets, item_embeddings, cfg.num_items)


def sweep_fixed_alpha(cfg, eval_seqs, retriever, faiss_index, diversity_module,
                      item_embeddings, alphas=(1.0, 0.9, 0.8, 0.7, 0.617, 0.5, 0.3)):
    """Run fixed-alpha across an alpha grid; returns list of {param, **metrics}."""
    out = []
    for a in alphas:
        m = evaluate_fixed_alpha(
            cfg, eval_seqs, retriever, faiss_index, diversity_module, item_embeddings, a)
        out.append({"method": "FixedAlpha", "param": a, **m})
    return out


# ===========================================================================
# Synthetic self-test (no checkpoints, no torch, numpy only)
# ===========================================================================

def _local_ild(slate, emb_lookup):
    """Mean pairwise cosine distance, mirroring metrics.intra_list_diversity."""
    if len(slate) < 2:
        return 0.0
    E = np.stack([emb_lookup[i] for i in slate])
    sims = E @ E.T
    n = len(slate)
    tot, cnt = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            tot += 1.0 - sims[i, j]
            cnt += 1
    return tot / cnt


def _self_test():
    rng = np.random.RandomState(0)
    D = 16
    n_clusters = 12
    per_cluster = 5
    N = n_clusters * per_cluster   # 60 candidates
    k = 10

    # Filter-bubble pool: items live in tight clusters, and relevance is
    # CONCENTRATED in the first two clusters. So relevance top-k piles into a
    # narrow region (low ILD), exactly the failure mode the paper motivates,
    # and a diversity reranker has clear room to spread across clusters.
    centers = rng.randn(n_clusters, D)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8

    emb = np.zeros((N, D))
    cluster_of = np.zeros(N, dtype=int)
    rel = np.zeros(N)
    for idx in range(N):
        c = idx // per_cluster
        cluster_of[idx] = c
        v = centers[c] + 0.03 * rng.randn(D)     # very tight -> near-duplicates
        emb[idx] = v / (np.linalg.norm(v) + 1e-8)
        # high relevance for clusters 0 and 1, low elsewhere (+ small jitter)
        base = 0.9 if c < 2 else 0.3
        rel[idx] = base + 0.05 * rng.rand()

    cand_ids = list(range(1, N + 1))
    emb_lookup = {cand_ids[i]: emb[i] for i in range(N)}
    rel_scores = list(rel)

    rel_topk = [cand_ids[i] for i in np.argsort(-rel)[:k]]

    def n_clusters_covered(slate):
        return len({cluster_of[i - 1] for i in slate})

    print("=" * 64)
    print("Synthetic self-test  (N=%d pool, k=%d, %d clusters, relevance "
          "concentrated in 2)" % (N, k, n_clusters))
    print("=" * 64)

    # --- boundary conditions -------------------------------------------------
    # ties in the concentrated relevance make exact-equality brittle, so check
    # the property that matters: lambda=1 / alpha=1 == argsort of relevance.
    mmr_rel = mmr_select(cand_ids, rel_scores, emb, lambda_mmr=1.0, k=k)
    assert mmr_rel == rel_topk, "MMR lambda=1 must equal relevance top-k"
    fa_rel = fixed_alpha_submodular_select(cand_ids, rel_scores, emb, alpha=1.0, k=k)
    assert fa_rel == rel_topk, "fixed-alpha=1 must equal relevance top-k"

    for slate, name in [
        (mmr_select(cand_ids, rel_scores, emb, 0.5, k), "MMR(0.5)"),
        (dpp_greedy_map_select(cand_ids, rel_scores, emb, 3.0, k), "DPP(3.0)"),
        (fixed_alpha_submodular_select(cand_ids, rel_scores, emb, 0.6, k), "FixedAlpha(0.6)"),
    ]:
        assert len(slate) == k, f"{name}: wrong slate size {len(slate)}"
        assert len(set(slate)) == k, f"{name}: duplicate items"
    print("[ok] boundary conditions and slate validity")

    # relevance top-k should be trapped in <=2 clusters (the filter bubble)
    base_clusters = n_clusters_covered(rel_topk)
    print(f"\nrelevance top-k covers {base_clusters} cluster(s) "
          f"(by construction it should be 2)")
    assert base_clusters <= 2, "relevance top-k should be concentrated"

    # --- trade-off sweeps: report ILD and cluster coverage -------------------
    def sweep_report(name, knobs, select):
        print(f"\n{name} sweep (left = relevance end, right = diversity end):")
        print(f"  {'knob':>7} {'ILD':>8} {'clusters':>9}")
        rows = []
        for kn in knobs:
            s = select(kn)
            ild = _local_ild(s, emb_lookup)
            cov = n_clusters_covered(s)
            rows.append((kn, ild, cov))
            print(f"  {kn:>7.2f} {ild:>8.4f} {cov:>9d}")
        return rows

    mmr_rows = sweep_report("MMR", [1.0, 0.8, 0.6, 0.4, 0.2],
                            lambda lam: mmr_select(cand_ids, rel_scores, emb, lam, k))
    dpp_rows = sweep_report("DPP", [8.0, 4.0, 2.0, 1.0, 0.0],
                            lambda th: dpp_greedy_map_select(cand_ids, rel_scores, emb, th, k))
    fa_rows = sweep_report("FixedAlpha", [1.0, 0.8, 0.6, 0.4, 0.2],
                           lambda a: fixed_alpha_submodular_select(cand_ids, rel_scores, emb, a, k))

    # endpoint invariant: diversity end must out-diversify the relevance end
    for name, rows in [("MMR", mmr_rows), ("DPP", dpp_rows), ("FixedAlpha", fa_rows)]:
        rel_end_ild, div_end_ild = rows[0][1], rows[-1][1]
        rel_end_cov, div_end_cov = rows[0][2], rows[-1][2]
        assert div_end_ild >= rel_end_ild - 1e-9, \
            f"{name}: diversity end should have ILD >= relevance end"
        assert div_end_cov >= rel_end_cov, \
            f"{name}: diversity end should cover >= clusters than relevance end"
    print("\n[ok] every method trades relevance for diversity in the right direction")

    # --- rerankers beat raw top-k on ILD at a middle setting -----------------
    base_ild = _local_ild(rel_topk, emb_lookup)
    mmr_ild = _local_ild(mmr_select(cand_ids, rel_scores, emb, 0.5, k), emb_lookup)
    dpp_ild = _local_ild(dpp_greedy_map_select(cand_ids, rel_scores, emb, 2.0, k), emb_lookup)
    print(f"\nILD: relevance-topk={base_ild:.4f}  MMR(0.5)={mmr_ild:.4f}  DPP(2.0)={dpp_ild:.4f}")
    assert mmr_ild > base_ild, "MMR should diversify beyond relevance top-k"
    assert dpp_ild > base_ild, "DPP should diversify beyond relevance top-k"
    print("[ok] both rerankers improve ILD over relevance top-k")

    print("\nAll self-tests passed.")


if __name__ == "__main__":
    _self_test()