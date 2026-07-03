"""
diagnose.py
===========

Two diagnostics that decide how the MARS results should be read, both prompted
by the baseline comparison:

  1. alpha_t / eta_t distribution
        The baseline table showed MARS sitting exactly on the fixed-alpha
        frontier (== fixed alpha 0.5 on Beauty, == relevance corner on ml-1m).
        The open question is whether the policy actually VARIES alpha per user
        or has just learned a constant. This logs the deterministic alpha_t the
        actor emits for every eval user and reports mean, std, range,
        percentiles, and a histogram.
          - std ~ 0        -> policy is effectively a learned fixed alpha; the
                              "per-interaction adaptivity" claim is empty.
          - mean ~ 1.0     -> collapsed to pure relevance (the ml-1m case).
          - substantial std but MARS still only on the frontier -> per-user
                              alpha variation is not buying anything measurable.

  2. item-embedding cosine degeneracy
        Beauty ILD sat near 1.0 with a spread of only ~0.065 across every
        method, versus ~0.51 in the paper. Near-1.0 ILD means the item
        embeddings are near-orthogonal (mean cosine ~ 0), so ILD saturates and
        can barely tell a redundant slate from a diverse one. This measures the
        pairwise-cosine distribution of the embeddings ILD is scored on, both
        over random item pairs and within the retrieved top-k pools that ILD
        actually sees. The within-top-k mean (1 - cos) should reproduce the ILD
        of relevance top-k in the results table -- a cross-check that we are
        looking at the same embedding space.

Usage:
    python diagnose.py --dataset ml-1m
    python diagnose.py --dataset beauty --checkpoint-dir checkpoints

Requires the trained checkpoints (reuses run_baselines.load_checkpoints).
`python diagnose.py --selftest` runs a synthetic check of the numpy pieces with
no checkpoints.
"""

import os
import sys
import numpy as np




# ===========================================================================
# Small reporting helpers (numpy only)
# ===========================================================================

def _report_scalar(name, x):
    x = np.asarray(x, dtype=np.float64)
    qs = np.percentile(x, [0, 5, 25, 50, 75, 95, 100])
    print(f"  {name}")
    print(f"    n={len(x)}  mean={x.mean():.4f}  std={x.std():.4f}")
    print(f"    min={qs[0]:.4f}  p5={qs[1]:.4f}  p25={qs[2]:.4f}  "
          f"median={qs[3]:.4f}  p75={qs[4]:.4f}  p95={qs[5]:.4f}  max={qs[6]:.4f}")


def _ascii_hist(x, lo, hi, bins=20, width=48, label=""):
    x = np.asarray(x, dtype=np.float64)
    counts, edges = np.histogram(x, bins=bins, range=(lo, hi))
    peak = counts.max() if counts.max() > 0 else 1
    print(f"  histogram of {label} (range [{lo:.2f}, {hi:.2f}]):")
    for c, e0, e1 in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(round(width * c / peak))
        print(f"    [{e0:5.2f},{e1:5.2f}) {c:>7d} |{bar}")


def _mean_pairwise_dist(E, ids):
    """Mean pairwise (1 - cos) over the given item ids. This IS the ILD of the
    set, so averaging it over users reproduces the reported ILD metric."""
    if len(ids) < 2:
        return np.nan
    V = E[np.asarray(ids, dtype=np.int64)]
    sims = V @ V.T
    n = len(ids)
    iu = np.triu_indices(n, k=1)
    return float(np.mean(1.0 - sims[iu]))


def _normalize_rows(E):
    E = np.asarray(E, dtype=np.float64)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0            # guard the padding row / zeros
    return E / norms


# ===========================================================================
# Diagnostic 1: alpha_t / eta_t distribution
# ===========================================================================

def diagnose_alpha(cfg, eval_seqs, state_encoder, actor):
    """Log the deterministic (alpha_t, eta_t) the actor emits per eval user."""
    import torch
    from evaluate import _get_history_window

    state_encoder.eval()
    actor.eval()
    device = cfg.device

    alphas, etas = [], []
    with torch.no_grad():
        for u, (hist_seq, target) in eval_seqs.items():
            if len(hist_seq) == 0:
                continue
            hw = _get_history_window(hist_seq, cfg).unsqueeze(0).to(device)
            state = state_encoder(hw)
            action, _, _ = actor.sample(state, deterministic=True)
            alphas.append(float(action[0, 0].item()))
            etas.append(float(action[0, 1].item()))

    alphas = np.asarray(alphas)
    etas = np.asarray(etas)

    # Persist for plot_paper_figures.py (fig_policy): the histogram in the
    # paper is regenerated from this file rather than re-running the model.
    np.save(os.path.join(cfg.checkpoint_dir, "alpha_values.npy"), alphas)
    np.save(os.path.join(cfg.checkpoint_dir, "eta_values.npy"), etas)

    print("=" * 66)
    print(f"Diagnostic 1: policy output distribution ({cfg.dataset}, "
          f"{len(alphas)} users)")
    print("=" * 66)
    _report_scalar("alpha_t (trade-off)", alphas)
    _ascii_hist(alphas, 0.0, 1.0, bins=20, label="alpha_t")
    print()
    _report_scalar("eta_t (exploration knob, unused at eval)", etas)

    # interpretation
    print("\n  reading:")
    if alphas.mean() > 0.95:
        print("    - mean alpha ~ 1.0: policy collapsed to PURE RELEVANCE. MARS "
              "reduces to relevance top-k (the ml-1m case).")
    if alphas.std() < 0.02:
        print("    - std < 0.02: alpha is effectively CONSTANT across users. "
              "MARS is a learned fixed alpha; per-interaction adaptivity is "
              "not happening in practice.")
    elif alphas.std() < 0.05:
        print("    - std small (<0.05): alpha barely varies across users.")
    else:
        print(f"    - alpha varies across users (std={alphas.std():.3f}). If "
              "MARS still only matches the fixed-alpha frontier, per-user "
              "variation is not improving the relevance-diversity trade-off.")

    return {"alpha_mean": float(alphas.mean()), "alpha_std": float(alphas.std()),
            "eta_mean": float(etas.mean()), "eta_std": float(etas.std()),
            "alphas": alphas, "etas": etas}


# ===========================================================================
# Diagnostic 2: embedding cosine degeneracy
# ===========================================================================

def _degeneracy_verdict(topk_ild, global_dist):
    """The decisive test. Healthy embeddings make retrieved-similar items much
    closer than random pairs, so within-top-k ILD should sit well below the
    global mean (1 - cos). If the ratio is near 1, retrieved items are no more
    similar than random pairs -- the diversity space ignores the structure the
    retriever uses, and ILD is degenerate."""
    ratio = topk_ild / global_dist if global_dist > 1e-9 else float("nan")
    if ratio > 0.9:
        msg = (f"ratio topk/global = {ratio:.2f} ~ 1: DEGENERATE. Retrieved "
               "similar items are no closer than random pairs in this space, "
               "so ILD near 1.0 is an artifact, not real diversity. Likely the "
               "diversity module is under-trained or ILD uses the wrong "
               "embeddings. Contrast the paper's ILD ~ 0.51.")
    elif ratio > 0.75:
        msg = (f"ratio topk/global = {ratio:.2f}: weak. Some structure, but "
               "retrieved items are only slightly more similar than random.")
    else:
        msg = (f"ratio topk/global = {ratio:.2f}: healthy. Retrieved items are "
               "clearly more similar than random pairs, so ILD is meaningful.")
    return ratio, msg


def diagnose_embeddings_global(item_embeddings, n_pairs=200000, seed=0):
    """Pairwise cosine over random item pairs from the ILD embedding matrix.

    Note: for high-dim embeddings spanning many item clusters, random-pair
    cosine near 0 (mean (1-cos) near 1) is NORMAL and not by itself a problem --
    most random pairs are unrelated. The degeneracy verdict is made in 2b by
    comparing this global value against the within-pool value.
    """
    E = _normalize_rows(item_embeddings)
    N = E.shape[0]                          # rows 1..N-1 are real items (0 = pad)
    rng = np.random.RandomState(seed)
    i = rng.randint(1, N, size=n_pairs)
    j = rng.randint(1, N, size=n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    cos = np.sum(E[i] * E[j], axis=1)

    print("=" * 66)
    print(f"Diagnostic 2a: global embedding geometry ({N - 1} items)")
    print("=" * 66)
    _report_scalar("random-pair cosine", cos)
    _report_scalar("random-pair (1 - cos) = ILD contribution", 1.0 - cos)
    _ascii_hist(cos, -0.5, 1.0, bins=20, label="cosine")
    mean_dist = float(np.mean(1.0 - cos))
    print(f"\n  global mean (1-cos) = {mean_dist:.3f}. On its own this says "
          "little; the verdict comes from comparing it to within-pool ILD (2b).")
    return {"global_mean_cos": float(cos.mean()), "global_mean_dist": mean_dist}


def diagnose_embeddings_pools(cfg, eval_seqs, retriever, faiss_index,
                              item_embeddings, global_mean_dist=None,
                              n_users=500, pool_cap=50, seed=0):
    """Within-pool cosine: what ILD actually sees.

    Retrieves each sampled user's top-m pool and measures mean pairwise
    (1 - cos) both within the relevance top-k slate and within the (capped)
    full pool. The top-k number should reproduce the ILD of relevance top-k in
    the results table -- if it does, we are certainly looking at the same
    embeddings. The verdict compares it to the global value from 2a.
    """
    import torch
    from evaluate import _get_query_window

    E = _normalize_rows(item_embeddings)
    device = cfg.device
    retriever.eval()

    users = list(eval_seqs.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(users)

    topk_d, pool_d = [], []
    with torch.no_grad():
        for u in users[:n_users]:
            hist_seq, target = eval_seqs[u]
            if len(hist_seq) == 0:
                continue
            seq_t = _get_query_window(hist_seq, cfg).unsqueeze(0).to(device)
            query = retriever.get_user_embedding(seq_t).detach().cpu().numpy()[0]
            cand_ids, scores = faiss_index.search(query, cfg.m)
            seen = set(hist_seq)
            f = [(int(i), float(s)) for i, s in zip(cand_ids, scores)
                 if int(i) not in seen]
            f.sort(key=lambda x: -x[1])
            ids = [i for i, _ in f]
            if len(ids) >= 2:
                topk_d.append(_mean_pairwise_dist(E, ids[:cfg.k]))
                pool_d.append(_mean_pairwise_dist(E, ids[:pool_cap]))

    topk_d = np.asarray([d for d in topk_d if not np.isnan(d)])
    pool_d = np.asarray([d for d in pool_d if not np.isnan(d)])

    print("=" * 66)
    print(f"Diagnostic 2b: within-pool geometry ({len(topk_d)} users sampled)")
    print("=" * 66)
    _report_scalar("mean ILD of relevance top-k slate (cross-check vs table)", topk_d)
    _report_scalar(f"mean (1-cos) within top-{pool_cap} retrieved pool", pool_d)
    print("\n  reading:")
    print("    - the top-k mean above should match relevance-top-k ILD in your "
          "results table (confirms same embedding space).")
    if global_mean_dist is not None:
        ratio, msg = _degeneracy_verdict(float(topk_d.mean()), global_mean_dist)
        print(f"    - {msg}")
    return {"topk_ild_mean": float(topk_d.mean()),
            "pool_dist_mean": float(pool_d.mean())}


# ===========================================================================
# Driver
# ===========================================================================

def main(cfg):
    from data import load_and_preprocess, split_data
    from evaluate import get_item_embeddings_for_ild
    from run_baselines import load_checkpoints

    data = load_and_preprocess(cfg.data_dir, cfg.dataset)
    cfg.num_items = data["num_items"]
    cfg.num_users = data["num_users"]
    _, _, test_seqs = split_data(data["sequences"])

    print(f"Loading checkpoints for {cfg.dataset}...\n")
    retriever, faiss_index, diversity_module, state_encoder, actor = load_checkpoints(cfg)
    # Audit the SAME space the reported ILD metric uses: the frozen retriever
    # embeddings (evaluate.get_item_embeddings_for_ild is called with the
    # retriever everywhere in train/eval). Passing diversity_module here was a
    # leftover from before the ILD-space fix and made the audit inspect a
    # different space than the one the tables report.
    item_embs = get_item_embeddings_for_ild(retriever, cfg.num_items)

    diagnose_alpha(cfg, test_seqs, state_encoder, actor)
    print()
    resG = diagnose_embeddings_global(item_embs)
    print()
    diagnose_embeddings_pools(cfg, test_seqs, retriever, faiss_index, item_embs,
                              global_mean_dist=resG["global_mean_dist"])


# ===========================================================================
# Synthetic self-test (numpy only, no checkpoints)
# ===========================================================================

def _selftest():
    print("Running synthetic self-test (no checkpoints)\n")

    # --- embedding diagnostics: the decisive signal is the GAP between global
    #     random-pair distance and within-pool distance ------------------------
    rng = np.random.RandomState(0)
    D = 64
    N = 2000
    n_clusters = 40

    # HEALTHY regime: items cluster, and a "pool" is drawn from ONE cluster
    # (mimicking retrieval of similar items). Within-pool distance should be
    # well below the global random-pair distance.
    centers = rng.randn(n_clusters, D)
    Estruct = np.zeros((N + 1, D))
    cluster_of = np.zeros(N + 1, dtype=int)
    for idx in range(1, N + 1):
        c = idx % n_clusters
        cluster_of[idx] = c
        Estruct[idx] = centers[c] + 0.25 * rng.randn(D)
    Estruct = _normalize_rows(Estruct)
    resH = diagnose_embeddings_global(Estruct, n_pairs=50000)
    # simulate same-cluster pools
    healthy_topk = []
    for c in range(n_clusters):
        ids = [i for i in range(1, N + 1) if cluster_of[i] == c][:10]
        healthy_topk.append(_mean_pairwise_dist(Estruct, ids))
    healthy_topk = float(np.nanmean(healthy_topk))
    ratioH, msgH = _degeneracy_verdict(healthy_topk, resH["global_mean_dist"])
    print(f"\n  [healthy sim] within-cluster topk ILD={healthy_topk:.3f}  {msgH}")
    assert ratioH < 0.9, "healthy regime should have within-pool << global"
    print()

    # DEGENERATE regime: near-orthogonal embeddings. Even same-index "pools"
    # are no closer than random pairs -> ratio ~ 1.
    Eortho = _normalize_rows(rng.randn(N + 1, D))
    Eortho[0] = 0.0
    resD = diagnose_embeddings_global(Eortho, n_pairs=50000)
    degen_topk = np.nanmean([
        _mean_pairwise_dist(Eortho, list(rng.randint(1, N + 1, size=10)))
        for _ in range(40)])
    ratioD, msgD = _degeneracy_verdict(float(degen_topk), resD["global_mean_dist"])
    print(f"\n  [degenerate sim] random-pool topk ILD={degen_topk:.3f}  {msgD}")
    assert ratioD > 0.9, "degenerate regime should have within-pool ~ global"
    print("\n[ok] gap-based degeneracy verdict separates healthy vs degenerate\n")

    # --- alpha diagnostic with mock encoder/actor (needs torch) -------------
    try:
        import torch
    except ImportError:
        print("[skip] alpha mock: torch not installed in this environment")
        print("\nAll numpy self-tests passed.")
        return
    class MockEncoder:
        def eval(self): pass
        def __call__(self, hw):
            return torch.zeros(1, 8)

    class MockActor:
        """Emits alpha ~ 0.617 with small jitter, eta ~ 0.5."""
        def __init__(self, seed=0):
            self.g = torch.Generator().manual_seed(seed)
        def eval(self): pass
        def sample(self, state, deterministic=True):
            a = 0.617 + 0.03 * torch.randn(1, generator=self.g).item()
            action = torch.tensor([[a, 0.5]])
            return action, None, None

    class _Cfg:
        device = "cpu"
        h = 20
    eval_seqs = {u: ([1, 2, 3, 4, 5], 6) for u in range(300)}
    res = diagnose_alpha(_Cfg(), eval_seqs, MockEncoder(), MockActor())
    assert 0.55 < res["alpha_mean"] < 0.68, "mock alpha mean off"
    assert res["alpha_std"] > 0.0
    print("\n[ok] alpha diagnostic runs and summarizes a mock policy")

    print("\nAll self-tests passed.")


if __name__ == "__main__":
    from cli import make_parser, config_from_args
    parser = make_parser("Diagnostics: policy alpha distribution + ILD geometry audit.")
    parser.add_argument("--selftest", action="store_true",
                        help="Run synthetic numpy checks (no checkpoints needed).")
    args = parser.parse_args()
    if args.selftest:
        _selftest()
    else:
        main(config_from_args(args))