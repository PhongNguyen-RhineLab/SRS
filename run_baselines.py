"""
run_baselines.py
================

Produces the baseline comparison for the experiments section: MARS against
(a) relevance top-k, (b) fixed-alpha submodular, (c) MMR, and (d) DPP, all on
the same FAISS-200/h=20 pipeline. Writes a combined results JSON and a Pareto
frontier figure (HR@10 vs ILD).

This is the apples-to-apples answer to the obvious reviewer question -- "is the
learned alpha doing anything a fixed alpha or a classic reranker couldn't?" The
sweeps trace each baseline's whole relevance-diversity curve; MARS is a single
point that should sit on or above those curves.

Requires trained checkpoints in cfg.checkpoint_dir (same ones run_test_eval.py
loads). Mirrors run_test_eval.py exactly so it drops into the existing repo.

Example:
    python run_baselines.py
"""

import os
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from data import load_and_preprocess, split_data
from retriever import FAISSIndex
from diversity import DiversityModule
from rl_policy import StateEncoder, Actor
from evaluate import (
    evaluate_srs, evaluate_icsrec_greedy, get_item_embeddings_for_ild,
)
from baselines import sweep_mmr, sweep_dpp, sweep_fixed_alpha

import torch


from cli import build_config

# Sweep grids. Keep alpha_star and 0.617 in the fixed-alpha grid so the table
# can directly contrast "pin alpha at the policy's own mean" vs "let the policy
# adapt alpha per interaction".
MMR_LAMBDAS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3)
DPP_THETAS = (8.0, 5.0, 3.0, 2.0, 1.0, 0.5, 0.0)
FIXED_ALPHAS = (1.0, 0.9, 0.8, 0.7, 0.617, 0.5, 0.3)


def _assert_checkpoint_matches(cfg):
    """Fail early and clearly if checkpoints are missing or belong to a
    different dataset than cfg expects.

    With the ICSRec pipeline (build_index.py), Stage 1 is the released
    frozen ICSRec checkpoint, so the dataset/checkpoint consistency check
    reads the ICSRec item table directly. load_icsrec_retriever preflights
    the same condition, but checking here first gives the diagnosis before
    any model construction. The trained Stage-3 checkpoints
    (diversity_module.pt, state_encoder.pt, actor.pt) must exist in
    cfg.checkpoint_dir; they are produced by pretrain_encoder.py and
    train_rl.py.
    """
    if not os.path.exists(cfg.icsrec_ckpt):
        raise FileNotFoundError(
            f"No ICSRec checkpoint at {cfg.icsrec_ckpt}. Download the "
            f"released ICSRec-SAS checkpoint for {cfg.dataset} and point "
            f"cfg.icsrec_ckpt at it."
        )
    sd = torch.load(cfg.icsrec_ckpt, map_location="cpu")
    ckpt_item_size = sd["item_embeddings.weight"].shape[0]  # num_items + 2
    want_item_size = cfg.num_items + 2                       # 0=pad, ..., mask
    if ckpt_item_size != want_item_size:
        raise RuntimeError(
            "ICSRec checkpoint/dataset mismatch.\n"
            f"  checkpoint at {cfg.icsrec_ckpt}: item_size={ckpt_item_size} "
            f"(num_items={ckpt_item_size - 2})\n"
            f"  config --dataset {cfg.dataset}: num_items={cfg.num_items}\n"
            "Fix: pass --dataset so it matches this checkpoint "
            "(num_items=3416 -> movielens_1m, num_items=12101 -> amazon_beauty), "
            "or point cfg.icsrec_ckpt at the right released checkpoint."
        )
    for fname in ("faiss_index.npy", "diversity_module.pt",
                  "state_encoder.pt", "actor.pt"):
        path = os.path.join(cfg.checkpoint_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No checkpoint at {path}. Run build_index.py, "
                f"pretrain_encoder.py, and train_rl.py first (or set "
                f"CHECKPOINT_DIR to where your {cfg.dataset} checkpoints "
                f"actually live)."
            )


def load_checkpoints(cfg: Config):
    _assert_checkpoint_matches(cfg)
    device = cfg.device
    from icsrec_retriever import load_icsrec_retriever
    retriever = load_icsrec_retriever(
        cfg.icsrec_ckpt, cfg.num_items, hidden_size=cfg.ret_emb_dim, device=device)

    faiss_index = FAISSIndex.load(
        os.path.join(cfg.checkpoint_dir, "faiss_index.npy"), cfg.ret_emb_dim)

    diversity_module = DiversityModule(cfg.num_items, cfg.div_emb_dim).to(device)
    diversity_module.load_state_dict(torch.load(
        os.path.join(cfg.checkpoint_dir, "diversity_module.pt"), map_location=device))
    diversity_module.eval()

    state_encoder = StateEncoder(
        cfg.num_items, cfg.ret_emb_dim, cfg.state_dim, cfg.h,
    ).to(device)
    state_encoder.load_state_dict(torch.load(
        os.path.join(cfg.checkpoint_dir, "state_encoder.pt"), map_location=device))
    state_encoder.eval()

    actor = Actor(cfg.state_dim, cfg.hidden_dim, cfg.alpha_init_bias).to(device)
    actor.load_state_dict(torch.load(
        os.path.join(cfg.checkpoint_dir, "actor.pt"), map_location=device))
    actor.eval()

    return retriever, faiss_index, diversity_module, state_encoder, actor


def _fmt_row(name, m):
    return (f"{name:<22}{m['HR@k']:<9.4f}{m['NDCG@k']:<9.4f}"
            f"{m['MRR@k']:<9.4f}{m['ILD']:<9.4f}{m['Coverage']:<9.4f}")


def make_pareto_figure(mars_pt, relevance_pt, sweeps, save_path, dataset=""):
    """Pareto scatter: HR@10 (x) vs ILD (y). MARS should dominate the curves,
    i.e. sit up-and-to-the-right relative to the baseline frontiers."""
    fig, ax = plt.subplots(figsize=(6.5, 5))

    styles = {
        "MMR": dict(color="tab:orange", marker="o", ls="-"),
        "DPP": dict(color="tab:green", marker="s", ls="-"),
        "FixedAlpha": dict(color="tab:purple", marker="^", ls="-"),
    }
    for method, rows in sweeps.items():
        xs = [r["HR@k"] for r in rows]
        ys = [r["ILD"] for r in rows]
        st = styles.get(method, dict(marker="o", ls="-"))
        ax.plot(xs, ys, label=method, markersize=5, linewidth=1.2, **st)

    ax.scatter([relevance_pt["HR@k"]], [relevance_pt["ILD"]],
               color="red", marker="D", s=90, zorder=5, label="Relevance top-k")
    ax.scatter([mars_pt["HR@k"]], [mars_pt["ILD"]],
               color="blue", marker="*", s=240, zorder=6, label="MARS (ours)")

    ax.set_xlabel("HR@10")
    ax.set_ylabel("ILD (Intra-List Diversity)")
    ax.set_title(f"Relevance-diversity Pareto frontier ({dataset})")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Pareto figure saved to {save_path}")


def main():
    cfg = build_config("Baseline sweep (fixed-alpha / MMR / DPP) vs MARS.")
    if CHECKPOINT_DIR is not None:
        cfg.checkpoint_dir = CHECKPOINT_DIR
    data = load_and_preprocess(cfg.data_dir, cfg.dataset)
    cfg.num_items = data["num_items"]
    cfg.num_users = data["num_users"]
    _, _, test_seqs = split_data(data["sequences"])

    print("Loading trained checkpoints...")
    retriever, faiss_index, diversity_module, state_encoder, actor = load_checkpoints(cfg)
    # Same space as run_test_eval/train_rl: ILD on the frozen retriever
    # embeddings (Section 5.1), NOT the diversity module's.
    item_embs = get_item_embeddings_for_ild(retriever, cfg.num_items)

    print("\n[1/5] Relevance top-k (ICSRec greedy)...")
    relevance_pt = evaluate_icsrec_greedy(cfg, test_seqs, retriever, faiss_index, item_embs)

    print("\n[2/5] MARS (ours)...")
    mars_pt = evaluate_srs(cfg, test_seqs, retriever, faiss_index,
                           diversity_module, state_encoder, actor, item_embs)

    print("\n[3/5] Fixed-alpha submodular sweep...")
    fixed_rows = sweep_fixed_alpha(cfg, test_seqs, retriever, faiss_index,
                                   diversity_module, item_embs, alphas=FIXED_ALPHAS)

    print("\n[4/5] MMR sweep...")
    mmr_rows = sweep_mmr(cfg, test_seqs, retriever, faiss_index, item_embs,
                         lambdas=MMR_LAMBDAS)

    print("\n[5/5] DPP sweep...")
    dpp_rows = sweep_dpp(cfg, test_seqs, retriever, faiss_index, item_embs,
                         thetas=DPP_THETAS)

    sweeps = {"FixedAlpha": fixed_rows, "MMR": mmr_rows, "DPP": dpp_rows}

    # ---- console table ------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"Baseline comparison on {cfg.dataset} (test set)")
    print("=" * 70)
    header = f"{'Method':<22}{'HR@10':<9}{'NDCG@10':<9}{'MRR@10':<9}{'ILD':<9}{'Cov.':<9}"
    print(header)
    print("-" * 70)
    print(_fmt_row("Relevance top-k", relevance_pt))
    for r in fixed_rows:
        print(_fmt_row(f"FixedAlpha a={r['param']}", r))
    for r in mmr_rows:
        print(_fmt_row(f"MMR lam={r['param']}", r))
    for r in dpp_rows:
        print(_fmt_row(f"DPP th={r['param']}", r))
    print("-" * 70)
    print(_fmt_row("MARS (ours)", mars_pt))

    # ---- save ---------------------------------------------------------------
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    out = {
        "dataset": cfg.dataset,
        "relevance_topk": relevance_pt,
        "mars": mars_pt,
        "sweeps": sweeps,
    }
    json_path = os.path.join(cfg.checkpoint_dir, "baseline_results.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {json_path}")

    make_pareto_figure(
        mars_pt, relevance_pt, sweeps,
        os.path.join(cfg.checkpoint_dir, "pareto_frontier.png"),
        dataset=cfg.dataset)


if __name__ == "__main__":
    main()