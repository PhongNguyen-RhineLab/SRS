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
from retriever import SASRec, FAISSIndex
from diversity import DiversityModule
from rl_policy import StateEncoder, Actor
from evaluate import (
    evaluate_srs, evaluate_icsrec_greedy, get_item_embeddings_for_ild,
)
from baselines import sweep_mmr, sweep_dpp, sweep_fixed_alpha

import torch


# Which dataset's checkpoints to evaluate. This MUST match the dataset the
# checkpoints in CHECKPOINT_DIR were trained on -- the model dimensions
# (num_items, max_seq_len) come from the dataset and a mismatch makes
# torch.load_state_dict fail with a size-mismatch error.
# DATASET = "movielens_1m"
DATASET = "amazon_beauty"
# Where the checkpoints actually live. Leave as None to use config.py's default
# (which for movielens_1m auto-redirects "checkpoints" -> "checkpoints_movielens_1m").
# Set this to an explicit path if your checkpoints are somewhere else -- e.g. if
# you trained ml-1m but saved into the plain "checkpoints" folder, set
# CHECKPOINT_DIR = "checkpoints" to stop the auto-redirect from looking in the
# wrong place.
CHECKPOINT_DIR = None

# Sweep grids. Keep alpha_star and 0.617 in the fixed-alpha grid so the table
# can directly contrast "pin alpha at the policy's own mean" vs "let the policy
# adapt alpha per interaction".
MMR_LAMBDAS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3)
DPP_THETAS = (8.0, 5.0, 3.0, 2.0, 1.0, 0.5, 0.0)
FIXED_ALPHAS = (1.0, 0.9, 0.8, 0.7, 0.617, 0.5, 0.3)


def _assert_checkpoint_matches(cfg):
    """Fail early and clearly if the checkpoint was trained on a different
    dataset than cfg expects.

    torch.load_state_dict's native error ("size mismatch for item_emb.weight:
    copying a param with shape [3707, 64] ... current model is [12102, 64]")
    is cryptic. This turns it into a one-line diagnosis naming the likely cause.
    Example: a [3707, 64] item_emb means num_items=3706 (MovieLens-1M), so if
    cfg is amazon_beauty (num_items=12101) you are pointing at the wrong
    DATASET or the wrong CHECKPOINT_DIR.
    """
    path = os.path.join(cfg.checkpoint_dir, "sasrec_retriever.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No checkpoint at {path}. Set CHECKPOINT_DIR to where your "
            f"{cfg.dataset} checkpoints actually live."
        )
    sd = torch.load(path, map_location="cpu")
    ckpt_items = sd["item_emb.weight"].shape[0]      # num_items + 1
    ckpt_maxlen = sd["pos_emb.weight"].shape[0]      # max_seq_len + 1
    want_items = cfg.num_items + 1
    want_maxlen = cfg.max_seq_len + 1
    if ckpt_items != want_items or ckpt_maxlen != want_maxlen:
        raise RuntimeError(
            "Checkpoint/config mismatch.\n"
            f"  checkpoint at {path}: num_items={ckpt_items - 1}, "
            f"max_seq_len={ckpt_maxlen - 1}\n"
            f"  config DATASET='{cfg.dataset}': num_items={cfg.num_items}, "
            f"max_seq_len={cfg.max_seq_len}\n"
            "Fix: set DATASET (and if needed CHECKPOINT_DIR) so the config "
            "matches the dataset these checkpoints were trained on. "
            "(num_items=3706 -> movielens_1m, num_items=12101 -> amazon_beauty.)"
        )


def load_checkpoints(cfg: Config):
    _assert_checkpoint_matches(cfg)
    device = cfg.device
    retriever = SASRec(
        cfg.num_items, cfg.ret_emb_dim, cfg.max_seq_len,
        cfg.ret_num_heads, cfg.ret_num_layers, cfg.ret_dropout,
    ).to(device)
    retriever.load_state_dict(torch.load(
        os.path.join(cfg.checkpoint_dir, "sasrec_retriever.pt"), map_location=device))
    retriever.eval()

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


def make_pareto_figure(mars_pt, relevance_pt, sweeps, save_path):
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
    ax.set_title(f"Relevance-diversity Pareto frontier ({DATASET})")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Pareto figure saved to {save_path}")


def main():
    cfg = Config(dataset=DATASET)
    if CHECKPOINT_DIR is not None:
        cfg.checkpoint_dir = CHECKPOINT_DIR
    data = load_and_preprocess(cfg.data_dir, cfg.dataset)
    cfg.num_items = data["num_items"]
    cfg.num_users = data["num_users"]
    _, _, test_seqs = split_data(data["sequences"])

    print("Loading trained checkpoints...")
    retriever, faiss_index, diversity_module, state_encoder, actor = load_checkpoints(cfg)
    item_embs = get_item_embeddings_for_ild(diversity_module, cfg.num_items)

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
    print(f"Baseline comparison on {DATASET} (test set)")
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
        "dataset": DATASET,
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
        os.path.join(cfg.checkpoint_dir, "pareto_frontier.png"))


if __name__ == "__main__":
    main()