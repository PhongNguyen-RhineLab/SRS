"""
Final test-set evaluation reproducing Table 2 (HR@10/NDCG@10/MRR@10/ILD/Coverage)
and Figure 2 (diversity-relevance trade-off scatter plot).

Example:
    python run_test_eval.py
"""

import os
import json
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from data import load_and_preprocess, split_data
from retriever import SASRec, FAISSIndex
from diversity import DiversityModule
from rl_policy import StateEncoder, Actor
from evaluate import evaluate_srs, evaluate_icsrec_greedy, get_item_embeddings_for_ild


def load_checkpoints(cfg: Config):
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


def make_figure2(srs_metrics: dict, baseline_metrics: dict, save_path: str):
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(baseline_metrics["HR@k"], baseline_metrics["ILD"],
              color="red", marker="D", s=80, label="ICSRec greedy top-10")
    ax.scatter(srs_metrics["HR@k"], srs_metrics["ILD"],
              color="blue", marker="*", s=150, label="SRS (ours)")

    ax.annotate(
        f"HR@10={baseline_metrics['HR@k']:.4f}\nILD={baseline_metrics['ILD']:.4f}",
        (baseline_metrics["HR@k"], baseline_metrics["ILD"]),
        textcoords="offset points", xytext=(10, -15), color="red", fontsize=9,
    )
    ax.annotate(
        f"HR@10={srs_metrics['HR@k']:.4f}\nILD={srs_metrics['ILD']:.4f}",
        (srs_metrics["HR@k"], srs_metrics["ILD"]),
        textcoords="offset points", xytext=(10, 10), color="blue", fontsize=9,
    )

    ax.set_xlabel("HR@10")
    ax.set_ylabel("ILD (Intra-List Diversity)")
    ax.set_title("Diversity-relevance trade-off on Amazon Beauty")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Figure 2 reproduction saved to {save_path}")


def main():
    cfg = Config()
    data = load_and_preprocess(cfg.data_dir)
    cfg.num_items = data["num_items"]
    cfg.num_users = data["num_users"]
    _, _, test_seqs = split_data(data["sequences"])

    print("Loading trained checkpoints...")
    retriever, faiss_index, diversity_module, state_encoder, actor = load_checkpoints(cfg)
    item_embs = get_item_embeddings_for_ild(diversity_module, cfg.num_items)

    print("\nEvaluating ICSRec top-10 (greedy) baseline on test set...")
    baseline_metrics = evaluate_icsrec_greedy(cfg, test_seqs, retriever, faiss_index, item_embs)

    print("\nEvaluating SRS (ours) on test set...")
    srs_metrics = evaluate_srs(cfg, test_seqs, retriever, faiss_index,
                              diversity_module, state_encoder, actor, item_embs)

    print("\n" + "=" * 60)
    print("Table 2: Test-set comparison on Amazon Beauty 2014 5-core")
    print("=" * 60)
    print(f"{'Method':<25}{'HR@10':<10}{'NDCG@10':<10}{'MRR@10':<10}{'ILD':<10}{'Coverage':<10}")
    print(f"{'ICSRec top-10 (greedy)':<25}"
          f"{baseline_metrics['HR@k']:<10.4f}{baseline_metrics['NDCG@k']:<10.4f}"
          f"{baseline_metrics['MRR@k']:<10.4f}{baseline_metrics['ILD']:<10.4f}"
          f"{baseline_metrics['Coverage']:<10.4f}")
    print(f"{'Ours (SRS)':<25}"
          f"{srs_metrics['HR@k']:<10.4f}{srs_metrics['NDCG@k']:<10.4f}"
          f"{srs_metrics['MRR@k']:<10.4f}{srs_metrics['ILD']:<10.4f}"
          f"{srs_metrics['Coverage']:<10.4f}")

    ild_change = (srs_metrics["ILD"] - baseline_metrics["ILD"]) / baseline_metrics["ILD"] * 100
    hr_change = (srs_metrics["HR@k"] - baseline_metrics["HR@k"]) / baseline_metrics["HR@k"] * 100
    print(f"\nILD change: {ild_change:+.1f}%   HR@10 change: {hr_change:+.1f}%")

    results = {
        "icsrec_top10_greedy": baseline_metrics,
        "srs_ours": srs_metrics,
        "ild_pct_change": ild_change,
        "hr_pct_change": hr_change,
    }
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    with open(os.path.join(cfg.checkpoint_dir, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {cfg.checkpoint_dir}/test_results.json")

    make_figure2(srs_metrics, baseline_metrics,
                os.path.join(cfg.checkpoint_dir, "figure2_tradeoff.png"))


if __name__ == "__main__":
    main()
