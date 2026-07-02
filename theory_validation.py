"""
Reproduces Table IV (theoretical quantities) and Figure 3 (alpha -> delta_bar(alpha)
curve with annotated points) from the theory paper, Section V-E.

In the paper draft, Table IV's "Value" column is literally placeholder text
("[measure]", "[compute]", "[estimate]") — this script computes real numbers
for those cells from the trained retriever + diversity module.

Example:
    python theory_validation.py
"""

import os
import csv
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config

# Set to "movielens_1m" to run against MovieLens-1M instead.
# Set to "movielens_1m" to run against MovieLens-1M instead.
DATASET = "amazon_beauty"
from data import load_and_preprocess, split_data
from retriever import FAISSIndex
from diversity import DiversityModule
from evaluate import compute_pool_statistics, compute_average_deficit
from theory import compute_alpha_star, compute_Ck, worst_case_deficit, asymptotic_Ck


# Reference alpha values from Table III of the paper: the converged
# mean-alpha WITHOUT bias initialization (alpha-collapse, 0.504) and WITH
# bias initialization (the paper's reported learned value, 0.617). These
# are quoted directly from the paper, not measured from this repo's own
# retraining (our retriever differs from ICSRec-SAS, see README), so treat
# them as fixed reference points rather than "our" measured alphas.
# Reference trade-off values for the delta/delta_bar table and Figure 3.
# 0.504 and 0.617 are the collapsed/learned alphas quoted by the ORIGINAL
# (pre-ICSRec) experiments; they no longer describe the current checkpoints
# and are kept only as fixed reference points on the curve. The verdict
# below uses the alpha this repo's own training actually realized.
REF_ALPHA_COLLAPSED = 0.504
REF_ALPHA_LEARNED = 0.617
REF_ALPHA_INIT = 0.9
PAPER_ALPHA_COLLAPSED = REF_ALPHA_COLLAPSED  # backward-compat aliases
PAPER_ALPHA_LEARNED = REF_ALPHA_LEARNED
PAPER_ALPHA_INIT = REF_ALPHA_INIT


def get_own_learned_alpha(cfg: Config) -> float:
    """
    If this repo's own train_rl.py was run, training_log.csv has a
    'mean_alpha' column logged per epoch. Returns the mean of the last 10
    epochs as this run's own converged alpha (separate from the paper's
    quoted 0.504 / 0.617, since our retriever differs).

    Returns None if the log isn't found.
    """
    log_path = os.path.join(cfg.checkpoint_dir, "training_log.csv")
    if not os.path.exists(log_path):
        return None

    alphas = []
    with open(log_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for row in rows[-10:]:
        try:
            alphas.append(float(row["mean_alpha"]))
        except (KeyError, ValueError):
            return None
    return float(np.mean(alphas)) if alphas else None


def main():
    cfg = Config(dataset=DATASET)
    data = load_and_preprocess(cfg.data_dir, cfg.dataset)
    cfg.num_items = data["num_items"]
    cfg.num_users = data["num_users"]
    _, _, test_seqs = split_data(data["sequences"])

    device = cfg.device
    print("Loading retriever + FAISS index + diversity module...")

    from icsrec_retriever import load_icsrec_retriever
    retriever = load_icsrec_retriever(
        cfg.icsrec_ckpt, cfg.num_items, hidden_size=cfg.ret_emb_dim, device=device)

    faiss_index = FAISSIndex.load(
        os.path.join(cfg.checkpoint_dir, "faiss_index.npy"), cfg.ret_emb_dim)

    diversity_module = DiversityModule(cfg.num_items, cfg.div_emb_dim).to(device)
    diversity_module.load_state_dict(torch.load(
        os.path.join(cfg.checkpoint_dir, "diversity_module.pt"), map_location=device))
    diversity_module.eval()

    # ---------------- pool statistics: r_perp, kappa_max (Table IV) ----------------
    print("\nComputing pool statistics (r_perp, kappa_max) over the test set...")
    stats = compute_pool_statistics(cfg, test_seqs, retriever, faiss_index, diversity_module)
    r_perp, kappa_max = stats["r_perp"], stats["kappa_max"]
    print(f"  r_perp = {r_perp:.4f}  kappa_max = {kappa_max:.4f}  (n_users={stats['n_users']})")

    # ---------------- alpha*, Ck (closed form, Eq. 7 and 12) ----------------
    alpha_star = compute_alpha_star(r_perp, kappa_max, cfg.k)
    Ck = compute_Ck(cfg.k)
    Ck_asymp = asymptotic_Ck(cfg.k)
    print(f"  alpha* = {alpha_star:.4f}   Ck = {Ck:.4f}  (asymptotic (k-1)/e = {Ck_asymp:.4f})")

    # ---------------- worst-case deficit bound, Eq. 6 ----------------
    delta_collapsed = worst_case_deficit(PAPER_ALPHA_COLLAPSED, r_perp, kappa_max, cfg.k)
    delta_learned = worst_case_deficit(PAPER_ALPHA_LEARNED, r_perp, kappa_max, cfg.k)

    # ---------------- average deficit via greedy rollouts, Definition 4 ----------------
    print("\nEstimating average deficit delta_bar(alpha) via greedy rollouts...")
    delta_bar_collapsed = compute_average_deficit(
        cfg, test_seqs, retriever, faiss_index, diversity_module, PAPER_ALPHA_COLLAPSED)
    delta_bar_learned = compute_average_deficit(
        cfg, test_seqs, retriever, faiss_index, diversity_module, PAPER_ALPHA_LEARNED)

    own_alpha = get_own_learned_alpha(cfg)

    # ---------------- print Table IV ----------------
    print("\n" + "=" * 70)
    print("Table IV: Theoretical quantities on the Amazon Beauty test set")
    print("=" * 70)
    rows = [
        ("r_perp (min pool relevance score)", f"{r_perp:.4f}"),
        ("kappa_max (max kernel value)", f"{kappa_max:.4f}"),
        ("alpha* (monotone threshold, Eq. 7)", f"{alpha_star:.4f}"),
        (f"delta({PAPER_ALPHA_COLLAPSED}) -- collapsed policy [worst-case]", f"{delta_collapsed:.4f}"),
        (f"delta({PAPER_ALPHA_LEARNED}) -- learned policy [worst-case]", f"{delta_learned:.4f}"),
        (f"delta_bar({PAPER_ALPHA_COLLAPSED}) -- collapsed policy [rollout avg]", f"{delta_bar_collapsed:.4f}"),
        (f"delta_bar({PAPER_ALPHA_LEARNED}) -- learned policy [rollout avg]", f"{delta_bar_learned:.4f}"),
        (f"C{cfg.k} (closed form, Eq. 12)", f"{Ck:.4f}"),
    ]
    for name, val in rows:
        print(f"  {name:<55}{val}")

    if own_alpha is not None:
        print(f"\n  (This repo's own train_rl.py converged to alpha_bar = {own_alpha:.4f}, "
              f"computed from the last 10 epochs of training_log.csv. The 0.504/0.617 "
              f"rows above are pre-ICSRec reference points, not this checkpoint's "
              f"realized trade-off.)")

    verdict_alpha = own_alpha if own_alpha is not None else REF_ALPHA_LEARNED
    verdict_src = "this run's realized" if own_alpha is not None else "reference"
    is_monotone_at_learned = verdict_alpha >= alpha_star
    print(f"\n  alpha*={alpha_star:.4f} vs {verdict_src} alpha={verdict_alpha:.4f}: "
          f"{'IN monotone regime (Theorem 1 applies)' if is_monotone_at_learned else 'BELOW threshold (near-monotone bound, Theorem 2, applies)'}")

    results = {
        "r_perp": r_perp,
        "kappa_max": kappa_max,
        "alpha_star": alpha_star,
        "Ck": Ck,
        "Ck_asymptotic": Ck_asymp,
        "delta_worst_case": {
            str(PAPER_ALPHA_COLLAPSED): delta_collapsed,
            str(PAPER_ALPHA_LEARNED): delta_learned,
        },
        "delta_bar_rollout": {
            str(PAPER_ALPHA_COLLAPSED): delta_bar_collapsed,
            str(PAPER_ALPHA_LEARNED): delta_bar_learned,
        },
        "own_converged_alpha": own_alpha,
        "is_learned_alpha_monotone": is_monotone_at_learned,
    }
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    with open(os.path.join(cfg.checkpoint_dir, "theory_validation.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {cfg.checkpoint_dir}/theory_validation.json")

    # ---------------- Figure 3: alpha -> delta_bar(alpha) curve ----------------
    print("\nBuilding Figure 3 (alpha -> delta_bar(alpha) curve)...")
    print("  (using a 100-user subsample per grid point to keep runtime reasonable)")
    alpha_grid = np.linspace(0.0, 1.0, 11)
    delta_bar_curve = []
    for a in alpha_grid:
        d = compute_average_deficit(cfg, test_seqs, retriever, faiss_index,
                                   diversity_module, float(a), max_users=100)
        delta_bar_curve.append(d)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(alpha_grid, delta_bar_curve, color="black", linewidth=1.5,
           label=r"$\bar{\delta}(\alpha)$ (rollout average)")

    annotated = [
        (PAPER_ALPHA_COLLAPSED, "reference: collapsed\n(pre-ICSRec)", "red"),
        (PAPER_ALPHA_LEARNED, "reference: learned\n(pre-ICSRec)", "blue"),
        (alpha_star, r"$\alpha^*$ (threshold)", "green"),
        (PAPER_ALPHA_INIT, "initialization", "purple"),
    ]
    if own_alpha is not None:
        annotated.append((own_alpha, "this run\n(own training_log.csv)", "darkorange"))
    for i, (a, label, color) in enumerate(annotated):
        d = compute_average_deficit(cfg, test_seqs, retriever, faiss_index,
                                   diversity_module, a, max_users=100)
        ax.scatter([a], [d], color=color, s=80, zorder=5)
        # alternate offset direction so nearby points' labels don't overlap
        y_off = 10 if i % 2 == 0 else -22
        ax.annotate(label, (a, d), textcoords="offset points", xytext=(8, y_off),
                   fontsize=8, color=color)

    ax.axvline(alpha_star, color="green", linestyle="--", alpha=0.4)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\bar{\delta}(\alpha)$ (average deficit)")
    ax.set_title("Average deficit vs. relevance-diversity trade-off")
    ax.legend()
    fig.tight_layout()

    fig_path = os.path.join(cfg.checkpoint_dir, "figure3_delta_bar_curve.png")
    fig.savefig(fig_path, dpi=150)
    print(f"Figure 3 reproduction saved to {fig_path}")


if __name__ == "__main__":
    main()