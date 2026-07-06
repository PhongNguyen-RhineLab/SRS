"""
plot_paper_figures.py
=====================

Generates the three paper figures from artifacts already saved by
run_baselines.py, diagnose.py, and theory_validation.py. Pure post-processing:
no model is loaded, so it runs in seconds and the figures are exactly
reproducible from the committed JSON/NPY files.

Inputs (in cfg.checkpoint_dir):
  baseline_results.json   (run_baselines.py)
  alpha_values.npy        (diagnose.py -- per-user deterministic alpha_t)
  theory_validation.json  (theory_validation.py)
  deficit_curve.json      (theory_validation.py -- alpha grid of delta_bar)
  training_log.csv        (train_rl.py, optional -- training-stability figure)

Outputs (PDF + PNG each):
  fig_pareto      relevance-diversity frontier, MARS vs baseline sweeps
  fig_policy      realized alpha_t distribution vs the alpha* threshold
  fig_theory      worst-case delta(alpha) vs measured delta_bar(alpha)
  fig_training    training stability (only if training_log.csv present)

Example:
    python plot_paper_figures.py
"""

import os
import csv
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config

DATASET = "amazon_beauty"

# -- shared style ------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})
C_FIXED = "#0072B2"   # blue
C_MMR = "#E69F00"     # orange
C_DPP = "#009E73"     # green
C_MARS = "#D55E00"    # vermillion
C_REL = "#000000"     # black
C_THR = "#CC79A7"     # magenta (alpha* threshold)


def _save(fig, ckpt_dir, name):
    for ext in ("pdf", "png"):
        path = os.path.join(ckpt_dir, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.pdf/.png")


# -- Figure 1: Pareto frontier ------------------------------------------------
def fig_pareto(ckpt_dir):
    r = json.load(open(os.path.join(ckpt_dir, "baseline_results.json")))
    rel, mars, sweeps = r["relevance_topk"], r["mars"], r["sweeps"]

    def xy(entries):
        e = sorted(entries, key=lambda d: d["ILD"])
        return [d["ILD"] for d in e], [d["HR@k"] for d in e], e

    fx, fy, fe = xy(sweeps["FixedAlpha"])
    mx, my, me = xy(sweeps["MMR"])
    dx, dy, de = xy(sweeps["DPP"])

    fig, ax = plt.subplots(figsize=(4.9, 3.5))

    ax.plot(fx, fy, "-o", color=C_FIXED, ms=3.5, lw=1.2,
            label=r"Fixed-$\alpha$ greedy (Eq. 5)")
    ax.plot(mx, my, "--s", color=C_MMR, ms=3.5, lw=1.2, label="MMR")
    ax.plot(dx, dy, ":^", color=C_DPP, ms=3.5, lw=1.2, label="DPP greedy MAP")
    ax.plot(rel["ILD"], rel["HR@k"], "D", color=C_REL, ms=6,
            label="Relevance top-$k$", zorder=5)
    ax.plot(mars["ILD"], mars["HR@k"], "*", color=C_MARS, ms=15,
            label="MARS (learned $\\alpha_t$)", zorder=6)

    # annotate a few fixed-alpha operating points
    for d in fe:
        if d["param"] in (0.9, 0.617, 0.3):
            ax.annotate(f"$\\alpha$={d['param']:g}", (d["ILD"], d["HR@k"]),
                        textcoords="offset points", xytext=(4, -10),
                        fontsize=7, color=C_FIXED)

    # zoom on the informative region; extreme points go to the inset
    ax.set_xlim(0.525, 0.66)
    ax.set_ylim(0.060, 0.092)
    ax.set_xlabel("ILD (intra-list diversity, retriever space)")
    ax.set_ylabel("HR@10")
    ax.legend(loc="lower left", frameon=False)

    # inset: full range, shows the DPP/MMR degradation tail
    axin = ax.inset_axes([0.62, 0.62, 0.36, 0.36])
    axin.plot(fx, fy, "-", color=C_FIXED, lw=1)
    axin.plot(mx, my, "--", color=C_MMR, lw=1)
    axin.plot(dx, dy, ":", color=C_DPP, lw=1)
    axin.plot(mars["ILD"], mars["HR@k"], "*", color=C_MARS, ms=8)
    axin.set_xlim(0.5, 0.95)
    axin.set_ylim(0.0, 0.095)
    axin.tick_params(labelsize=6)
    axin.set_title("full range", fontsize=6, pad=2)

    _save(fig, ckpt_dir, "fig_pareto")


# -- Figure 2: realized policy vs the monotone threshold ----------------------
def fig_policy(ckpt_dir):
    alphas = np.load(os.path.join(ckpt_dir, "alpha_values.npy"))
    tv = json.load(open(os.path.join(ckpt_dir, "theory_validation.json")))
    a_star = tv["alpha_star"]

    fig, ax = plt.subplots(figsize=(4.9, 3.0))

    ax.hist(alphas, bins=40, range=(0, 1), color=C_FIXED, alpha=0.85,
            edgecolor="white", linewidth=0.3)

    # regime shading: Theorem A right of alpha*, Theorem 1 left of it
    ax.axvspan(a_star, 1.0, color=C_THR, alpha=0.10)
    ax.axvline(a_star, color=C_THR, lw=1.4, ls="--")
    ax.axvline(alphas.mean(), color=C_MARS, lw=1.4)

    ymax = ax.get_ylim()[1]
    ax.text(a_star + 0.008, ymax * 0.96,
            f"$\\alpha^*={a_star:.3f}$\nmonotone regime\n(Theorem A)",
            fontsize=7.5, color=C_THR, va="top")
    ax.text(a_star - 0.012, ymax * 0.96,
            "near-monotone regime\n(Theorem 1)",
            fontsize=7.5, color="0.35", va="top", ha="right")
    ax.text(alphas.mean() - 0.012, ymax * 0.55,
            f"$\\bar\\alpha={alphas.mean():.3f}$\n(std {alphas.std():.3f})",
            fontsize=7.5, color=C_MARS, ha="right")

    ax.set_xlim(0, 1)
    ax.set_xlabel(r"deterministic policy output $\alpha_t$ (one per test user)")
    ax.set_ylabel("users")
    _save(fig, ckpt_dir, "fig_policy")


# -- Figure 3: worst-case bound vs measured deficit ---------------------------
def fig_theory(ckpt_dir):
    dc = json.load(open(os.path.join(ckpt_dir, "deficit_curve.json")))
    tv = json.load(open(os.path.join(ckpt_dir, "theory_validation.json")))
    a_star = tv["alpha_star"]
    grid = sorted(dc["grid"], key=lambda d: d["alpha"])
    A = [d["alpha"] for d in grid]
    W = [d["delta_worst"] for d in grid]
    B = [d["delta_bar"] for d in grid]
    ar = dc.get("at_realized", {})

    fig, ax = plt.subplots(figsize=(4.9, 3.2))

    ax.plot(A, W, "-", color="0.25", lw=1.4,
            label=r"$\delta(\alpha)$ worst-case (Theorem 1)")
    ax.plot(A, B, "-o", color=C_MARS, ms=4, lw=1.4,
            label=r"$\bar\delta(\alpha)$ measured (greedy rollouts)")
    ax.fill_between(A, B, W, color="0.85", alpha=0.5,
                    label="slack of the worst-case bound")

    ax.axvline(a_star, color=C_THR, lw=1.2, ls="--")
    ax.text(a_star, ax.get_ylim()[1] * 0.98, f" $\\alpha^*={a_star:.3f}$",
            fontsize=7.5, color=C_THR, va="top")

    if ar:
        ax.plot(ar["alpha"], ar["delta_bar_full"], "*", color=C_FIXED, ms=13,
                zorder=6,
                label=(f"realized $\\bar\\alpha={ar['alpha']:.3f}$: "
                       f"$\\bar\\delta={ar['delta_bar_full']:.4f}$"))

    ax.set_xlabel(r"trade-off $\alpha$")
    ax.set_ylabel("monotonicity deficit")
    ax.set_xlim(min(A), 1.0)
    ax.legend(loc="upper right", frameon=False)
    _save(fig, ckpt_dir, "fig_theory")


# -- Figure 4 (optional): training stability ----------------------------------
def fig_training(ckpt_dir):
    path = os.path.join(ckpt_dir, "training_log.csv")
    if not os.path.exists(path):
        return
    rows = list(csv.DictReader(open(path)))
    ep = [int(r["epoch"]) for r in rows]
    al = [float(r["mean_alpha"]) for r in rows]
    ac = [float(r["actor_loss"]) for r in rows]
    hr = [(int(r["epoch"]), float(r["val_HR@10"])) for r in rows
          if float(r["val_HR@10"]) > 0]

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.3))
    axes[0].plot(ep, ac, color=C_FIXED, lw=1)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("actor loss")
    axes[1].plot(ep, al, color=C_MARS, lw=1)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel(r"mean $\alpha_t$ (train)")
    axes[1].set_ylim(0, 1)
    axes[2].plot([e for e, _ in hr], [v for _, v in hr], "-o", color=C_DPP,
                 ms=2.5, lw=1)
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("val HR@10")
    fig.tight_layout()
    _save(fig, ckpt_dir, "fig_training")


def main():
    cfg = Config(dataset=DATASET)
    print(f"[{cfg.dataset}] generating paper figures from {cfg.checkpoint_dir}/")
    fig_pareto(cfg.checkpoint_dir)
    fig_policy(cfg.checkpoint_dir)
    fig_theory(cfg.checkpoint_dir)
    fig_training(cfg.checkpoint_dir)
    print("done")


if __name__ == "__main__":
    main()
