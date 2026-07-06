# MARS: Monotone-Aware Reinforcement Learning for Diversity-Constrained Slate Recommendation

Reference implementation and theory-validation code for the MARS paper: a three-stage sequential-recommendation pipeline (frozen dense retriever, learned diversity module, actor-critic RL policy) with a provable near-monotone approximation guarantee.

## Overview

**Problem.** In slate recommendation the goal is to build a short list (a *slate*) that is both relevant and diverse. This is usually posed as submodular maximization, where a simple greedy selector enjoys the classical `(1 - 1/e)` approximation guarantee, but only when the objective is *monotone*. Once the relevance-diversity trade-off is learned adaptively (per user, per step), the objective can become non-monotone and that guarantee no longer applies.

**Method.** MARS learns a per-user trade-off parameter `alpha_t` with an actor-critic policy while keeping the greedy slate selector. Its main contribution is theoretical: a computable monotonicity condition (a threshold `alpha*`), a greedy approximation bound in the near-monotone regime of the form `F(S_greedy) >= (1 - 1/e)*OPT - C_k * delta(alpha)`, and an extension of that bound to stochastic policies. The paper also shows that a hit/rank-only reward drives the policy toward the pure-relevance corner (`alpha -> 1`), and gives the reward-shaping condition under which adaptive behavior is recovered without losing the guarantee.

**What is in this repository.** End-to-end runnable code: data fetching, FAISS index construction over the frozen ICSRec backbone, state-encoder pretraining, RL training, baseline sweeps (relevance top-k, fixed-alpha, MMR, DPP), test-set evaluation, theory validation, and figure generation. Trained checkpoints, result JSONs, and paper figures for four datasets (Beauty, Sports, Toys, MovieLens-1M) are committed under the `checkpoints*` folders, so results can be inspected or re-plotted without retraining.

## Repository Structure

```
MARS/
├── cli.py                    Shared argument parsing (--dataset, --data-dir, --checkpoint-dir) with aliases
├── config.py                 Config dataclass (Table 1 hyperparameters) + DATASET_REGISTRY
├── data.py                   Sequence loading and Leave-Last-2-Out (LL2O) split
├── retriever.py              SASRec model and FAISS inner-product index
├── icsrec_retriever.py       Adapter that loads the frozen released ICSRec-SAS checkpoints
├── diversity.py              Learned diversity module (item embeddings + ranking loss)
├── rl_policy.py              StateEncoder, Actor, Critic networks
├── buffer.py                 Replay buffer
├── env.py                    Slate-construction environment (greedy selector)
├── theory.py                 alpha*, C_k, and worst-case deficit computations
├── metrics.py                HR@10, NDCG@10, MRR@10, ILD, Coverage
├── evaluate.py               Retrieval-constrained evaluation routines
├── baselines.py              MMR / DPP / fixed-alpha sweeps
│
│   # --- entry scripts (run in this order for a new dataset) ---
├── fetch_data.py             Download sequence file + ICSRec checkpoint from the ICSRec repo
├── build_index.py            Build the FAISS index from the frozen ICSRec item table
├── pretrain_encoder.py       Pretrain and freeze the RL state encoder (next-item prediction)
├── train_rl.py               Joint RL policy + diversity module training (Section 4.3)
├── run_baselines.py          MARS vs. relevance / fixed-alpha / MMR / DPP; Pareto frontier
├── run_test_eval.py          Final test-set metrics (Table 2) + trade-off scatter (Figure 2)
├── theory_validation.py      Measures alpha*, delta, delta_bar for Table 4 / Figure 3
├── plot_paper_figures.py     Regenerates paper figures from committed JSON/NPY artifacts
├── diagnose.py               alpha distribution and embedding-degeneracy diagnostics
├── smoke_test.py             Synthetic end-to-end sanity check (no downloads)
│
├── icsrec_ckpts/             Frozen released ICSRec-SAS backbones (e.g. Beauty, ml-1m)
├── checkpoints/              Trained artifacts + figures + results for Amazon Beauty (default)
├── checkpoints_amazon_sports/    ... for Amazon Sports and Outdoors
├── checkpoints_amazon_toys/      ... for Amazon Toys and Games
├── checkpoints_movielens_1m/     ... for MovieLens-1M
│
├── requirements.txt
└── README.md
```

Each `checkpoints*` folder holds the trained networks (`actor.pt`, `critic.pt`, `diversity_module.pt`, `state_encoder.pt`), the FAISS index (`faiss_index.npy`), logged values (`alpha_values.npy`, `eta_values.npy`, `training_log.csv`), result files (`test_results.json`, `baseline_results.json`, `theory_validation.json`, `deficit_curve.json`), and the generated figures.

## Requirements

- **Language:** Python 3.9 or newer.
- **Main libraries** (see `requirements.txt`): `torch >= 2.0`, `faiss-cpu >= 1.7`, `numpy >= 1.24`, `tqdm >= 4.65`, `requests >= 2.28`, `matplotlib >= 3.7`.
- **Hardware:** A GPU is optional. The device is selected automatically (`cuda` if available, otherwise `cpu`; see `Config.device` in `config.py`). Training on GPU is recommended; the datasets and models are small (backbone checkpoints are around 5 MB, embedding dimension 64), so memory requirements are modest. `faiss-cpu` is used, so no GPU FAISS build is needed.

## Installation

```bash
# 1. Clone
git clone https://github.com/PhongNguyen-RhineLab/MARS.git
cd MARS

# 2. Create an environment (example uses venv; conda works too)
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

If you need GPU FAISS instead of the CPU build, replace `faiss-cpu` with the appropriate `faiss-gpu` package for your CUDA version.

## Dataset

**Source.** All four datasets come from the official ICSRec repository (`https://github.com/QinHsiu/ICSRec`). Each dataset is a `.txt` sequence file paired with a released SASRec-backbone checkpoint. Both are downloaded automatically:

```bash
python fetch_data.py --dataset beauty     # one dataset
python fetch_data.py --all                # all supported datasets
```

**Preprocessing.** Sequences are used verbatim, with no re-preprocessing. Each line is `user_id item1 item2 ... itemN`, with item IDs already 1-indexed in the *same* ID space as the released checkpoints. This is deliberate: re-preprocessing the raw Amazon/MovieLens dumps would produce a different ID mapping and silently misalign sequences with the checkpoint's item-embedding table. The train/val/test split is Leave-Last-2-Out (LL2O). For example, a sequence `[A, B, C, D, E]` gives train history `[A, B, C]`, validation target `D`, and test target `E`.

**Expected layout.** `fetch_data.py` writes the sequence file into the dataset's `data_dir` and the checkpoint into `icsrec_ckpts/`, following the per-dataset paths in `DATASET_REGISTRY` (`config.py`). Supported keys and their aliases:

| Registry key    | Aliases                                | Sequence file              |
|-----------------|----------------------------------------|----------------------------|
| `amazon_beauty` | `beauty`                               | `Beauty.txt`               |
| `amazon_sports` | `sports`, `sports_and_outdoors`        | `Sports_and_Outdoors.txt`  |
| `amazon_toys`   | `toys`, `toys_and_games`               | `Toys_and_Games.txt`       |
| `movielens_1m`  | `ml-1m`, `ml1m`, `movielens`           | `ml-1m.txt`                |

**Access / licensing.** The Amazon Review and MovieLens-1M datasets are released by their original authors (McAuley et al., 2015 for Amazon; Harper and Konstan, 2015 for MovieLens) under their own terms. Please consult those original sources for licensing and usage restrictions. This repository does not redistribute the raw datasets; it fetches the preprocessed sequence files from the ICSRec repository at run time.

## Running the Experiments

The full pipeline for a new dataset, in order (shown for `sports`; swap in any dataset name or alias):

```bash
python fetch_data.py        --dataset sports   # download sequences + ICSRec checkpoint
python build_index.py       --dataset sports   # build FAISS index from frozen backbone
python pretrain_encoder.py  --dataset sports   # pretrain + freeze the state encoder
python train_rl.py          --dataset sports   # train the actor-critic policy + diversity module
```

**Evaluation and analysis** (require the trained checkpoints above):

```bash
python run_test_eval.py     --dataset sports   # Table 2 metrics + Figure 2 trade-off plot
python run_baselines.py     --dataset sports   # MARS vs. relevance / fixed-alpha / MMR / DPP + Pareto frontier
python theory_validation.py --dataset sports   # measure alpha*, delta, delta_bar (Table 4 / Figure 3)
python diagnose.py          --dataset sports   # alpha distribution + embedding-degeneracy checks
```

**Figures.** Once the JSON/NPY artifacts exist, `plot_paper_figures.py` regenerates the paper figures in seconds without loading any model:

```bash
python plot_paper_figures.py
```

**Quick sanity check.** To exercise every module end-to-end on tiny synthetic data (no downloads, no GPU needed):

```bash
python smoke_test.py
```

There is no separate "inference" script: MARS produces a slate as part of evaluation. `evaluate.py` runs the deterministic policy (the actor emits `alpha_t`, and the greedy selector builds the slate) inside the retrieval-constrained pipeline, which is exactly the serving-time behavior.

## Configuration

Experiment settings are controlled in two places:

- **`config.py`** holds the `Config` dataclass, whose fields mirror Table 1 of the paper. Per-dataset defaults (sequence file, checkpoint name, history window `h`, `max_seq_len`, directories) live in `DATASET_REGISTRY`; passing `Config(dataset="amazon_toys")` pulls the right defaults. To change a single value, set it before use, for example `cfg.k = 5` to shrink the slate.
- **`cli.py`** provides the command-line flags shared by every entry script: `--dataset` / `-d` (with the aliases above), `--data-dir`, and `--checkpoint-dir` to override the per-dataset directories.

Important hyperparameters:

| Parameter                 | Field                     | Default | Meaning                                                        |
|---------------------------|---------------------------|---------|----------------------------------------------------------------|
| Slate size                | `k`                       | 10      | Number of items in the final slate                             |
| Candidate pool            | `m`                       | 200     | Items retrieved by FAISS before reranking                      |
| History window            | `h`                       | 20 / 50 | State-encoder window (50 for ml-1m)                            |
| Alpha init bias           | `alpha_init_bias`         | 2.197   | `logit(0.9)`; biases the policy toward relevance at start      |
| Diversity reward weight   | `diversity_reward_weight` | 0.0     | Off by default (faithful to the paper's hit/rank-only reward); a small positive value adds `weight * ILD` to prevent alpha collapse |
| Entropy coefficient       | `beta_ent`                | 0.005   | Exploration bonus in the actor loss                            |
| Discount                  | `gamma`                   | 0.9     | RL discount factor                                             |
| Epochs                    | `num_epochs`              | 100     | Training epochs                                                |

The config also documents two advantage-stabilization knobs (`advantage_eps`, `advantage_clip`) and a stale-log-prob floor (`logp_clip_min`) that were added after an actor-loss blow-up was traced to tightly clustered TD residuals in the literal Eq. (11); the comments in `config.py` explain each one.

## Citation

Pending.

## License

MIT.
