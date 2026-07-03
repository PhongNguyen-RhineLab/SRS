# MARS: Monotone-Aware Reinforcement Learning for Diversity-Constrained Slate Recommendation

Reference implementation for the MARS paper (ORLab, Phenikaa University).

MARS is a sequential slate-recommendation framework combining a frozen
pre-trained retriever (ICSRec), a learned diversity module, and an
actor-critic RL policy that controls the relevance-diversity trade-off
parameter alpha per interaction. The primary contribution is theoretical: a
near-monotone analysis showing when greedy submodular slate selection retains
an approximation guarantee under the non-monotone objective, with greedy
achieving at least (1 - 1/e) OPT - C_k * delta(alpha). The experiments here
serve as proof-of-concept validation of that framework, including honestly
reported negative findings (see Known findings below).

## Pipeline overview

```
Stage 0  fetch_data.py         download ICSRec sequence file + checkpoint
Stage 1  build_index.py        FAISS index over the frozen ICSRec item table
Stage 2  pretrain_encoder.py   warm-start the state encoder (next-item prediction)
Stage 3  train_rl.py           joint RL policy + diversity module training
Eval     run_test_eval.py      Table 2 (HR/NDCG/MRR/ILD/Coverage) + Figure 2
Eval     run_baselines.py      fixed-alpha / MMR / DPP sweeps + Pareto frontier
Eval     theory_validation.py  Table IV theoretical quantities (alpha*, C_k, delta)
Debug    diagnose.py           alpha distribution + ILD embedding-geometry audit
```

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10+, PyTorch 2.x. CPU works for evaluation; a GPU is recommended for
`train_rl.py`.

## Datasets

Four datasets are supported, all taken verbatim from the official
[ICSRec repository](https://github.com/QinHsiu/ICSRec) so that item ids line
up with the released pre-trained checkpoints (re-preprocessing raw dumps
would produce a different id mapping and silently misalign with the
checkpoint's embedding table):

| name (canonical) | aliases | sequences | items | source file |
|---|---|---|---|---|
| `amazon_beauty` | `beauty` | 22,363 | 12,101 | Beauty.txt |
| `movielens_1m` | `ml-1m`, `movielens` | 6,040 | 3,416 | ml-1m.txt |
| `amazon_sports` | `sports` | 35,598 | 18,357 | Sports_and_Outdoors.txt |
| `amazon_toys` | `toys` | 19,412 | 11,924 | Toys_and_Games.txt |

Every entry script takes the same flags, so switching dataset is a
command-line argument rather than an edit to the source:

```bash
python fetch_data.py --dataset toys      # or --all
python build_index.py --dataset toys
python pretrain_encoder.py --dataset toys
python train_rl.py --dataset toys
python run_baselines.py --dataset toys
python run_test_eval.py --dataset toys
python theory_validation.py --dataset toys
python diagnose.py --dataset toys
```

Optional overrides on every script: `--data-dir`, `--checkpoint-dir`.
Per-dataset outputs go to `checkpoints_<dataset>/` (Amazon Beauty keeps the
legacy `checkpoints/` and `data/` locations for backward compatibility).

Adding a fifth dataset means adding one entry to `DATASET_REGISTRY` in
`config.py` (txt file name, checkpoint name, history window `h`,
`max_seq_len`) -- nothing else changes.

## File map

Core pipeline:

- `config.py` -- all hyperparameters (Table 1) and `DATASET_REGISTRY`
- `cli.py` -- shared `--dataset` argument handling with aliases
- `fetch_data.py` -- downloads ICSRec txt + checkpoint from GitHub
- `data.py` -- ICSRec sequence loading, LL2O split, retriever DataLoader
- `icsrec_retriever.py` -- vendored inference-only ICSRec (SASRec backbone)
  adapter; auto-infers item size and sequence length from checkpoint tensors
  and preflights checkpoint/dataset compatibility
- `retriever.py` -- FAISS index wrapper (plus the legacy self-trained SASRec
  class, used only by `smoke_test.py` and `legacy/`)
- `diversity.py` -- learned diversity module (submodular kernel)
- `rl_policy.py` -- StateEncoder, Actor (squashed Gaussian with Jacobian
  correction), Critic
- `env.py` -- slate environment; `buffer.py` -- replay buffer
- `theory.py` -- closed-form alpha*, C_k, worst-case deficit
- `metrics.py`, `evaluate.py` -- HR/NDCG/MRR/ILD/Coverage and eval loops

Entry scripts: `build_index.py`, `pretrain_encoder.py`, `train_rl.py`,
`run_test_eval.py`, `run_baselines.py` (+ `baselines.py` with the MMR / DPP /
fixed-alpha implementations), `theory_validation.py`, `diagnose.py`,
`smoke_test.py` (synthetic-data sanity check, no downloads needed).

`legacy/` holds the retired pre-ICSRec pipeline (`train_retriever.py`,
`diagnose_collapse.py`); see `legacy/README.md`.

## Known findings (reported honestly in the paper)

- Under the paper's literal hit/rank-only reward (Eq. 9), alpha = 1 is
  reward-optimal for every user state, so policy drift toward pure relevance
  is theoretically expected. `diversity_reward_weight` in `config.py` is 0.0
  by default to stay faithful to Eq. 9; the comment there documents the
  observed collapse and the deviation required to prevent it.
- On MovieLens-1M the trained policy produces per-user alpha variation
  (mean 0.84, std 0.20) but lands on, not beyond, the fixed-alpha / MMR /
  DPP Pareto frontier. MARS improves slightly over the pure relevance
  corner (+0.3% HR@10, +0.8% ILD) and is matched or dominated by several
  tuned baselines. The paper reports this as a proof-of-concept result for
  the theory, not an empirical state-of-the-art claim.
- `diagnose.py` audits whether ILD is meaningful on a given dataset by
  comparing within-pool to global embedding distances (ratio well below 1.0
  means retrieved items are genuinely more similar than random pairs and ILD
  carries signal). Run it once per dataset before quoting ILD numbers.

## Advantage-normalization stabilizers

The paper's literal Eq. (11) (advantage normalization with ~0 epsilon and no
clipping) is numerically unstable when most transitions share the identical
miss-penalty reward: sigma collapses and single samples get unbounded
advantages, exploding the actor loss. `config.py` exposes `advantage_eps`,
`advantage_clip`, and `logp_clip_min` as standard, documented stabilizers;
the comments in `config.py` and `train_rl.py` record the exact observed
failure modes that motivated each one.

## Reproducing the paper tables

```bash
# per dataset (example: beauty)
python fetch_data.py -d beauty
python build_index.py -d beauty
python pretrain_encoder.py -d beauty
python train_rl.py -d beauty            # ~100 epochs
python run_test_eval.py -d beauty       # Table 2, Figure 2
python run_baselines.py -d beauty       # baseline sweeps, Pareto frontier
python theory_validation.py -d beauty   # Table IV
```

Outputs (JSON results, PNG figures, training log CSV) are written to the
dataset's checkpoint directory.
