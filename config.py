"""
Hyperparameters matching Table 1 of the SRS paper.

Example: to change slate size from 10 to 5 just do cfg.k = 5 before passing cfg around.
"""

import torch
from dataclasses import dataclass, field


@dataclass
class Config:
    # ------------------------------------------------------------------ data
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"

    # will be overwritten after dataset is loaded
    num_items: int = 12101
    num_users: int = 22363

    # ----------------------------------------------------------- slate / pipeline
    k: int = 10           # final slate size
    m: int = 200          # candidates retrieved by FAISS
    h: int = 20           # state-encoder history window (h=20 in SRS, h=50 in ICSRec eval)
    max_seq_len: int = 50 # SASRec input length

    # ------------------------------------------------------ SASRec retriever
    ret_emb_dim: int = 64
    ret_num_heads: int = 2
    ret_num_layers: int = 2
    ret_dropout: float = 0.2
    ret_lr: float = 1e-3
    ret_epochs: int = 50
    ret_batch_size: int = 256

    # ---------------------------------------------------- diversity module
    div_emb_dim: int = 64   # item embedding dimension (Table 1)
    div_margin: float = 0.3  # margin m in L_div_rank (not stated, reasonable default)
    div_clamp: float = 1e-7  # numerical clamp in log

    # ------------------------------------------------------- state encoder
    state_dim: int = 128    # d_s (Table 1)

    # --------------------------------------------------------- actor-critic
    hidden_dim: int = 256            # Table 1
    alpha_init_bias: float = 2.197   # logit(0.9), Section 4.3

    # ------------------------------------------------------------------ RL
    gamma: float = 0.9
    beta_bc: float = 0.01   # BC regularizer on eta only
    beta_ent: float = 0.005  # entropy coefficient
    lambda_rank: float = 0.1 # Table 1
    rho: float = 0.2          # rank-bonus coefficient, Eq. (9)
    p_miss: float = 0.01      # miss penalty, Eq. (9)

    # ---------------------------------------------------------- exploration
    tau_min: float = 0.1
    tau_max: float = 1.0

    # ------------------------------------------------------- training loop
    buffer_size: int = 10000
    batch_size: int = 32
    lr_rl: float = 3e-4
    lr_sub: float = 1e-3
    steps_per_epoch: int = 500
    num_epochs: int = 100
    eval_every: int = 5

    # ---------------------------------------------------------------- misc
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
