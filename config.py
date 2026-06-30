"""
Hyperparameters matching Table 1 of the SRS paper.

Example: to change slate size from 10 to 5 just do cfg.k = 5 before passing cfg around.
"""

import torch
from dataclasses import dataclass, field


@dataclass
class Config:
    # ------------------------------------------------------------------ data
    # "amazon_beauty" (default, matches the paper) or "movielens_1m".
    # Pass as a constructor kwarg, e.g. Config(dataset="movielens_1m"), so
    # __post_init__ below can pick sensible per-dataset directory and
    # sequence-length defaults. Setting cfg.dataset = ... AFTER construction
    # will NOT retroactively update data_dir/checkpoint_dir/h/max_seq_len --
    # set it at construction time, or set those fields yourself too.
    dataset: str = "movielens_1m"

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

    # -------------------------------------------------- advantage stabilization
    # See compute_normalized_advantage in train_rl.py for why these exist:
    # the paper's literal Eq. (11) has no clip and an implicit ~0 epsilon,
    # which blows up whenever a mini-batch's TD residuals happen to cluster
    # tightly (very plausible given ~75-80% identical miss-penalty
    # transitions) -- confirmed as the cause of an actor_loss explosion from
    # -309 at epoch 1 to -1.4e8 by epoch 30 in a real training run.
    advantage_eps: float = 1e-3
    advantage_clip: float = 10.0

    # -------------------------------------------------- diversity reward shaping
    # OFF (0.0) by default to stay faithful to the paper's literal Eq. (9),
    # which rewards hit/rank only. Flagging this because with reward defined
    # that way, nothing prevents alpha from drifting all the way to 1.0 over
    # training (confirmed: mean_alpha hit 1.000 by epoch 24 and stayed there
    # for the rest of a 100-epoch run) -- the reward never directly values
    # diversity, so policy-gradient pressure has no reason not to saturate
    # alpha at "pure relevance". Setting this to a small positive value
    # (e.g. 0.1) adds lambda * realized_ILD(slate) onto the reward, giving
    # the policy a persistent incentive to keep some diversity weight. This
    # is a real deviation from the paper, not something stated in Eq. (9),
    # but it's the only thing that reliably prevented the collapse above.
    diversity_reward_weight: float = 0.05

    # --------------------------------------------------------- training loop
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

    def __post_init__(self):
        if self.dataset == "movielens_1m":
            # Separate cache/checkpoint dirs so a MovieLens run never
            # collides with an existing Amazon Beauty run. Only applied if
            # the fields are still at their dataclass defaults, so an
            # explicit data_dir=/checkpoint_dir= kwarg still wins.
            if self.data_dir == "data":
                self.data_dir = "data_movielens_1m"
            if self.checkpoint_dir == "checkpoints":
                self.checkpoint_dir = "checkpoints_movielens_1m"
            # ml-1m users average ~165 ratings each (vs. Amazon Beauty
            # 5-core's much shorter per-user histories), so the Amazon-tuned
            # h=20/max_seq_len=50 would truncate most of each user's real
            # history. Bumping these toward what the original SASRec paper
            # uses for ml-1m (it reports maxlen=200 for this dataset).
            # Same "only if still at the Amazon-tuned default" guard.
            if self.h == 20:
                self.h = 50
            if self.max_seq_len == 50:
                self.max_seq_len = 200
        elif self.dataset not in ("amazon_beauty", "movielens_1m"):
            raise ValueError(
                f"Unknown dataset '{self.dataset}'. Expected 'amazon_beauty' "
                f"or 'movielens_1m'."
            )