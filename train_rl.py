"""
Joint training loop for the RL policy + diversity module (Section 4.3).

Two independent optimizers update at each step:
  1. RL optimizer   minimizes L_RL = L_critic + L_actor      (Eq. 12)
  2. Sub optimizer  minimizes L_sub = L_hit + lambda_rank * L_div_rank

L_actor = -sg[Abar_t] * log pi(a_tilde_t|s_t)
          + beta_bc * (eta_cur_t - sg[eta_beh_t])^2      (BC term, eta only)
          - beta_ent * H[pi(.|s_t)]                       (entropy bonus)

Example:
    python train_rl.py
"""

import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import Config

# Set to "movielens_1m" to run against MovieLens-1M instead.
# DATASET = "amazon_beauty"
DATASET = "movielens_1m"
from data import load_and_preprocess, split_data
from retriever import SASRec, FAISSIndex
from diversity import DiversityModule
from rl_policy import StateEncoder, Actor, Critic
from buffer import ReplayBuffer
from env import SlateEnv
from evaluate import evaluate_srs, evaluate_icsrec_greedy, get_item_embeddings_for_ild


def compute_normalized_advantage(deltas: torch.Tensor, eps: float = 1e-3,
                                clip: float = 10.0) -> torch.Tensor:
    """
    A_bar_t = (delta_t - mu_delta) / (sigma_delta + eps), Eq. (11).

    With the paper's literal eps (implicitly ~0) and no clip, a mini-batch
    whose TD residuals happen to cluster tightly (very plausible here, since
    ~75-80% of transitions share the same fixed miss-penalty reward -p_miss)
    makes sigma_delta collapse toward 0, and any single sample that differs
    even slightly then gets an unbounded advantage. I confirmed this
    empirically: a batch of 31 identical deltas plus one differing by 1e-5
    produces sigma_delta ~ 1.7e-6 and an advantage of several units from
    that tiny perturbation alone; with a real hit-vs-miss reward gap mixed
    into an otherwise-tight cluster, this routinely produces advantages in
    the hundreds to thousands, which is the direct cause of the actor_loss
    exploding by 9 orders of magnitude over 100 epochs in real training
    (see the diagnosis in chat: -309 at epoch 1 to -1.4e8 by epoch 30).

    Raising eps to a non-negligible floor and hard-clipping the result are
    both standard, defensible stabilizers (the same idea as PPO's advantage
    clipping) and don't change what the advantage is "trying to do" --
    just bounds how much a single degenerate batch can dominate one
    gradient step.
    """
    mu = deltas.mean()
    sigma = deltas.std(unbiased=False)
    adv = (deltas - mu) / (sigma + eps)
    return torch.clamp(adv, -clip, clip)


def train_one_epoch(cfg, env: SlateEnv, actor: Actor, critic: Critic,
                    diversity_module: DiversityModule,
                    rl_optimizer, sub_optimizer, buffer: ReplayBuffer):
    device = cfg.device
    actor.train()
    critic.train()
    diversity_module.train()

    epoch_logs = {"critic_loss": [], "actor_loss": [], "sub_loss": [],
                 "reward": [], "alpha": []}

    for step in range(cfg.steps_per_epoch):
        # ---- collect one transition (single-environment rollout) ----
        steps_batch = env.sample_batch_steps(1)
        user, t = steps_batch[0]

        transition = env.step(user, t, actor, training=True)
        if transition is None:
            continue

        buffer.push(
            transition["state"], transition["next_state"], transition["a_tilde"],
            transition["reward"], transition["slate"], transition["slate_rel"],
            transition["candidates"], transition["done"],
        )
        epoch_logs["reward"].append(transition["reward"])
        epoch_logs["alpha"].append(transition["alpha"])

        if len(buffer) < cfg.batch_size:
            continue

        # ---- sample a mini-batch and update ----
        batch = buffer.sample(cfg.batch_size)

        states      = torch.stack([b.state for b in batch]).to(device)
        next_states = torch.stack([b.next_state for b in batch]).to(device)
        a_tilde_beh = torch.stack([b.a_tilde for b in batch]).to(device)
        rewards = torch.FloatTensor([b.reward for b in batch]).to(device)
        dones = torch.FloatTensor([float(b.done) for b in batch]).to(device)

        # ---------------- critic loss (Eq. 10) ----------------
        with torch.no_grad():
            v_next = critic(next_states)
            td_target = rewards + cfg.gamma * (1.0 - dones) * v_next

        v_curr = critic(states)
        critic_loss = F.mse_loss(v_curr, td_target.detach())

        with torch.no_grad():
            delta = rewards + cfg.gamma * (1.0 - dones) * v_next - v_curr
            advantage = compute_normalized_advantage(
                delta, eps=cfg.advantage_eps, clip=cfg.advantage_clip)

        # ---------------- actor loss (Eq. 13) ----------------
        # Floor the re-evaluated behavior log-prob: transitions so stale that
        # the current policy assigns them log pi < logp_clip_min contribute a
        # constant (zero gradient) instead of an unbounded -A*logp pull. See
        # config.logp_clip_min for the failure mode this prevents.
        logp_cur = actor.log_prob_of(states, a_tilde_beh).clamp(
            min=cfg.logp_clip_min)

        action_cur, _, _ = actor.sample(states, deterministic=False)
        eta_cur = action_cur[:, 1]
        eta_beh = torch.sigmoid(a_tilde_beh[:, 1]).detach()

        bc_term = cfg.beta_bc * ((eta_cur - eta_beh) ** 2).mean()
        entropy_term = cfg.beta_ent * actor.entropy(states).mean()

        actor_loss = (-(advantage.detach() * logp_cur).mean()
                     + bc_term - entropy_term)

        rl_loss = critic_loss + actor_loss

        if len(epoch_logs["actor_loss"]) == 0:  # log once per epoch, first minibatch
            with torch.no_grad():
                print(f"    [dbg] pg={-(advantage.detach() * logp_cur).mean().item():.2f} "
                      f"bc={bc_term.item():.2f} ent_term={entropy_term.item():.2f} "
                      f"| logp_cur[min/mean/max]={logp_cur.min().item():.1f}/"
                      f"{logp_cur.mean().item():.1f}/{logp_cur.max().item():.1f} "
                      f"| adv[min/max]={advantage.min().item():.2f}/{advantage.max().item():.2f} "
                      f"| logstd[min/max]={actor.forward(states)[1].min().item():.2f}/"
                      f"{actor.forward(states)[1].max().item():.2f}")

        rl_optimizer.zero_grad()
        rl_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(actor.parameters()) + list(critic.parameters()), max_norm=5.0)
        rl_optimizer.step()

        # ---------------- diversity-module loss ----------------
        sub_losses = []
        for b in batch:
            alpha_b = float(torch.sigmoid(b.a_tilde[0]).item())
            score = diversity_module.compute_slate_score(
                b.slate_items, b.rel_scores, alpha_b)
            hit_l = diversity_module.hit_loss(score, b.reward, clamp=cfg.div_clamp)
            rank_l = diversity_module.diversity_rank_loss(
                b.slate_items, b.candidate_items, margin=cfg.div_margin)
            sub_losses.append(hit_l + cfg.lambda_rank * rank_l)

        sub_loss = torch.stack(sub_losses).mean()

        sub_optimizer.zero_grad()
        sub_loss.backward()
        torch.nn.utils.clip_grad_norm_(diversity_module.parameters(), max_norm=5.0)
        sub_optimizer.step()

        epoch_logs["critic_loss"].append(critic_loss.item())
        epoch_logs["actor_loss"].append(actor_loss.item())
        epoch_logs["sub_loss"].append(sub_loss.item())

    return {k: float(np.mean(v)) if v else 0.0 for k, v in epoch_logs.items()}


def main():
    cfg = Config(dataset=DATASET)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("Loading preprocessed data...")
    data = load_and_preprocess(cfg.data_dir, cfg.dataset)
    cfg.num_items = data["num_items"]
    cfg.num_users = data["num_users"]
    train_seqs, val_seqs, test_seqs = split_data(data["sequences"])

    device = cfg.device
    print(f"Using device: {device}")

    # ---------------- load frozen retriever + FAISS index ----------------
    from icsrec_retriever import load_icsrec_retriever
    retriever = load_icsrec_retriever(
        cfg.icsrec_ckpt, cfg.num_items, hidden_size=cfg.ret_emb_dim, device=device)
    for p in retriever.parameters():
        p.requires_grad_(False)

    faiss_index = FAISSIndex.load(
        os.path.join(cfg.checkpoint_dir, "faiss_index.npy"), cfg.ret_emb_dim)

    # ---------------- build trainable modules ----------------
    diversity_module = DiversityModule(cfg.num_items, cfg.div_emb_dim).to(device)
    state_encoder = StateEncoder(
        cfg.num_items, item_emb_dim=cfg.ret_emb_dim,
        state_dim=cfg.state_dim, h=cfg.h,
    ).to(device)
    # muc 6: load pretrained encoder and freeze (paper: "frozen after pretraining")
    enc_path = os.path.join(cfg.checkpoint_dir, "state_encoder.pt")
    if not os.path.exists(enc_path):
        raise FileNotFoundError(f"{enc_path} not found. Run pretrain_encoder.py first.")
    state_encoder.load_state_dict(torch.load(enc_path, map_location=device))
    state_encoder.eval()
    for p in state_encoder.parameters():
        p.requires_grad_(False)
    actor = Actor(cfg.state_dim, cfg.hidden_dim, cfg.alpha_init_bias).to(device)
    critic = Critic(cfg.state_dim, cfg.hidden_dim).to(device)

    rl_optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=cfg.lr_rl)
    sub_optimizer = torch.optim.Adam(diversity_module.parameters(), lr=cfg.lr_sub)

    buffer = ReplayBuffer(cfg.buffer_size)
    env = SlateEnv(cfg, train_seqs, retriever, faiss_index, diversity_module, state_encoder)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    log_path = os.path.join(cfg.checkpoint_dir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "critic_loss", "actor_loss", "sub_loss",
                         "mean_reward", "mean_alpha", "val_HR@10", "val_NDCG@10"])

    print(f"Starting joint RL training for {cfg.num_epochs} epochs, "
          f"{cfg.steps_per_epoch} steps/epoch...")

    for epoch in range(1, cfg.num_epochs + 1):
        logs = train_one_epoch(cfg, env, actor, critic, diversity_module,
                              rl_optimizer, sub_optimizer, buffer)

        print(f"Epoch {epoch}: critic_loss={logs['critic_loss']:.4f} "
              f"actor_loss={logs['actor_loss']:.4f} sub_loss={logs['sub_loss']:.4f} "
              f"mean_reward={logs['reward']:.4f} mean_alpha={logs['alpha']:.3f}")

        val_hr, val_ndcg = 0.0, 0.0
        if epoch % cfg.eval_every == 0 or epoch == 1 or epoch == cfg.num_epochs:
            item_embs = get_item_embeddings_for_ild(retriever, cfg.num_items)
            val_metrics = evaluate_srs(
                cfg, val_seqs, retriever, faiss_index, diversity_module,
                state_encoder, actor, item_embs,
            )
            val_hr, val_ndcg = val_metrics["HR@k"], val_metrics["NDCG@k"]
            print(f"  Val HR@{cfg.k}={val_hr:.4f} NDCG@{cfg.k}={val_ndcg:.4f} "
                  f"ILD={val_metrics['ILD']:.4f}")

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, logs["critic_loss"], logs["actor_loss"],
                            logs["sub_loss"], logs["reward"], logs["alpha"],
                            val_hr, val_ndcg])

    # ---------------- save checkpoints ----------------
    torch.save(diversity_module.state_dict(),
               os.path.join(cfg.checkpoint_dir, "diversity_module.pt"))
    torch.save(actor.state_dict(), os.path.join(cfg.checkpoint_dir, "actor.pt"))
    torch.save(critic.state_dict(), os.path.join(cfg.checkpoint_dir, "critic.pt"))
    print("All Stage-3 checkpoints saved.")


if __name__ == "__main__":
    main()