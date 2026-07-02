"""
Quick smoke test using synthetic data (real Amazon Beauty download is not
reachable from this sandbox's network allowlist). Exercises every module
end-to-end with tiny sizes to catch shape/logic bugs.
"""

import numpy as np
import torch

from config import Config
from data import split_data, get_retriever_loader
from retriever import SASRec, FAISSIndex
from diversity import DiversityModule
from rl_policy import StateEncoder, Actor, Critic
from buffer import ReplayBuffer
from env import SlateEnv
from train_rl import train_one_epoch
from evaluate import evaluate_srs, evaluate_icsrec_greedy, get_item_embeddings_for_ild


def make_synthetic_sequences(num_users=50, num_items=200, min_len=5, max_len=15, seed=0):
    rng = np.random.RandomState(seed)
    seqs = {}
    for u in range(1, num_users + 1):
        L = rng.randint(min_len, max_len + 1)
        seqs[u] = list(rng.randint(1, num_items + 1, size=L))
    return seqs


def main():
    cfg = Config()
    cfg.num_items = 200
    cfg.num_users = 50
    cfg.max_seq_len = 10
    cfg.h = 5
    cfg.m = 30
    cfg.k = 5
    cfg.state_dim = 32
    cfg.hidden_dim = 32
    cfg.ret_emb_dim = 16
    cfg.div_emb_dim = 16
    cfg.ret_epochs = 1
    cfg.ret_batch_size = 16
    cfg.buffer_size = 200
    cfg.batch_size = 8
    cfg.steps_per_epoch = 20
    cfg.num_epochs = 2
    cfg.device = "cpu"

    torch.manual_seed(0)
    np.random.seed(0)

    print("[1/7] Building synthetic sequences...")
    sequences = make_synthetic_sequences(cfg.num_users, cfg.num_items)
    train_seqs, val_seqs, test_seqs = split_data(sequences)
    print(f"  train={len(train_seqs)} val={len(val_seqs)} test={len(test_seqs)}")

    print("[2/7] Training tiny SASRec retriever (1 epoch)...")
    retriever = SASRec(cfg.num_items, cfg.ret_emb_dim, cfg.max_seq_len,
                       cfg.ret_num_heads, cfg.ret_num_layers, cfg.ret_dropout)
    loader = get_retriever_loader(train_seqs, cfg.max_seq_len, cfg.ret_batch_size)
    opt = torch.optim.Adam(retriever.parameters(), lr=cfg.ret_lr)
    for seq, target in loader:
        target = target.squeeze(-1)
        opt.zero_grad()
        loss = retriever.full_softmax_loss(seq, target)
        loss.backward()
        opt.step()
    print(f"  final batch loss={loss.item():.4f}")
    retriever.eval()
    for p in retriever.parameters():
        p.requires_grad_(False)

    print("[3/7] Building FAISS index...")
    with torch.no_grad():
        embs = retriever.get_all_item_embeddings().numpy()
    index = FAISSIndex(cfg.ret_emb_dim)
    index.build(embs, list(range(1, cfg.num_items + 1)))
    q = embs[0]
    cand_ids, scores = index.search(q, cfg.m)
    assert len(cand_ids) == cfg.m, f"expected {cfg.m} candidates, got {len(cand_ids)}"
    print(f"  FAISS search OK, top candidate={cand_ids[0]} score={scores[0]:.4f}")

    print("[4/7] Testing DiversityModule greedy_select + losses...")
    div = DiversityModule(cfg.num_items, cfg.div_emb_dim)
    slate = div.greedy_select(cand_ids.tolist(), scores.tolist(), alpha=0.7, eta=0.3, k=cfg.k)
    assert len(slate) == cfg.k
    assert len(set(slate)) == cfg.k, "duplicate items in slate!"
    score = div.compute_slate_score(slate, [1.0] * cfg.k, alpha=0.7)
    hit_l = div.hit_loss(score, reward=1.0)
    rank_l = div.diversity_rank_loss(slate, cand_ids.tolist())
    (hit_l + rank_l).backward()
    print(f"  slate={slate}")
    print(f"  score={score.item():.4f} hit_loss={hit_l.item():.4f} rank_loss={rank_l.item():.4f}")

    print("[5/7] Testing StateEncoder + Actor + Critic shapes...")
    div = DiversityModule(cfg.num_items, cfg.div_emb_dim)  # fresh, no leftover grads
    state_enc = StateEncoder(cfg.num_items, cfg.ret_emb_dim, cfg.state_dim, cfg.h)
    actor = Actor(cfg.state_dim, cfg.hidden_dim, cfg.alpha_init_bias)
    critic = Critic(cfg.state_dim, cfg.hidden_dim)

    hist = torch.LongTensor([[1, 2, 3, 0, 0]])
    s = state_enc(hist)
    assert s.shape == (1, cfg.state_dim)
    action, logp, a_tilde = actor.sample(s)
    assert action.shape == (1, 2)
    v = critic(s)
    assert v.shape == (1,)
    alpha0 = float(torch.sigmoid(actor.mu_head.bias[0]).item())
    print(f"  state shape OK, action={action.tolist()}, alpha_init~{alpha0:.3f} (expect ~0.9)")

    print("[6/7] Running 2 tiny epochs of joint RL training...")
    env = SlateEnv(cfg, train_seqs, retriever, index, div, state_enc)
    rl_opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=cfg.lr_rl)
    sub_opt = torch.optim.Adam(div.parameters(), lr=cfg.lr_sub)
    buf = ReplayBuffer(cfg.buffer_size)

    for ep in range(1, cfg.num_epochs + 1):
        logs = train_one_epoch(cfg, env, actor, critic, div, rl_opt, sub_opt, buf)
        print(f"  epoch {ep}: {logs}")

    print("[7/7] Running evaluation (val split)...")
    item_embs = get_item_embeddings_for_ild(retriever, cfg.num_items)
    baseline_metrics = evaluate_icsrec_greedy(cfg, val_seqs, retriever, index, item_embs)
    srs_metrics = evaluate_srs(cfg, val_seqs, retriever, index, div, state_enc, actor, item_embs)
    print(f"  baseline: {baseline_metrics}")
    print(f"  srs:      {srs_metrics}")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
