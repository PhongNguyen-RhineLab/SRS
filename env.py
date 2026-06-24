"""
RL environment for slate-optimization training (Stage 3).

At each step t:
  1. Build the user's history window (last h items) -> state s_t via StateEncoder
  2. Use the frozen retriever + FAISS index to fetch top-m candidates
  3. Actor outputs (alpha_t, eta_t)
  4. Greedy diversity-aware selector builds slate S_t of size k
  5. Reward R_t computed from whether the next true item is in S_t (Eq. 9)
  6. Move to next position in the user's sequence -> s_{t+1}

Per Section 4.2, the state encoder is frozen (states are precomputed under
no_grad and cached), so this environment precomputes encoded states once
per epoch for efficiency, rather than recomputing every gradient step.

Example:
    env = SlateEnv(cfg, train_seqs, retriever, faiss_index, diversity_module)
    transitions = env.run_episode_batch(user_ids, actor, training=True)
"""

import numpy as np
import torch


class SlateEnv:

    def __init__(self, cfg, train_seqs: dict, retriever, faiss_index,
                diversity_module, state_encoder):
        self.cfg = cfg
        self.train_seqs = train_seqs
        self.retriever = retriever
        self.faiss_index = faiss_index
        self.diversity_module = diversity_module
        self.state_encoder = state_encoder
        self.device = cfg.device

        # build list of (user, position) steps that have a valid target
        # position t predicts seq[t] given seq[:t]
        self.steps = []
        for u, seq in train_seqs.items():
            for t in range(1, len(seq)):
                self.steps.append((u, t))

    def _history_window(self, seq: list, t: int) -> torch.Tensor:
        """Left-padded last-h items ending right before position t."""
        h = self.cfg.h
        hist = seq[max(0, t - h): t]
        pad = [0] * (h - len(hist))
        return torch.LongTensor(pad + hist)

    @torch.no_grad()
    def _encode_state(self, hist_batch: torch.Tensor) -> torch.Tensor:
        """Frozen state encoding (Section 4.2: encoder is frozen, no_grad)."""
        self.state_encoder.eval()
        return self.state_encoder(hist_batch.to(self.device))

    @torch.no_grad()
    def _retrieve_candidates(self, seq: list, t: int):
        """
        Use the frozen retriever to compute the user query embedding and
        search the FAISS index for top-m candidates.

        Returns: (candidate_item_ids, relevance_scores) both length <= m
        """
        max_len = self.cfg.max_seq_len
        hist = seq[max(0, t - max_len): t]
        pad = [0] * (max_len - len(hist))
        seq_t = torch.LongTensor([pad + hist]).to(self.device)

        self.retriever.eval()
        query = self.retriever.get_user_embedding(seq_t).cpu().numpy()[0]

        cand_ids, scores = self.faiss_index.search(query, self.cfg.m)
        return cand_ids.tolist(), scores.tolist()

    def compute_reward(self, slate: list, target: int) -> float:
        """
        Shaped hit reward (Eq. 9):
            R_t = w_t * (1 + rho * (k-1-rank)/k)   if target in slate
                = -p_miss                           otherwise
        w_t = 1 (no per-target rating available, as in reported runs).

        If cfg.diversity_reward_weight > 0 (off by default -- see config.py
        for why), adds lambda * realized_ILD(slate) on top. This is a
        deviation from the paper's literal Eq. 9, added because nothing in
        the hit/rank-only reward stops alpha from drifting to 1.0 (confirmed
        in a real training run -- see config.py's comment for details).
        """
        cfg = self.cfg
        if target in slate:
            rank = slate.index(target)  # 0-indexed position
            reward = 1.0 * (1.0 + cfg.rho * (cfg.k - 1 - rank) / cfg.k)
        else:
            reward = -cfg.p_miss

        if cfg.diversity_reward_weight > 0:
            reward += cfg.diversity_reward_weight * self._realized_ild(slate)

        return reward

    @torch.no_grad()
    def _realized_ild(self, slate: list) -> float:
        """
        Mean pairwise cosine distance within the slate, using the diversity
        module's CURRENT embeddings (not detached/frozen, since this is only
        used as a scalar reward signal, never backpropagated through).
        """
        if len(slate) < 2:
            return 0.0
        items_t = torch.LongTensor(slate).to(self.device)
        embs = self.diversity_module.get_embeddings(items_t)
        sims = embs @ embs.T
        n = len(slate)
        total, count = 0.0, 0
        for i in range(n):
            for j in range(i + 1, n):
                total += (1.0 - sims[i, j].item())
                count += 1
        return total / count if count else 0.0

    def step(self, user: int, t: int, actor, training: bool = True):
        """
        Execute one environment step for a single (user, position).

        Returns a dict with: state, next_state, a_tilde, action, reward,
                             slate, rel_scores, candidates, done
        """
        seq = self.train_seqs[user]
        target = seq[t]

        hist_window = self._history_window(seq, t).unsqueeze(0)
        state = self._encode_state(hist_window)  # (1, state_dim)

        cand_ids, rel_scores = self._retrieve_candidates(seq, t)
        if len(cand_ids) == 0:
            return None  # no candidates retrieved, skip this step

        if training:
            action, log_prob, a_tilde = actor.sample(state, deterministic=False)
        else:
            action, log_prob, a_tilde = actor.sample(state, deterministic=True)

        alpha = float(action[0, 0].item())
        eta   = float(action[0, 1].item())

        slate = self.diversity_module.greedy_select(
            cand_ids, rel_scores, alpha, eta, self.cfg.k,
            tau_min=self.cfg.tau_min, tau_max=self.cfg.tau_max,
            training=training,
        )

        reward = self.compute_reward(slate, target)

        done = (t == len(seq) - 1)
        next_t = min(t + 1, len(seq) - 1)
        next_hist = self._history_window(seq, next_t).unsqueeze(0)
        next_state = self._encode_state(next_hist)

        # relevance scores aligned with the slate, for diversity-module loss
        score_map = dict(zip(cand_ids, rel_scores))
        slate_rel = [score_map[i] for i in slate]

        return {
            "state": state.squeeze(0).detach(),
            "next_state": next_state.squeeze(0).detach(),
            "a_tilde": a_tilde.squeeze(0).detach(),
            "alpha": alpha,
            "eta": eta,
            "reward": reward,
            "slate": slate,
            "slate_rel": slate_rel,
            "candidates": cand_ids,
            "done": done,
        }

    def sample_batch_steps(self, batch_size: int):
        """Randomly sample (user, t) pairs for one training batch of episodes."""
        idxs = np.random.randint(0, len(self.steps), size=batch_size)
        return [self.steps[i] for i in idxs]