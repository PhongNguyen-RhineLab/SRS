"""
DiversityModule: implements the submodular template F^sub_theta (Eq. 2) and the
greedy slate selector (Section 4.1).

The objective is:
    F^sub_theta(S | u, alpha) = alpha * sum_{i in S} r_u(i)
                              - (1-alpha) * sum_{{i,j} in S} kappa_theta(i,j)

where kappa_theta is an RBF-style kernel on cosine distance:
    kappa(i,j) = exp( -(1 - cos(e_i, e_j)) / sigma )
    sigma      = exp(log_sigma)   (always positive)

Theorem 1 says F^sub_theta is submodular because the marginal gain of adding x to S:
    Delta(x|S) = alpha*r(x) - (1-alpha) * sum_{i in S} kappa(i,x)
decreases as S grows (more items already selected => larger penalty for x).

Greedy construction picks item x* = argmax Delta(x|S) at each step, using
epsilon-greedy + softmax exploration during training (controlled by eta_t).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiversityModule(nn.Module):

    def __init__(self, num_items: int, emb_dim: int = 64,
                log_sigma_min: float = -2.0, log_sigma_max: float = 2.0):
        super().__init__()
        self.num_items = num_items
        self.emb_dim   = emb_dim

        # Learned item embeddings for diversity (independent of retriever)
        self.item_emb  = nn.Embedding(num_items + 1, emb_dim, padding_idx=0)

        # Learnable bandwidth: sigma = exp(log_sigma)
        self.log_sigma = nn.Parameter(torch.zeros(1))
        # log_sigma is left UNCONSTRAINED as a parameter, but _sigma() clamps
        # it before exponentiating. Why: minimizing L_hit on positive-reward
        # transitions gives gradient descent a one-directional incentive to
        # shrink sigma toward 0 (since a smaller penalty term trivially
        # raises the slate score, lowering the loss), regardless of whether
        # the embeddings themselves are well-separated. I verified this
        # empirically over a real (synthetic-data) training run: sigma fell
        # monotonically from 1.0 to 0.56 in just 15 epochs with no sign of
        # leveling off. Left unchecked over 100 epochs, sigma can collapse
        # far enough that kappa(i,j) underflows toward 0 for nearly all
        # pairs, silently turning off the diversity penalty regardless of
        # alpha. Clamping keeps sigma in [exp(-2), exp(2)] ~= [0.135, 7.39],
        # a wide enough range to still learn a meaningful bandwidth.
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max

        nn.init.xavier_uniform_(self.item_emb.weight[1:])

    # ------------------------------------------------------------------ kernel

    def _sigma(self) -> torch.Tensor:
        clamped = torch.clamp(self.log_sigma, self.log_sigma_min, self.log_sigma_max)
        return torch.exp(clamped)

    def get_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        """L2-normalised embeddings; (N,) -> (N, D)."""
        return F.normalize(self.item_emb(item_ids), dim=-1)

    def kernel_matrix(self, embs: torch.Tensor) -> torch.Tensor:
        """
        Pairwise kappa matrix.
        embs : (N, D) L2-normalised
        returns: (N, N)
        """
        cos_sim = embs @ embs.T                          # (N, N)
        return torch.exp(-(1.0 - cos_sim) / self._sigma())

    # ------------------------------------------------------------ greedy selector

    def greedy_select(
        self,
        candidate_items: list,
        relevance_scores: list,
        alpha: float,
        eta: float,
        k: int,
        tau_min: float = 0.1,
        tau_max: float = 1.0,
        training: bool = True,
    ) -> list:
        """
        Greedy submodular selector (Section 4.1).

        candidate_items  : list of m item IDs (1-indexed)
        relevance_scores : list of m retrieval scores r_u(i)
        alpha            : relevance-diversity trade-off in [0,1]
        eta              : exploration knob in [0,1]
        k                : slate size
        training         : if False, uses deterministic greedy (no exploration)

        Returns: list of k item IDs in selection order (position 0 = highest priority)

        Example:
            slate = div.greedy_select([101,202,303,...], [0.9,0.8,0.7,...],
                                      alpha=0.7, eta=0.3, k=10)
        """
        N      = len(candidate_items)
        device = self.item_emb.weight.device

        items_t = torch.LongTensor(candidate_items).to(device)
        rel_t   = torch.FloatTensor(relevance_scores).to(device)

        with torch.no_grad():
            embs = self.get_embeddings(items_t)  # (N, D) normalised

            # Precompute full (N x N) kernel matrix once (one GPU matmul)
            sigma     = self._sigma().detach()
            cos_sim   = embs @ embs.T
            K         = torch.exp(-(1.0 - cos_sim) / sigma)  # (N, N)

        selected = torch.zeros(N, dtype=torch.bool, device=device)
        # pen_accum[j] = sum_{i in S} kappa(i, j), updated incrementally
        pen_accum = torch.zeros(N, device=device)
        slate     = []

        for _ in range(k):
            # marginal gain: Delta(x|S) = alpha*r(x) - (1-alpha)*pen_accum[x]
            gains = alpha * rel_t - (1.0 - alpha) * pen_accum
            gains[selected] = float("-inf")

            if training:
                eps = 0.5 * eta
                tau = tau_min + (tau_max - tau_min) * eta

                if torch.rand(1).item() < eps:
                    # softmax-temperature sampling
                    valid = gains.clone()
                    probs = F.softmax(valid / tau, dim=0)
                    chosen = torch.multinomial(probs, 1).item()
                else:
                    chosen = int(gains.argmax())
            else:
                chosen = int(gains.argmax())

            selected[chosen] = True
            slate.append(candidate_items[chosen])
            pen_accum = pen_accum + K[chosen]  # O(N) update

        return slate

    # --------------------------------------------------- average deficit (Def. 4)

    def estimate_average_deficit(
        self,
        candidate_items: list,
        relevance_scores: list,
        alpha: float,
        k: int,
        num_x_samples: int = 5,
    ) -> list:
        """
        Definition 4: average deficit along a greedy rollout.

            delta_bar(alpha) = E_{S~greedy path, x~Unif(V\\S)}[max(0, -Delta(x|S))]

        Unlike theory.worst_case_deficit (the pessimistic closed-form bound
        from Eq. 6), this walks the actual deterministic greedy path for
        this candidate pool and alpha, and at each intermediate state S_t
        (|S_t| = 0, 1, ..., k-1) samples a few random items x from the
        remaining pool, returning every max(0, -Delta(x|S_t)) sample
        encountered. Average these (across many users) to get delta_bar(alpha).

        Returns: list of float deficit samples (not yet averaged), so the
        caller can pool samples across many users before taking the mean.

        Example:
            samples = div.estimate_average_deficit(cand_ids, scores, alpha=0.617, k=10)
            # pool `samples` across all test users, then np.mean(all_samples)
        """
        N = len(candidate_items)
        device = self.item_emb.weight.device

        items_t = torch.LongTensor(candidate_items).to(device)
        rel_t = torch.FloatTensor(relevance_scores).to(device)

        with torch.no_grad():
            embs = self.get_embeddings(items_t)
            sigma = self._sigma().detach()
            cos_sim = embs @ embs.T
            K = torch.exp(-(1.0 - cos_sim) / sigma)

        selected = torch.zeros(N, dtype=torch.bool, device=device)
        pen_accum = torch.zeros(N, device=device)
        deficit_samples = []

        for _ in range(k):
            gains = alpha * rel_t - (1.0 - alpha) * pen_accum  # Delta(x|S_t) for all x

            avail_idx = (~selected).nonzero(as_tuple=True)[0]
            if len(avail_idx) > 0:
                n_samp = min(num_x_samples, len(avail_idx))
                perm = avail_idx[torch.randperm(len(avail_idx), device=device)[:n_samp]]
                for idx in perm:
                    deficit_samples.append(max(0.0, -float(gains[idx].item())))

            gains_masked = gains.clone()
            gains_masked[selected] = float("-inf")
            chosen = int(gains_masked.argmax())
            selected[chosen] = True
            pen_accum = pen_accum + K[chosen]

        return deficit_samples

    # ----------------------------------------------- differentiable slate score

    def compute_slate_score(
        self,
        slate_items: list,
        relevance_scores: list,
        alpha: float,
    ) -> torch.Tensor:
        """
        Compute F^sub_theta(S|u,alpha) as a differentiable scalar.

        Relevance scores are detached so gradients flow only through theta
        (item_emb, log_sigma), as stated in Section 4.3.

        slate_items     : list of selected item IDs
        relevance_scores: relevance values matching slate_items order
        alpha           : float
        """
        if not slate_items:
            return torch.tensor(0.0, device=self.item_emb.weight.device)

        device  = self.item_emb.weight.device
        items_t = torch.LongTensor(slate_items).to(device)
        embs    = self.get_embeddings(items_t)              # (|S|, D)

        rel = torch.FloatTensor(relevance_scores).to(device).detach()
        rel_term = alpha * rel.sum()

        if len(slate_items) < 2:
            return rel_term

        K    = self.kernel_matrix(embs)                     # (|S|, |S|)
        mask = torch.triu(torch.ones_like(K), diagonal=1)  # upper triangle
        pen  = (1.0 - alpha) * (K * mask).sum()

        return rel_term - pen

    # --------------------------------------------------- diversity module losses

    def hit_loss(
        self,
        score: torch.Tensor,
        reward: float,
        clamp: float = 1e-7,
    ) -> torch.Tensor:
        """
        L_hit = -R_t * log(max(sigma(F^sub), clamp))
        Applied only when reward > 0 (positive transitions).
        """
        if reward <= 0:
            return torch.tensor(0.0, device=self.item_emb.weight.device)
        q = torch.sigmoid(score)
        return -reward * torch.log(q.clamp(min=clamp))

    def diversity_rank_loss(
        self,
        slate_items: list,
        all_candidates: list,
        margin: float = 0.3,
    ) -> torch.Tensor:
        """
        L_div_rank: margin regulariser that:
          (a) pushes selected-pair similarity below (1 - margin)
          (b) pushes random-pair similarity below -margin

        L = E[max(0, sim(e_i,e_j) - (1-m)) + max(0, sim(e_i,e_k) + m)]

        where (i,j) are selected pairs and (i,k) are random candidate pairs.
        """
        device = self.item_emb.weight.device

        if len(slate_items) < 2:
            return torch.tensor(0.0, device=device)

        slate_t  = torch.LongTensor(slate_items).to(device)
        embs_s   = self.get_embeddings(slate_t)        # (|S|, D)

        # (a) selected pairs
        cos_s   = embs_s @ embs_s.T
        mask_u  = torch.triu(torch.ones_like(cos_s), diagonal=1)
        sim_sel = cos_s[mask_u.bool()]
        loss_a  = F.relu(sim_sel - (1.0 - margin))

        # (b) random pairs: sample negatives from candidates not in slate
        not_in = [c for c in all_candidates if c not in set(slate_items)]
        if not_in:
            n_neg = min(len(slate_items), len(not_in))
            neg_ids = np.random.choice(len(not_in), n_neg, replace=False)
            neg_t   = torch.LongTensor([not_in[i] for i in neg_ids]).to(device)
            embs_n  = self.get_embeddings(neg_t)       # (n_neg, D)
            # anchor: first selected item vs random negatives
            sim_neg = (embs_s[0:1] @ embs_n.T).squeeze(0)
            loss_b  = F.relu(sim_neg + margin)
        else:
            loss_b = torch.tensor(0.0, device=device)

        n_a = loss_a.numel()
        n_b = loss_b.numel() if loss_b.dim() > 0 else 1

        return (loss_a.sum() + loss_b.sum()) / max(n_a + n_b, 1)