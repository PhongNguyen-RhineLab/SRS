"""
StateEncoder + Actor-Critic policy (Section 4.2).

StateEncoder:
    x_t = GRU([e_{i_tau}; f_tau] for tau in last h items)   -> last hidden state
    h_t = mean of last h item embeddings                    -> bag-of-items summary
    s_t = MLP([x_t ; h_t])

Actor (squashed Gaussian, SAC-style):
    (mu_t, log_sigma_t) = pi_phi(s_t)
    a_tilde_t ~ N(mu_t, sigma_t^2 I)
    a_t = (alpha_t, eta_t) = sigmoid(a_tilde_t)

Critic:
    V_psi(s_t)  - simple value MLP

Example:
    encoder = StateEncoder(num_items=12101, item_emb_dim=64, state_dim=128, h=20)
    s_t = encoder(history_item_ids, history_ratings)   # (B, state_dim)
    actor = Actor(state_dim=128, hidden_dim=256)
    action, logp = actor.sample(s_t)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class StateEncoder(nn.Module):

    def __init__(self, num_items: int, item_emb_dim: int = 64,
                 state_dim: int = 128, h: int = 20,
                 use_rating_feature: bool = False):
        super().__init__()
        self.h = h
        self.item_emb_dim = item_emb_dim
        self.use_rating_feature = use_rating_feature

        # GRU input is [item_emb ; optional scalar rating]
        gru_input_dim = item_emb_dim + (1 if use_rating_feature else 0)

        self.item_emb = nn.Embedding(num_items + 1, item_emb_dim, padding_idx=0)
        self.gru      = nn.GRU(gru_input_dim, item_emb_dim, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(item_emb_dim * 2, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim),
        )

        nn.init.xavier_uniform_(self.item_emb.weight[1:])

    def forward(self, hist_items: torch.Tensor,
               hist_ratings: torch.Tensor = None) -> torch.Tensor:
        """
        hist_items   : (B, h) LongTensor, 0 = padding, most recent item last
        hist_ratings : (B, h) FloatTensor, optional

        returns: (B, state_dim)
        """
        embs = self.item_emb(hist_items)        # (B, h, D)

        if self.use_rating_feature and hist_ratings is not None:
            gru_in = torch.cat([embs, hist_ratings.unsqueeze(-1)], dim=-1)
        else:
            gru_in = embs

        # mask padding for mean-pool
        mask = (hist_items != 0).float().unsqueeze(-1)         # (B, h, 1)
        h_t  = (embs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, D)

        _, x_t = self.gru(gru_in)               # x_t: (1, B, D)
        x_t = x_t.squeeze(0)                     # (B, D)

        return self.mlp(torch.cat([x_t, h_t], dim=-1))  # (B, state_dim)


class Actor(nn.Module):
    """Squashed Gaussian policy outputting (alpha_t, eta_t)."""

    def __init__(self, state_dim: int = 128, hidden_dim: int = 256,
                 alpha_init_bias: float = 2.197):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head      = nn.Linear(hidden_dim, 2)   # [alpha, eta]
        self.log_std_head = nn.Linear(hidden_dim, 2)

        # Bias the alpha-dimension toward 0.9 at init (Section 4.3)
        with torch.no_grad():
            self.mu_head.bias[0] = alpha_init_bias  # alpha dim
            self.mu_head.bias[1] = 0.0              # eta dim

    def forward(self, state: torch.Tensor):
        h = self.net(state)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, state: torch.Tensor, deterministic: bool = False):
        """
        Returns:
            action       : (B, 2) in [0,1]^2  -> (alpha_t, eta_t)
            log_prob     : (B,) log pi(a_tilde|s)  [pre-squash, Eq. (7)]
            latent_action: (B, 2) pre-squash a_tilde (needed for replay buffer)
        """
        mu, log_std = self.forward(state)
        std = log_std.exp()

        if deterministic:
            a_tilde = mu
        else:
            dist = Normal(mu, std)
            a_tilde = dist.rsample()

        action = torch.sigmoid(a_tilde)

        dist = Normal(mu, std)
        log_prob = dist.log_prob(a_tilde).sum(dim=-1)  # Eq. (7): sum over dims

        return action, log_prob, a_tilde

    def log_prob_of(self, state: torch.Tensor,
                    a_tilde: torch.Tensor) -> torch.Tensor:
        """Re-evaluate log pi(a_tilde|s) for a stored latent action (used in actor loss)."""
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        return dist.log_prob(a_tilde).sum(dim=-1)

    def entropy(self, state: torch.Tensor) -> torch.Tensor:
        """Differential entropy of the pre-squash Gaussian (per-batch mean)."""
        _, log_std = self.forward(state)
        # entropy of N(mu, sigma^2) per-dim = 0.5*log(2*pi*e*sigma^2)
        ent = 0.5 * (1.0 + torch.log(2 * torch.pi * torch.exp(2 * log_std)))
        return ent.sum(dim=-1)


class Critic(nn.Module):
    """Value function V_psi(s_t)."""

    def __init__(self, state_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)  # (B,)
