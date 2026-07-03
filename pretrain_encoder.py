"""
Pretrain the RL StateEncoder with next-item prediction, then freeze (mac 6).

Why this exists
---------------
The paper states the state encoder is "frozen after pretraining", and the
hyperparameter table lists an encoder learning rate. Previously train_rl.py
built the StateEncoder with random weights, never trained it, and never put it
in an optimizer -- so states were features of a random-frozen GRU. This script
gives the encoder a real pretraining objective and saves state_encoder.pt,
which train_rl.py then loads and freezes.

Objective
---------
It must match exactly how the encoder is consumed at RL / eval time
(env._history_window + evaluate._get_history_window):

    window = left-padded last-h items ending right before position t
    target = seq[t]                     (the next item)

so we train the encoder to produce a state s_t from which the next item is
predictable, via full-softmax cross-entropy over the catalog. Only train_seqs
(= seq[:-2]) are used, so the val target (seq[-2]) and test target (seq[-1])
never enter encoder pretraining -- no leakage.

Example
-------
For a train sequence [A, B, C, D] with h=3 the samples are
    window [0,0,A]   -> B
    window [0,A,B]   -> C
    window [A,B,C]   -> D
(each window left-padded to width h; most recent item last).

Run
---
    python pretrain_encoder.py            # after setting DATASET below
produces  <checkpoint_dir>/state_encoder.pt
"""

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from config import Config
from cli import build_config
from data import load_and_preprocess, split_data
from rl_policy import StateEncoder


class NextItemWindowDataset(Dataset):
    """(h-window, next-item) samples built exactly like env._history_window."""

    def __init__(self, train_seqs: dict, h: int):
        self.h = h
        self.samples = []
        for _u, seq in train_seqs.items():
            for t in range(1, len(seq)):
                hist = seq[max(0, t - h): t]
                target = seq[t]
                self.samples.append((hist, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hist, target = self.samples[idx]
        pad = [0] * (self.h - len(hist))
        return torch.LongTensor(pad + hist), target


class EncoderPretrainHead(nn.Module):
    """StateEncoder + a linear next-item head (the head is discarded after)."""

    def __init__(self, num_items: int, item_emb_dim: int, state_dim: int, h: int):
        super().__init__()
        self.encoder = StateEncoder(
            num_items, item_emb_dim=item_emb_dim, state_dim=state_dim, h=h)
        self.head = nn.Linear(state_dim, num_items + 1)  # class 0 = padding, never a target

    def forward(self, hist_items):
        return self.head(self.encoder(hist_items))  # (B, num_items+1)


def main():
    cfg = build_config("Pretrain the state encoder with next-item prediction.")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    lr = getattr(cfg, "lr_encoder", 1e-3)               # paper: "LR for Sub + Encoder 1e-3"
    epochs = int(os.environ.get("ENC_EPOCHS", getattr(cfg, "encoder_pretrain_epochs", 30)))
    batch_size = getattr(cfg, "ret_batch_size", 256)
    device = cfg.device

    print(f"[{cfg.dataset}] loading data...")
    data = load_and_preprocess(cfg.data_dir, cfg.dataset)
    cfg.num_items = data["num_items"]
    train_seqs, _val, _test = split_data(data["sequences"])

    ds = NextItemWindowDataset(train_seqs, cfg.h)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"[{cfg.dataset}] {len(ds)} next-item windows | h={cfg.h} "
          f"| num_items={cfg.num_items} | epochs={epochs} lr={lr}")

    model = EncoderPretrainHead(
        cfg.num_items, item_emb_dim=cfg.ret_emb_dim,
        state_dim=cfg.state_dim, h=cfg.h).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for hist, target in loader:
            hist, target = hist.to(device), target.to(device)
            opt.zero_grad()
            loss = ce(model(hist), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            running += loss.item() * hist.size(0)
            seen += hist.size(0)
        print(f"  epoch {epoch:3d}/{epochs}  ce_loss={running / seen:.4f}")

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    out = os.path.join(cfg.checkpoint_dir, "state_encoder.pt")
    # save ONLY the encoder (the head is throwaway); keys match a plain
    # StateEncoder(...) so train_rl / run_test_eval load it unchanged.
    torch.save(model.encoder.state_dict(), out)
    print(f"Saved pretrained StateEncoder to {out}")


if __name__ == "__main__":
    main()