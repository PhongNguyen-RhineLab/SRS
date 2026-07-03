"""
Data loading and preprocessing.

Sequences are loaded from the official ICSRec .txt files (one file per
dataset, declared in config.DATASET_REGISTRY). Each line is
"user_id item1 item2 ... itemN" with items already 1-indexed in the SAME id
space as the released ICSRec checkpoints, so ids are used verbatim -- no
remapping. This is deliberate: re-preprocessing raw Amazon/MovieLens dumps
would produce a different user2idx/item2idx mapping and silently misalign
sequences with the checkpoint's item-embedding table. Run fetch_data.py to
download the files.

All datasets share the same LL2O split, RetrieverDataset, and DataLoader
logic below.

The paper uses Leave-Last-2-Out (LL2O):
  train   : [i_0, ..., i_{n-3}]          (sliding window for retriever training)
  val     : history=[i_0,..,i_{n-3}]  target=i_{n-2}
  test    : history=[i_0,..,i_{n-2}]  target=i_{n-1}

Example sequence [A, B, C, D, E]:
  train_seq = [A, B, C]
  val  = (history=[A,B,C], target=D)
  test = (history=[A,B,C,D], target=E)
"""

import os

import torch
from torch.utils.data import Dataset, DataLoader

from config import DATASET_REGISTRY


# --------------------------------------------------------------------------- loader

def load_and_preprocess(data_dir: str, dataset: str = "amazon_beauty") -> dict:
    """
    Load sequences from the official ICSRec data file for `dataset`.

    Returns a dict with keys:
        sequences : {user_id: [item_id, ...]}  (1-indexed, chronological)
        num_items : int  (= max item id observed)
        num_users : int
    """
    if dataset not in DATASET_REGISTRY:
        known = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset '{dataset}'. Expected one of: {known}")

    fname = DATASET_REGISTRY[dataset]["txt"]
    path = os.path.join(data_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Fetch it with:\n"
            f"  python fetch_data.py --dataset {dataset}\n"
            f"or copy the official ICSRec data file there manually:\n"
            f"  cp <ICSRec>/data/{fname} {path}")

    sequences, max_item = {}, 0
    with open(path) as f:
        for line in f:
            toks = line.split()
            if not toks:
                continue
            u = int(toks[0])
            items = list(map(int, toks[1:]))
            if len(items) >= 3:                # need >=3 for LL2O
                sequences[u] = items
                max_item = max(max_item, max(items))
    print(f"[{dataset}] loaded {len(sequences)} sequences, max_item={max_item}")
    return {"sequences": sequences, "num_items": max_item,
            "num_users": len(sequences)}


# --------------------------------------------------------------------------- splits

def split_data(sequences: dict):
    """
    Leave-Last-2-Out split as described in Section 5.1.

    Returns:
        train_seqs : {user: list}           sequences for RL environment
        val_seqs   : {user: (list, int)}    (history, target) for validation
        test_seqs  : {user: (list, int)}    (history, target) for test
    """
    train_seqs = {}
    val_seqs = {}
    test_seqs = {}

    for u, seq in sequences.items():
        train_seqs[u] = seq[:-2]            # [i_0 .. i_{n-3}]
        val_seqs[u]   = (seq[:-2], seq[-2]) # history, target=i_{n-2}
        test_seqs[u]  = (seq[:-1], seq[-1]) # history, target=i_{n-1}

    return train_seqs, val_seqs, test_seqs


# --------------------------------------------------------------------------- retriever dataloader

class RetrieverDataset(Dataset):
    """
    Sliding-window next-item prediction dataset for SASRec training.

    For a sequence [A,B,C,D] we create samples:
        ([A],          B)
        ([A,B],        C)
        ([A,B,C],      D)
    Each input is left-padded with zeros to max_seq_len.
    """

    def __init__(self, train_seqs: dict, max_seq_len: int):
        self.max_seq_len = max_seq_len
        self.samples = []

        for u, seq in train_seqs.items():
            for t in range(1, len(seq)):
                hist = seq[max(0, t - max_seq_len): t]
                target = seq[t]
                self.samples.append((hist, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hist, target = self.samples[idx]
        pad_len = self.max_seq_len - len(hist)
        padded = [0] * pad_len + hist
        return (
            torch.LongTensor(padded),
            torch.LongTensor([target]),
        )

def get_retriever_loader(train_seqs: dict, max_seq_len: int,
                         batch_size: int, shuffle: bool = True) -> DataLoader:
    ds = RetrieverDataset(train_seqs, max_seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)