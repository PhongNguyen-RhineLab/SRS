"""
Data loading and preprocessing for Amazon Beauty 2014 5-core.

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
import gzip
import json
import pickle
import requests
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------------------------------- download

BEAUTY_URL = (
    "http://snap.stanford.edu/data/amazon/productGraph/"
    "categoryFiles/reviews_Beauty_5.json.gz"
)
BEAUTY_URL_ALT = (
    "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon/"
    "productGraph/categoryFiles/reviews_Beauty_5.json.gz"
)


def download_dataset(data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, "reviews_Beauty_5.json.gz")

    if os.path.exists(filepath):
        print("Dataset file already present, skipping download.")
        return filepath

    for url in [BEAUTY_URL, BEAUTY_URL_ALT]:
        print(f"Downloading from {url} ...")
        try:
            r = requests.get(url, stream=True, timeout=60)
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(filepath, "wb") as f:
                with tqdm(total=total, unit="B", unit_scale=True, desc="Beauty 5-core") as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            print("Download complete.")
            return filepath
        except Exception as e:
            print(f"  Failed: {e}")

    raise RuntimeError(
        "Could not download dataset. Please download manually from:\n"
        "  http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
        "reviews_Beauty_5.json.gz\n"
        f"and place it at {filepath}"
    )


# --------------------------------------------------------------------------- preprocess

def load_and_preprocess(data_dir: str) -> dict:
    """
    Returns a dict with keys:
        sequences  : {user_idx: [item_idx, ...]}  (1-indexed, chronological)
        num_items  : int
        num_users  : int
    """
    cache_path = os.path.join(data_dir, "processed.pkl")

    if os.path.exists(cache_path):
        print("Loading cached preprocessed data...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    filepath = download_dataset(data_dir)

    print("Parsing and preprocessing dataset...")
    triples = []  # (user_str, item_str, timestamp)
    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        for line in f:
            rev = json.loads(line.strip())
            triples.append((
                rev["reviewerID"],
                rev["asin"],
                int(rev["unixReviewTime"]),
            ))

    # build 1-indexed mappings (0 is padding)
    users_sorted = sorted(set(u for u, _, _ in triples))
    items_sorted = sorted(set(i for _, i, _ in triples))

    user2idx = {u: i + 1 for i, u in enumerate(users_sorted)}
    item2idx = {i: j + 1 for j, i in enumerate(items_sorted)}

    num_users = len(users_sorted)
    num_items = len(items_sorted)

    # build per-user sorted interaction lists
    raw_seqs: dict = defaultdict(list)
    for u, i, t in triples:
        raw_seqs[user2idx[u]].append((t, item2idx[i]))

    sequences = {}
    for u, pairs in raw_seqs.items():
        sorted_items = [item for _, item in sorted(pairs)]
        # deduplicate while preserving order
        seen: set = set()
        seq = []
        for item in sorted_items:
            if item not in seen:
                seen.add(item)
                seq.append(item)
        if len(seq) >= 3:   # need at least 3 for LL2O
            sequences[u] = seq

    print(f"Users after filtering: {len(sequences)}, Items: {num_items}")

    result = {
        "sequences": sequences,
        "num_items": num_items,
        "num_users": num_users,
        "user2idx": user2idx,
        "item2idx": item2idx,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    print(f"Preprocessed data cached at {cache_path}")

    return result


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
