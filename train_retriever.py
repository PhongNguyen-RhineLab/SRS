"""
Train the Stage-1 retriever (SASRec backbone, full-softmax cross-entropy
over all items, Section 5.1).

This corresponds to ICSRec-SAS minus the two contrastive objectives (CICL,
FICL); the full-softmax SASRec is the closest open re-implementation
available without the original ICSRec training code.

Example:
    python train_retriever.py
"""

import os
import torch
from tqdm import tqdm

from config import Config

# Set to "movielens_1m" to run against MovieLens-1M instead.
DATASET = "amazon_beauty"
from data import load_and_preprocess, split_data, get_retriever_loader
from retriever import SASRec, FAISSIndex


def train_retriever(cfg: Config, train_seqs: dict, num_items: int):
    device = cfg.device
    model = SASRec(
        num_items=num_items,
        emb_dim=cfg.ret_emb_dim,
        max_seq_len=cfg.max_seq_len,
        num_heads=cfg.ret_num_heads,
        num_layers=cfg.ret_num_layers,
        dropout=cfg.ret_dropout,
    ).to(device)

    loader = get_retriever_loader(train_seqs, cfg.max_seq_len, cfg.ret_batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.ret_lr)

    print(f"Training SASRec retriever on {len(loader.dataset)} samples, "
          f"{cfg.ret_epochs} epochs...")

    for epoch in range(1, cfg.ret_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Retriever epoch {epoch}/{cfg.ret_epochs}")
        for seq, target in pbar:
            seq = seq.to(device)
            target = target.squeeze(-1).to(device)

            optimizer.zero_grad()
            loss = model.full_softmax_loss(seq, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch}: avg full-softmax CE loss = {avg_loss:.4f}")

    return model


def build_faiss_index(cfg: Config, model: SASRec, num_items: int) -> FAISSIndex:
    print("Building FAISS index from learned item embeddings...")
    model.eval()
    with torch.no_grad():
        embs = model.get_all_item_embeddings().cpu().numpy()  # (num_items, D)

    item_ids = list(range(1, num_items + 1))
    index = FAISSIndex(emb_dim=cfg.ret_emb_dim)
    index.build(embs, item_ids)
    return index


def main():
    cfg = Config(dataset=DATASET)
    torch.manual_seed(cfg.seed)

    print("Loading and preprocessing Amazon Beauty 2014 5-core dataset...")
    data = load_and_preprocess(cfg.data_dir, cfg.dataset)
    cfg.num_items = data["num_items"]
    cfg.num_users = data["num_users"]

    train_seqs, val_seqs, test_seqs = split_data(data["sequences"])
    print(f"Train users: {len(train_seqs)}, Val: {len(val_seqs)}, Test: {len(test_seqs)}")

    model = train_retriever(cfg, train_seqs, cfg.num_items)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(),
               os.path.join(cfg.checkpoint_dir, "sasrec_retriever.pt"))
    print("Retriever checkpoint saved.")

    index = build_faiss_index(cfg, model, cfg.num_items)
    index.save(os.path.join(cfg.checkpoint_dir, "faiss_index.npy"))
    print("FAISS index saved.")


if __name__ == "__main__":
    main()