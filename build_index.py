"""
Build the FAISS index from a frozen ICSRec-SAS checkpoint.

This replaces retriever *training* (old train_retriever.py): with the mac-4
change, Stage 1 is the released ICSRec model used frozen, so nothing is
trained here -- we only load the checkpoint, pull its item embedding table,
and build the inner-product index the RL environment retrieves from.

    python build_index.py         # after setting DATASET below

Prereqs:
  - cfg.icsrec_ckpt points at the released checkpoint for this dataset
    (ICSRec-SAS-Beauty-0.pt / ICSRec-SAS-ml-1m-0.pt).
  - The dataset .txt file (Beauty.txt / ml-1m.txt) is in cfg.data_dir so that
    num_items (= max_item) matches the checkpoint's item table.
"""

import os

from config import Config
from data import load_and_preprocess
from retriever import FAISSIndex
from icsrec_retriever import load_icsrec_retriever

# Set to "movielens_1m" to build the ml-1m index instead.
DATASET = "amazon_beauty"


def main():
    cfg = Config(dataset=DATASET)

    print(f"[{cfg.dataset}] loading sequences to determine num_items...")
    data = load_and_preprocess(cfg.data_dir, cfg.dataset)
    cfg.num_items = data["num_items"]
    cfg.num_users = data["num_users"]

    print(f"[{cfg.dataset}] loading frozen ICSRec retriever from {cfg.icsrec_ckpt}")
    # load_icsrec_retriever also preflights: it raises if item_size-2 != num_items,
    # i.e. if this checkpoint does not belong to this dataset.
    retriever = load_icsrec_retriever(
        cfg.icsrec_ckpt, cfg.num_items, hidden_size=cfg.ret_emb_dim, device=cfg.device)

    embs = retriever.get_all_item_embeddings().cpu().numpy()  # (num_items, 64)
    item_ids = list(range(1, cfg.num_items + 1))

    print(f"[{cfg.dataset}] building FAISS index over {len(item_ids)} items...")
    index = FAISSIndex(cfg.ret_emb_dim)
    index.build(embs, item_ids)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    out = os.path.join(cfg.checkpoint_dir, "faiss_index.npy")
    index.save(out)
    print(f"FAISS index saved to {out}")


if __name__ == "__main__":
    main()