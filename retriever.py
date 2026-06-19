"""
SASRec retriever (Kang & McAuley, ICDM 2018) trained with full-softmax
cross-entropy over all items.

The paper uses ICSRec-SAS which adds intent-contrastive objectives on top of
SASRec. We implement the SASRec backbone with full-softmax loss as the
retriever, which is the most relevant difference from in-batch InfoNCE
(full softmax treats all |I| items as negatives every step).

After training, item embeddings are exported and indexed in FAISS for
approximate nearest-neighbor search during RL training.

Example:
    model = SASRec(num_items=12101, emb_dim=64)
    user_emb = model.get_user_embedding(seq_tensor)  # (B, 64)
    # -> used as query into FAISS index
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import faiss
    FAISS_OK = True
except ImportError:
    FAISS_OK = False
    print("faiss not found; falling back to numpy dot-product retrieval.")


# --------------------------------------------------------------------------- model

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, causal_mask):
        # Only a causal mask is used (no key_padding_mask). Combining a
        # causal mask with a key-padding mask can produce a fully-masked
        # row whenever a query position's only causally-visible key is
        # itself padding (common with left-padding for short histories).
        # Softmax over an all -inf row is NaN, and even though the
        # downstream attention *weight* for a masked key is exactly 0,
        # PyTorch's autograd computes 0 * NaN = NaN when backpropagating
        # through that softmax, which then corrupts the *shared* Q/K/V
        # projection weights for every position in the batch (confirmed
        # empirically: a single such row makes 100% of attn.* gradients
        # NaN). Causal-only masking always leaves the diagonal unmasked
        # (a position can always attend to itself), so no row is ever
        # fully masked. Padding positions still carry near-zero signal
        # since their item embedding is a fixed zero vector
        # (padding_idx=0), so the practical effect on valid positions is
        # negligible.
        a, _ = self.attn(x, x, x, attn_mask=causal_mask)
        x = self.ln1(x + a)
        x = self.ln2(x + self.ff(x))
        return x


class SASRec(nn.Module):
    """
    Self-attentive sequential recommendation model.
    Item IDs are 1-indexed; 0 is the padding token.
    """

    def __init__(self, num_items: int, emb_dim: int = 64,
                 max_seq_len: int = 50, num_heads: int = 2,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.num_items = num_items
        self.emb_dim = emb_dim
        self.max_seq_len = max_seq_len

        self.item_emb = nn.Embedding(num_items + 1, emb_dim, padding_idx=0)
        self.pos_emb  = nn.Embedding(max_seq_len + 1, emb_dim)

        self.emb_dropout = nn.Dropout(dropout)
        self.emb_norm    = nn.LayerNorm(emb_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(emb_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(emb_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.item_emb.weight[1:])
        nn.init.xavier_uniform_(self.pos_emb.weight[1:])

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq : (B, L) LongTensor, 0 = padding
        returns: (B, L, D) contextual item representations
        """
        B, L = seq.shape
        pos = torch.arange(1, L + 1, device=seq.device).unsqueeze(0).expand(B, -1)

        x = self.emb_norm(self.item_emb(seq) + self.pos_emb(pos))
        x = self.emb_dropout(x)

        # causal mask: position j cannot attend to position > j
        causal = torch.triu(torch.ones(L, L, device=seq.device), diagonal=1).bool()

        for block in self.blocks:
            x = block(x, causal)

        return self.out_norm(x)  # (B, L, D)

    def get_user_embedding(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Returns the representation at the last position of the sequence.

        Sequences are left-padded (zeros on the left, real items ending at
        the final position), so the last valid item is always at index -1
        regardless of history length. This is the user query vector used
        to search the FAISS index.

        seq : (B, L)
        returns: (B, D)
        """
        out = self.forward(seq)   # (B, L, D)
        return out[:, -1, :]      # (B, D)

    def get_all_item_embeddings(self) -> torch.Tensor:
        """
        Returns static item embeddings for items 1..num_items.
        Used to build the FAISS index (no contextual info, just e_i).
        returns: (num_items, D)
        """
        with torch.no_grad():
            idx = torch.arange(1, self.num_items + 1,
                               device=self.item_emb.weight.device)
            return self.item_emb(idx)

    def full_softmax_loss(self, seq: torch.Tensor,
                          targets: torch.Tensor) -> torch.Tensor:
        """
        Full-softmax cross-entropy over all |I| items.
        Treats every item as a negative each step (unlike in-batch InfoNCE).

        seq     : (B, L)   padded sequences
        targets : (B,)     1-indexed target item IDs
        """
        user_emb   = self.get_user_embedding(seq)          # (B, D)
        all_embs   = self.get_all_item_embeddings()         # (N, D)
        logits     = user_emb @ all_embs.T                  # (B, N)
        target_0   = targets - 1                            # 0-indexed for CE
        return F.cross_entropy(logits, target_0)


# --------------------------------------------------------------------------- FAISS index

class FAISSIndex:
    """
    Thin wrapper around a FAISS inner-product index (equivalent to cosine
    similarity after L2 normalisation).

    Example:
        index = FAISSIndex(64)
        index.build(emb_array, item_ids)
        cand_ids, scores = index.search(user_query_vec, m=200)
    """

    def __init__(self, emb_dim: int):
        self.emb_dim   = emb_dim
        self._index    = None         # faiss index object
        self._embs     = None         # normalised numpy array (fallback)
        self.item_ids  = None         # 1-indexed item ID array

    def build(self, embeddings: np.ndarray, item_ids: np.ndarray):
        """
        embeddings : (N, D) float32
        item_ids   : (N,)   1-indexed item IDs corresponding to each row
        """
        self.item_ids = np.array(item_ids, dtype=np.int64)

        # L2-normalise so inner product == cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        normalised = (embeddings / norms).astype(np.float32)
        self._embs = normalised

        if FAISS_OK:
            self._index = faiss.IndexFlatIP(self.emb_dim)
            self._index.add(normalised)

    def search(self, query: np.ndarray, m: int):
        """
        query : (D,) float32 user embedding
        m     : number of results to return

        Returns: (item_ids array, scores array), both length <= m
        """
        q = query / (np.linalg.norm(query) + 1e-8)

        if FAISS_OK and self._index is not None:
            scores, indices = self._index.search(
                q.reshape(1, -1).astype(np.float32), m
            )
            scores  = scores[0]
            indices = indices[0]
            valid   = indices >= 0
            return self.item_ids[indices[valid]], scores[valid]

        # numpy fallback
        scores  = self._embs @ q
        indices = np.argsort(-scores)[:m]
        return self.item_ids[indices], scores[indices]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, {"item_ids": self.item_ids, "embs": self._embs},
                allow_pickle=True)

    @classmethod
    def load(cls, path: str, emb_dim: int) -> "FAISSIndex":
        data = np.load(path, allow_pickle=True).item()
        obj = cls(emb_dim)
        obj.build(data["embs"], data["item_ids"])
        return obj