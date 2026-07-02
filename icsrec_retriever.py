"""
Frozen ICSRec-SAS retriever for the MARS pipeline (Stage 1).

This replaces the previously-used plain full-softmax SASRec (train_retriever.py)
with the *actual* pretrained ICSRec-SAS model from the official release:

    Qin et al., "Intent Contrastive Learning with Cross Subsequences for
    Sequential Recommendation", WSDM 2024.  https://github.com/QinHsiu/ICSRec

Only the inference path is vendored here: the transformer encoder + item
embedding table. None of the training-time machinery (intent K-means, CICL /
FICL contrastive losses) is needed, because the paper uses the retriever
*frozen* -- Stages 1-2 are collapsed and ICSRec similarity scores serve
directly as r_u(i). The vendored module names below (item_embeddings,
position_embeddings, item_encoder, LayerNorm) are kept identical to the
official SASRecModel so that the released checkpoint state_dict loads with
zero missing/unexpected keys.

Vendored transformer code (LayerNorm / SelfAttention / Intermediate / Layer /
Encoder / SASRecModel.forward) is adapted from the ICSRec release, which is in
turn based on CoSeRec / ICLRec (Copyright (c) 2022 salesforce.com, inc.,
SPDX-License-Identifier: BSD-3-Clause). See the ICSRec repo LICENSE.

Checkpoint / dataset facts (verified against the released data files):
    Beauty : max_item=12101 -> item_size=12103 (0=pad, 1..12101 items, 12102=mask)
    ml-1m  : max_item=3416  -> item_size=3418
ICSRec-SAS hyperparameters (from the release): hidden_size=64, num_hidden_layers=2,
num_attention_heads=2, max_seq_length=50, hidden_act=gelu.

Example:
    retr = load_icsrec_retriever(
        "src/output/ICSRec-SAS-Beauty-0.pt", num_items=12101, device="cpu")
    q = retr.get_user_embedding(seq)            # (B, 64), last-position output
    E = retr.get_all_item_embeddings()          # (num_items, 64) for FAISS
"""

import copy
import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------- activations
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}


# --------------------------------------------------------------------- transformer
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(*new_shape).permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        q = self.transpose_for_scores(self.query(input_tensor))
        k = self.transpose_for_scores(self.key(input_tensor))
        v = self.transpose_for_scores(self.value(input_tensor))

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        scores = scores + attention_mask
        probs = self.attn_dropout(nn.Softmax(dim=-1)(scores))
        ctx = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous()
        ctx = ctx.view(*(ctx.size()[:-2] + (self.all_head_size,)))
        hidden = self.out_dropout(self.dense(ctx))
        return self.LayerNorm(hidden + input_tensor)


class Intermediate(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dense_1 = nn.Linear(args.hidden_size, args.hidden_size * 4)
        self.intermediate_act_fn = (
            ACT2FN[args.hidden_act] if isinstance(args.hidden_act, str) else args.hidden_act
        )
        self.dense_2 = nn.Linear(args.hidden_size * 4, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, input_tensor):
        h = self.intermediate_act_fn(self.dense_1(input_tensor))
        h = self.dropout(self.dense_2(h))
        return self.LayerNorm(h + input_tensor)


class Layer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attention = SelfAttention(args)
        self.intermediate = Intermediate(args)

    def forward(self, hidden_states, attention_mask):
        return self.intermediate(self.attention(hidden_states, attention_mask))


class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        layer = Layer(args)
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(args.num_hidden_layers)]
        )

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_layers.append(hidden_states)
        return all_layers


class SASRecModel(nn.Module):
    """Inference-only mirror of the official ICSRec SASRecModel (same key names)."""

    def __init__(self, args):
        super().__init__()
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

    def add_position_embedding(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        emb = self.item_embeddings(sequence) + self.position_embeddings(position_ids)
        return self.dropout(self.LayerNorm(emb))

    def forward(self, input_ids):
        attention_mask = (input_ids > 0).long()
        extended = attention_mask.unsqueeze(1).unsqueeze(2)
        max_len = attention_mask.size(-1)
        subsequent = torch.triu(torch.ones((1, max_len, max_len), device=input_ids.device), diagonal=1)
        subsequent = (subsequent == 0).unsqueeze(1).long()
        extended = extended * subsequent
        extended = extended.to(dtype=next(self.parameters()).dtype)
        extended = (1.0 - extended) * -10000.0

        seq_emb = self.add_position_embedding(input_ids)
        layers = self.item_encoder(seq_emb, extended, output_all_encoded_layers=True)
        return layers[-1]  # (B, L, H)


# --------------------------------------------------------------------- adapter
class ICSRecRetriever(nn.Module):
    """
    Drop-in replacement for the old SASRec retriever. Exposes the exact
    interface the MARS pipeline already calls:
        - get_user_embedding(seq) -> (B, H)   last-position encoder output
        - get_all_item_embeddings() -> (num_items, H)   real items 1..num_items
        - .item_emb  (alias of the ICSRec item embedding table, for ILD)
    """

    def __init__(self, model: SASRecModel, num_items: int, max_seq_length: int):
        super().__init__()
        self.model = model
        self.num_items = num_items          # real item count = max_item (rows 1..num_items)
        self.max_seq_length = max_seq_length  # fixed by the checkpoint (ICSRec: 50)

    # alias so evaluate.get_item_embeddings_for_ild(retriever) works unchanged:
    # retriever.item_emb.weight / retriever.item_emb(idx) both resolve here.
    @property
    def item_emb(self) -> nn.Embedding:
        return self.model.item_embeddings

    def get_user_embedding(self, seq: torch.Tensor) -> torch.Tensor:
        # ICSRec fixes max_seq_length=50 for every dataset (incl. ml-1m, whose
        # MARS window is 200). Truncate to the most recent max_seq_length items
        # so a longer caller window never indexes past position_embeddings.
        if seq.size(1) > self.max_seq_length:
            seq = seq[:, -self.max_seq_length:]
        return self.model(seq)[:, -1, :]  # (B, H)

    def get_all_item_embeddings(self) -> torch.Tensor:
        with torch.no_grad():
            idx = torch.arange(
                1, self.num_items + 1, device=self.model.item_embeddings.weight.device
            )
            return self.model.item_embeddings(idx)  # (num_items, H)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        return self.get_user_embedding(seq)


def _infer_dataset_args(item_size: int, max_seq_length: int,
                        hidden_size: int = 64) -> SimpleNamespace:
    return SimpleNamespace(
        item_size=item_size,
        hidden_size=hidden_size,
        max_seq_length=max_seq_length,
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_act="gelu",
        hidden_dropout_prob=0.5,          # inactive at eval; value irrelevant
        attention_probs_dropout_prob=0.5, # inactive at eval; value irrelevant
    )


def load_icsrec_retriever(ckpt_path: str, num_items: int,
                          hidden_size: int = 64, device: str = "cpu") -> ICSRecRetriever:
    """
    Load a released ICSRec-SAS checkpoint as a frozen retriever.

    item_size and max_seq_length are read directly from the checkpoint tensors
    (not trusted from config), so a checkpoint built with a different catalog
    or sequence length can never be silently loaded into a mismatched model.

    This also doubles as the checkpoint/dataset preflight: num_items (from the
    dataset) must equal item_size-2 (0=pad, 1..num_items items, num_items+1=mask).
    A mismatch here is exactly the silent-corruption failure we want to catch.
    """
    state = torch.load(ckpt_path, map_location=device)

    item_size = state["item_embeddings.weight"].shape[0]
    max_seq_length = state["position_embeddings.weight"].shape[0]
    ckpt_hidden = state["item_embeddings.weight"].shape[1]

    if item_size - 2 != num_items:
        raise RuntimeError(
            f"checkpoint/dataset mismatch loading {ckpt_path}: checkpoint has "
            f"item_size={item_size} (=> {item_size-2} real items) but the dataset "
            f"reports num_items={num_items}. Wrong checkpoint for this dataset?"
        )
    if ckpt_hidden != hidden_size:
        raise RuntimeError(
            f"hidden size mismatch: checkpoint={ckpt_hidden}, expected={hidden_size}"
        )

    args = _infer_dataset_args(item_size, max_seq_length, hidden_size)
    model = SASRecModel(args)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"ICSRec checkpoint key mismatch loading {ckpt_path}\n"
            f"  missing={missing}\n  unexpected={unexpected}"
        )

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)

    retr = ICSRecRetriever(model, num_items=num_items,
                           max_seq_length=max_seq_length).to(device)
    retr.eval()
    return retr