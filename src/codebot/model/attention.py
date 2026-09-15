from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def soft_dictionary_attention(
    query: Tensor, keys: Tensor, values: Tensor, causal: bool = False
) -> tuple[Tensor, Tensor]:
    """Scaled dot-product attention, i.e. a "soft" dictionary lookup.

    A hard dictionary picks exactly one value by exact key match. This
    instead scores the query against every key by similarity, turns those
    scores into a probability distribution with softmax, and returns the
    probability-weighted sum of all values -- a lookup that blends
    everything close to the query instead of returning one hard hit.

    Shapes (leading dims are any number of batch/head dims):
        query:  (..., seq_q, d_k)
        keys:   (..., seq_k, d_k)
        values: (..., seq_k, d_v)

    Returns:
        output:       (..., seq_q, d_v)
        attn_weights: (..., seq_q, seq_k), each row sums to 1

    If causal=True, query position i is only allowed to attend to key
    positions <= i (for autoregressive, GPT-style decoding).
    """
    d_k = query.shape[-1]
    scores = query @ keys.transpose(-2, -1) / math.sqrt(d_k)

    if causal:
        seq_q, seq_k = scores.shape[-2], scores.shape[-1]
        future = torch.triu(
            torch.ones(seq_q, seq_k, dtype=torch.bool, device=scores.device),
            diagonal=1,
        )
        scores = scores.masked_fill(future, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)
    output = attn_weights @ values
    return output, attn_weights


class MultiHeadAttention(nn.Module):
    """Causal self-attention: run soft_dictionary_attention in parallel

    across n_heads smaller subspaces instead of one d_model-wide lookup,
    so different heads can learn to attend on different criteria (e.g.
    one head tracks the matching bracket, another the current indent).
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split_heads(self, x: Tensor, batch: int, seq_len: int) -> Tensor:
        # (batch, seq_len, d_model) -> (batch, n_heads, seq_len, d_head)
        return x.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)

    def forward(self, x: Tensor, causal: bool = True) -> tuple[Tensor, Tensor]:
        batch, seq_len, d_model = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = (self._split_heads(t, batch, seq_len) for t in (q, k, v))

        out, attn_weights = soft_dictionary_attention(q, k, v, causal=causal)

        out = out.transpose(1, 2).reshape(batch, seq_len, d_model)
        return self.out_proj(out), attn_weights
