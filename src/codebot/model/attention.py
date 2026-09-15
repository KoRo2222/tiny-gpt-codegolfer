from __future__ import annotations

import math

import torch
from torch import Tensor


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
