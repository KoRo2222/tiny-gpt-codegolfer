from __future__ import annotations

import torch
from torch import Tensor, nn


class TokenPositionalEmbedding(nn.Module):
    """Token embedding + learned positional embedding, summed.

    Token embedding alone can't tell "a" at position 0 from "a" at
    position 10 -- attention would see the same vector either way. Adding
    a position vector breaks that symmetry so word order matters.
    """

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, token_ids: Tensor) -> Tensor:
        # token_ids: (batch, seq_len) -> (batch, seq_len, d_model)
        seq_len = token_ids.shape[-1]
        positions = torch.arange(seq_len, device=token_ids.device)
        return self.token_embedding(token_ids) + self.position_embedding(positions)
