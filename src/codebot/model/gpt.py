from __future__ import annotations

import torch
from torch import Tensor, nn

from .attention import MultiHeadAttention
from .embedding import TokenPositionalEmbedding
from .feedforward import FeedForward


class TinyGPT(nn.Module):
    """One decoder block: Embed -> Attention -> FFN -> Linear(-> Softmax).

    Pre-norm + residual around attention and the FFN, same as GPT-2 --
    without them, gradients through a real transformer don't train.
    Softmax isn't baked into forward() (logits are what cross-entropy and
    sampling temperature need); see next_token_probs for that step.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int = 256,
        max_seq_len: int = 256,
    ) -> None:
        super().__init__()
        self.embedding = TokenPositionalEmbedding(vocab_size, d_model, max_seq_len)
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids: Tensor) -> Tensor:
        # token_ids: (batch, seq_len) -> logits: (batch, seq_len, vocab_size)
        x = self.embedding(token_ids)
        x = x + self.attention(self.ln1(x), causal=True)[0]
        x = x + self.feed_forward(self.ln2(x))
        return self.lm_head(x)

    def next_token_probs(self, token_ids: Tensor) -> Tensor:
        """Softmax over the last position's logits: (batch, vocab_size)."""
        logits = self.forward(token_ids)
        return torch.softmax(logits[:, -1, :], dim=-1)
