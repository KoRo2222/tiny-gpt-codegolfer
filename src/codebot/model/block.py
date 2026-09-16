from __future__ import annotations

from torch import Tensor, nn

from .attention import MultiHeadAttention
from .feedforward import FeedForward


class TransformerBlock(nn.Module):
    """One decoder block: Attention -> FFN, both pre-norm + residual.

    Pre-norm (normalize, then feed the sublayer, then add the *un*-
    normalized residual back) is what GPT-2 uses instead of the original
    Transformer's post-norm; it trains far more stably once you stack
    more than a couple of these.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff)

    def forward(self, x: Tensor, causal: bool = True) -> Tensor:
        x = x + self.attention(self.ln1(x), causal=causal)[0]
        x = x + self.feed_forward(self.ln2(x))
        return x
