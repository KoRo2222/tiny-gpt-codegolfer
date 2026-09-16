from __future__ import annotations

import torch
from torch import Tensor, nn

from .block import TransformerBlock
from .embedding import TokenPositionalEmbedding


class TinyGPT(nn.Module):
    """GPT-2 style decoder: Embed -> N x [Attention -> FFN] -> LN -> Linear.

    Softmax isn't baked into forward() (logits are what cross-entropy and
    sampling temperature need); see next_token_probs for that step.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int = 256,
        n_layers: int = 4,
        max_seq_len: int = 256,
    ) -> None:
        super().__init__()
        self.embedding = TokenPositionalEmbedding(vocab_size, d_model, max_seq_len)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        # Final LayerNorm before the output projection -- every block above
        # is pre-norm, so nothing has normalized the very last residual sum.
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        # Weight tying (GPT-2, "Using the Output Embedding..."): the input
        # embedding and the output projection both map between token ids
        # and d_model-space, so share one matrix instead of learning two.
        self.lm_head.weight = self.embedding.token_embedding.weight

    def forward(self, token_ids: Tensor) -> Tensor:
        # token_ids: (batch, seq_len) -> logits: (batch, seq_len, vocab_size)
        x = self.embedding(token_ids)
        for block in self.blocks:
            x = block(x, causal=True)
        x = self.ln_f(x)
        return self.lm_head(x)

    def next_token_probs(self, token_ids: Tensor) -> Tensor:
        """Softmax over the last position's logits: (batch, vocab_size)."""
        logits = self.forward(token_ids)
        return torch.softmax(logits[:, -1, :], dim=-1)
