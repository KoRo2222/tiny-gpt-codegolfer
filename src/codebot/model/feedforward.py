from __future__ import annotations

from torch import Tensor, nn


class FeedForward(nn.Module):
    """Position-wise FFN: Linear -> GELU -> Linear.

    Attention only mixes information *between* positions; this is where
    each position's mixed representation gets nonlinearly transformed on
    its own, one token at a time.
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
