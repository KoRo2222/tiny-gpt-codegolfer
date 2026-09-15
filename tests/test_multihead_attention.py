import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from codebot.model import MultiHeadAttention


def test_output_shape():
    mha = MultiHeadAttention(d_model=16, n_heads=4)
    x = torch.randn(2, 5, 16)

    out, attn_weights = mha(x)

    assert out.shape == (2, 5, 16)
    assert attn_weights.shape == (2, 4, 5, 5)  # (batch, heads, seq_q, seq_k)


def test_rejects_d_model_not_divisible_by_n_heads():
    with pytest.raises(ValueError):
        MultiHeadAttention(d_model=17, n_heads=4)


def test_causal_by_default():
    mha = MultiHeadAttention(d_model=16, n_heads=4)
    x = torch.randn(1, 4, 16)

    _, attn_weights = mha(x)

    for head in range(4):
        for i in range(4):
            assert attn_weights[0, head, i, i + 1 :].sum().item() == 0.0


def test_non_causal_can_attend_to_future():
    mha = MultiHeadAttention(d_model=16, n_heads=4)
    x = torch.randn(1, 4, 16)

    _, attn_weights = mha(x, causal=False)

    assert attn_weights[0, 0, 0, 1:].sum().item() > 0.0
