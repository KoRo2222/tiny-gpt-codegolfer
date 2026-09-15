import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from codebot.model import soft_dictionary_attention


def test_output_and_weight_shapes():
    query = torch.randn(2, 3, 8)  # (batch, seq_q, d_k)
    keys = torch.randn(2, 5, 8)  # (batch, seq_k, d_k)
    values = torch.randn(2, 5, 4)  # (batch, seq_k, d_v)

    output, attn_weights = soft_dictionary_attention(query, keys, values)

    assert output.shape == (2, 3, 4)
    assert attn_weights.shape == (2, 3, 5)


def test_weights_form_a_probability_distribution():
    query = torch.randn(1, 4, 8)
    keys = torch.randn(1, 6, 8)
    values = torch.randn(1, 6, 4)

    _, attn_weights = soft_dictionary_attention(query, keys, values)

    row_sums = attn_weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
    assert (attn_weights >= 0).all()


def test_degenerates_to_hard_lookup_for_a_dominant_key():
    # One key scaled way up looks nothing like the query in direction, so
    # give it a huge, unmistakably matching direction instead: a soft
    # dictionary should collapse onto that key's value almost exactly.
    d_k = 8
    dominant_direction = torch.zeros(d_k)
    dominant_direction[0] = 10.0  # large dot product with a matching query

    query = dominant_direction.view(1, 1, d_k)
    keys = torch.stack(
        [dominant_direction, torch.randn(d_k), torch.randn(d_k)]
    ).view(1, 3, d_k)
    values = torch.tensor([[1.0, 2.0], [30.0, 40.0], [50.0, 60.0]]).view(1, 3, 2)

    output, attn_weights = soft_dictionary_attention(query, keys, values)

    assert attn_weights[0, 0, 0].item() > 0.99
    assert torch.allclose(output[0, 0], values[0, 0], atol=0.1)


def test_causal_mask_blocks_future_positions():
    query = torch.randn(1, 4, 8)
    keys = torch.randn(1, 4, 8)
    values = torch.randn(1, 4, 4)

    _, attn_weights = soft_dictionary_attention(query, keys, values, causal=True)

    for i in range(4):
        for j in range(4):
            if j > i:
                assert attn_weights[0, i, j].item() == 0.0
        assert abs(attn_weights[0, i, : i + 1].sum().item() - 1.0) < 1e-6


def test_gradients_flow_through_attention():
    query = torch.randn(1, 3, 8, requires_grad=True)
    keys = torch.randn(1, 5, 8, requires_grad=True)
    values = torch.randn(1, 5, 4, requires_grad=True)

    output, _ = soft_dictionary_attention(query, keys, values)
    output.sum().backward()

    assert query.grad is not None
    assert keys.grad is not None
    assert values.grad is not None
