import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from codebot.model import FeedForward


def test_output_shape_matches_d_model():
    ff = FeedForward(d_model=16, d_ff=64)
    x = torch.randn(2, 5, 16)

    out = ff(x)

    assert out.shape == (2, 5, 16)


def test_is_nonlinear():
    ff = FeedForward(d_model=8, d_ff=32)
    x = torch.randn(1, 3, 8)

    # If the FFN behaved linearly, scaling the input would scale the
    # output by the same factor. GELU breaks that.
    out = ff(x)
    scaled_out = ff(3.0 * x)

    assert not torch.allclose(scaled_out, 3.0 * out, atol=1e-4)


def test_applies_independently_per_position():
    ff = FeedForward(d_model=8, d_ff=32)
    x = torch.randn(1, 4, 8)

    full_out = ff(x)
    single_out = ff(x[:, 2:3, :])

    assert torch.allclose(full_out[:, 2:3, :], single_out, atol=1e-6)
