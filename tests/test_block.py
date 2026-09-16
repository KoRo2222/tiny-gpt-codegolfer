import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from codebot.model import TransformerBlock


def test_output_shape_matches_input():
    block = TransformerBlock(d_model=16, n_heads=4, d_ff=32)
    x = torch.randn(2, 5, 16)

    out = block(x)

    assert out.shape == x.shape


def test_zeroed_sublayers_leave_the_residual_untouched():
    # If both sublayers are forced to output exactly zero, x + 0 + 0 must
    # equal x -- proof this is x + sublayer(...), not a replacement of x.
    block = TransformerBlock(d_model=16, n_heads=4, d_ff=32)
    with torch.no_grad():
        block.attention.out_proj.weight.zero_()
        block.attention.out_proj.bias.zero_()
        block.feed_forward.net[-1].weight.zero_()
        block.feed_forward.net[-1].bias.zero_()

    x = torch.randn(1, 3, 16)
    out = block(x)

    assert torch.allclose(out, x, atol=1e-6)


def test_causal_flag_forwarded_to_attention():
    block = TransformerBlock(d_model=16, n_heads=4, d_ff=32)
    x = torch.randn(1, 4, 16)

    causal_out = block(x, causal=True)
    non_causal_out = block(x, causal=False)

    assert not torch.allclose(causal_out, non_causal_out)
