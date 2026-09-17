import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from codebot.model import TinyGPT
from codebot.train import compute_loss, estimate_loss, get_batch, train_loop


def test_get_batch_shapes():
    data = np.arange(200, dtype=np.uint16)
    x, y = get_batch(data, block_size=16, batch_size=4)

    assert x.shape == (4, 16)
    assert y.shape == (4, 16)


def test_get_batch_target_is_input_shifted_by_one():
    data = np.arange(200, dtype=np.uint16)
    x, y = get_batch(data, block_size=16, batch_size=4)

    # y[i] was sampled starting one position after x[i], so y[:, :-1]
    # must equal x[:, 1:] -- that's the definition of next-token targets.
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_get_batch_rejects_data_shorter_than_block_size():
    data = np.arange(5, dtype=np.uint16)
    with pytest.raises(ValueError):
        get_batch(data, block_size=16, batch_size=4)


def make_tiny_model(vocab_size=20):
    return TinyGPT(
        vocab_size=vocab_size, d_model=8, n_heads=2, d_ff=16, n_layers=2, max_seq_len=16
    )


def test_compute_loss_is_a_positive_scalar():
    model = make_tiny_model()
    x = torch.randint(0, 20, (4, 8))
    y = torch.randint(0, 20, (4, 8))

    loss = compute_loss(model, x, y)

    assert loss.dim() == 0
    assert loss.item() > 0


def test_estimate_loss_does_not_leave_gradients():
    model = make_tiny_model()
    data = np.random.randint(0, 20, size=200).astype(np.uint16)

    estimate_loss(model, data, block_size=8, batch_size=4, eval_iters=3)

    assert all(p.grad is None for p in model.parameters())


def test_train_loop_reduces_loss_on_a_repeating_pattern():
    # A trivially learnable signal (a short repeating sequence) so a
    # handful of steps should visibly reduce loss -- this is the sanity
    # check that gradients are actually flowing end to end, not just
    # that the loop runs without crashing.
    torch.manual_seed(0)
    vocab_size = 10
    pattern = np.array([1, 2, 3, 4, 5] * 40, dtype=np.uint16)

    model = make_tiny_model(vocab_size=vocab_size)
    history = train_loop(
        model,
        train_data=pattern,
        val_data=pattern,
        steps=100,
        block_size=8,
        batch_size=8,
        lr=1e-2,
        eval_interval=25,
        eval_iters=5,
    )

    assert history[-1]["val_loss"] < history[0]["val_loss"]
