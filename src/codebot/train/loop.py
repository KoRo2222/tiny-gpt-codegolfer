from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ..model import TinyGPT


def get_batch(
    data: np.ndarray, block_size: int, batch_size: int, device: str = "cpu"
) -> tuple[Tensor, Tensor]:
    """Sample random (input, target) windows for next-token prediction.

    target is input shifted by one position, so predicting target[i]
    from input[:i+1] is exactly next-token prediction at every position
    in the window, not just the last one.
    """
    max_start = len(data) - block_size - 1
    if max_start < 1:
        raise ValueError(
            f"data has {len(data)} tokens, too short for block_size={block_size}"
        )
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack(
        [torch.from_numpy(data[s : s + block_size].astype(np.int64)) for s in starts]
    )
    y = torch.stack(
        [
            torch.from_numpy(data[s + 1 : s + block_size + 1].astype(np.int64))
            for s in starts
        ]
    )
    return x.to(device), y.to(device)


def compute_loss(model: TinyGPT, x: Tensor, y: Tensor) -> Tensor:
    logits = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


@torch.no_grad()
def estimate_loss(
    model: TinyGPT,
    data: np.ndarray,
    block_size: int,
    batch_size: int,
    eval_iters: int,
    device: str = "cpu",
) -> float:
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data, block_size, batch_size, device)
        losses.append(compute_loss(model, x, y).item())
    model.train(was_training)
    return sum(losses) / len(losses)


def train_loop(
    model: TinyGPT,
    train_data: np.ndarray,
    val_data: np.ndarray,
    steps: int,
    block_size: int = 64,
    batch_size: int = 16,
    lr: float = 3e-4,
    eval_interval: int = 50,
    eval_iters: int = 20,
    device: str = "cpu",
    log: Callable[[str], None] = print,
) -> list[dict]:
    """Train by sampling random windows for `steps` optimizer steps.

    Returns the eval history (train/val loss every eval_interval steps),
    for plotting or just eyeballing that val_loss actually goes down.
    """
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history = []
    for step in range(1, steps + 1):
        x, y = get_batch(train_data, block_size, batch_size, device)
        loss = compute_loss(model, x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0 or step == steps:
            val_loss = estimate_loss(
                model, val_data, block_size, batch_size, eval_iters, device
            )
            history.append(
                {"step": step, "train_loss": loss.item(), "val_loss": val_loss}
            )
            log(f"step {step}/{steps}: train_loss={loss.item():.4f} val_loss={val_loss:.4f}")

    return history
