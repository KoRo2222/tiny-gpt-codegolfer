from __future__ import annotations

from pathlib import Path

import torch

from .gpt import TinyGPT


def save_checkpoint(model: TinyGPT, path: str | Path, config: dict) -> None:
    """Save weights bundled with the config needed to rebuild the model.

    Without the config, whoever loads this later has to know and re-type
    every constructor argument (d_model, n_heads, ...) exactly right, or
    load_state_dict fails on a shape mismatch with no clue why.
    """
    torch.save({"state_dict": model.state_dict(), "config": config}, path)


def load_checkpoint(path: str | Path) -> TinyGPT:
    checkpoint = torch.load(path, map_location="cpu")
    model = TinyGPT(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    # Stash the config so callers that re-save (e.g. after further fine-
    # tuning) don't need to re-read the checkpoint file just for this.
    model.config = checkpoint["config"]
    return model
