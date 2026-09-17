import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from codebot.model import TinyGPT, load_checkpoint, save_checkpoint

CONFIG = {
    "vocab_size": 20,
    "d_model": 8,
    "n_heads": 2,
    "d_ff": 16,
    "n_layers": 2,
    "max_seq_len": 16,
}


def test_load_checkpoint_reconstructs_matching_config(tmp_path):
    model = TinyGPT(**CONFIG)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(model, path, CONFIG)

    loaded = load_checkpoint(path)

    assert len(loaded.blocks) == CONFIG["n_layers"]
    assert loaded.lm_head.out_features == CONFIG["vocab_size"]


def test_load_checkpoint_reproduces_identical_outputs(tmp_path):
    model = TinyGPT(**CONFIG)
    model.eval()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(model, path, CONFIG)

    loaded = load_checkpoint(path)
    loaded.eval()

    token_ids = torch.randint(0, CONFIG["vocab_size"], (1, 5))
    assert torch.allclose(model(token_ids), loaded(token_ids))


def test_load_checkpoint_stashes_config_for_resaving(tmp_path):
    model = TinyGPT(**CONFIG)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(model, path, CONFIG)

    loaded = load_checkpoint(path)

    assert loaded.config == CONFIG
