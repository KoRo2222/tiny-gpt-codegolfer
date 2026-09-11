import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from codebot.data import save_ids, split_train_val
from codebot.data.prepare import load_ids


def test_split_train_val_proportions():
    ids = list(range(100))
    train, val = split_train_val(ids, val_fraction=0.2)
    assert len(train) == 80
    assert len(val) == 20
    assert train + val == ids


def test_split_train_val_rejects_bad_fraction():
    for bad in (-0.1, 1.0, 1.5):
        try:
            split_train_val([1, 2, 3], val_fraction=bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for val_fraction={bad}")


def test_save_and_load_ids_roundtrip(tmp_path):
    ids = [0, 1, 255, 300, 512]
    path = tmp_path / "ids.bin"
    save_ids(ids, path, vocab_size=513)

    loaded = load_ids(path, vocab_size=513)
    assert loaded.dtype == np.uint16
    assert loaded.tolist() == ids


def test_save_ids_uses_wider_dtype_for_large_vocab(tmp_path):
    path = tmp_path / "ids.bin"
    save_ids([70000], path, vocab_size=100_000)
    loaded = load_ids(path, vocab_size=100_000)
    assert loaded.dtype == np.uint32
    assert loaded.tolist() == [70000]
