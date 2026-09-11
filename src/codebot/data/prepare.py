from __future__ import annotations

from pathlib import Path

import numpy as np


def split_train_val(
    ids: list[int], val_fraction: float = 0.1
) -> tuple[list[int], list[int]]:
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")
    split = int(len(ids) * (1 - val_fraction))
    return ids[:split], ids[split:]


def save_ids(ids: list[int], path: str | Path, vocab_size: int) -> None:
    """Dump token ids as a flat little-endian array (nanoGPT-style .bin).

    Kept as raw dtype bytes, not JSON, so the future GPT-2 data loader can
    np.memmap it instead of loading the whole corpus into memory.
    """
    dtype = np.uint16 if vocab_size <= 2**16 else np.uint32
    np.asarray(ids, dtype=dtype).tofile(path)


def load_ids(path: str | Path, vocab_size: int) -> np.ndarray:
    dtype = np.uint16 if vocab_size <= 2**16 else np.uint32
    return np.fromfile(path, dtype=dtype)
