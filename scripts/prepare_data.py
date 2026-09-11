"""Pre-tokenize the corpus into train/val id arrays for pretraining.

Runs the trained BPE tokenizer over data/corpus once, ahead of time, and
saves the result as train.bin / val.bin so the pretraining loop can
np.memmap them instead of re-tokenizing text on every run.

Usage:
    python scripts/prepare_data.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from codebot.data import save_ids, split_train_val  # noqa: E402
from codebot.tokenizer import BPETokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", default=str(ROOT / "data" / "corpus"))
    parser.add_argument(
        "--tokenizer", default=str(ROOT / "data" / "tokenizer.json")
    )
    parser.add_argument("--out-dir", default=str(ROOT / "data"))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()

    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        raise SystemExit(
            f"{tokenizer_path} not found -- run scripts/train_tokenizer.py first"
        )
    tokenizer = BPETokenizer.load(tokenizer_path)

    corpus_dir = Path(args.corpus_dir)
    texts = [
        p.read_text(encoding="utf-8") for p in sorted(corpus_dir.glob("*.txt"))
    ]
    if not texts:
        raise SystemExit(f"no .txt files found under {corpus_dir}")

    ids = tokenizer.encode_with_eot(texts)
    train_ids, val_ids = split_train_val(ids, val_fraction=args.val_fraction)

    out_dir = Path(args.out_dir)
    vocab_size = len(tokenizer.vocab)
    save_ids(train_ids, out_dir / "train.bin", vocab_size)
    save_ids(val_ids, out_dir / "val.bin", vocab_size)

    print(f"vocab size: {vocab_size}")
    print(f"total tokens: {len(ids)}")
    print(f"train: {len(train_ids)} tokens -> {out_dir / 'train.bin'}")
    print(f"val:   {len(val_ids)} tokens -> {out_dir / 'val.bin'}")


if __name__ == "__main__":
    main()
