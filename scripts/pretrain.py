"""Pretrain TinyGPT on the pre-tokenized corpus (data/train.bin, data/val.bin).

Usage:
    python scripts/pretrain.py --steps 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from codebot.data.prepare import load_ids  # noqa: E402
from codebot.model import TinyGPT  # noqa: E402
from codebot.tokenizer import BPETokenizer  # noqa: E402
from codebot.train import train_loop  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default=str(ROOT / "data" / "tokenizer.json"))
    parser.add_argument("--train-bin", default=str(ROOT / "data" / "train.bin"))
    parser.add_argument("--val-bin", default=str(ROOT / "data" / "val.bin"))
    parser.add_argument("--out", default=str(ROOT / "data" / "checkpoint.pt"))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    args = parser.parse_args()

    for path in (args.tokenizer, args.train_bin, args.val_bin):
        if not Path(path).exists():
            raise SystemExit(
                f"{path} not found -- run train_tokenizer.py and prepare_data.py first"
            )

    tokenizer = BPETokenizer.load(args.tokenizer)
    vocab_size = len(tokenizer.vocab)
    train_data = load_ids(args.train_bin, vocab_size)
    val_data = load_ids(args.val_bin, vocab_size)

    model = TinyGPT(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        max_seq_len=args.block_size,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"vocab_size={vocab_size}, params={n_params:,}")

    prompt = "def "
    prompt_ids = torch.tensor([tokenizer.encode(prompt)])
    before = tokenizer.decode(
        model.generate(prompt_ids, max_new_tokens=30)[0].tolist()
    )
    print(f"\nbefore training: {before!r}")

    train_loop(
        model,
        train_data,
        val_data,
        steps=args.steps,
        block_size=args.block_size,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_interval=args.eval_interval,
    )

    after = tokenizer.decode(
        model.generate(prompt_ids, max_new_tokens=30)[0].tolist()
    )
    print(f"\nafter training:  {after!r}")

    torch.save(model.state_dict(), args.out)
    print(f"\nsaved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
