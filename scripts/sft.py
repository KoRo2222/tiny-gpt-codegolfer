"""Supervised fine-tuning on prompt/response pairs (data/sft/examples.jsonl).

Starts from a pretrained checkpoint (scripts/pretrain.py's output) and
fine-tunes it to follow instructions, masking the loss so only the
response tokens are scored.

Usage:
    python scripts/sft.py --epochs 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from codebot.model import load_checkpoint, save_checkpoint  # noqa: E402
from codebot.tokenizer import BPETokenizer  # noqa: E402
from codebot.train import sft_loop  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default=str(ROOT / "data" / "tokenizer.json"))
    parser.add_argument(
        "--checkpoint", default=str(ROOT / "data" / "checkpoint.pt")
    )
    parser.add_argument(
        "--examples", default=str(ROOT / "data" / "sft" / "examples.jsonl")
    )
    parser.add_argument("--out", default=str(ROOT / "data" / "checkpoint_sft.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    for path in (args.tokenizer, args.checkpoint, args.examples):
        if not Path(path).exists():
            raise SystemExit(f"{path} not found -- run pretrain.py first")

    tokenizer = BPETokenizer.load(args.tokenizer)
    model = load_checkpoint(args.checkpoint)

    examples = [
        json.loads(line)
        for line in Path(args.examples).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"loaded {len(examples)} prompt/response examples")

    prompt = "# Write a function that checks whether a string is a palindrome.\n"
    prompt_ids = torch.tensor([tokenizer.encode(prompt)])
    before = tokenizer.decode(model.generate(prompt_ids, max_new_tokens=40)[0].tolist())
    print(f"\nbefore SFT: {before!r}")

    sft_loop(model, tokenizer, examples, epochs=args.epochs, lr=args.lr)

    after = tokenizer.decode(model.generate(prompt_ids, max_new_tokens=40)[0].tolist())
    print(f"\nafter SFT:  {after!r}")

    save_checkpoint(model, args.out, model.config)
    print(f"\nsaved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
