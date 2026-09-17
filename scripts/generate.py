"""Generate text from a trained TinyGPT checkpoint.

Usage:
    python scripts/generate.py --prompt "def " --max-new-tokens 50
    python scripts/generate.py --prompt "def " --temperature 0.5 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from codebot.model import load_checkpoint  # noqa: E402
from codebot.tokenizer import BPETokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default=str(ROOT / "data" / "tokenizer.json"))
    parser.add_argument("--checkpoint", default=str(ROOT / "data" / "checkpoint.pt"))
    parser.add_argument("--prompt", default="def ")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    for path in (args.tokenizer, args.checkpoint):
        if not Path(path).exists():
            raise SystemExit(
                f"{path} not found -- run train_tokenizer.py and pretrain.py first"
            )

    if args.seed is not None:
        torch.manual_seed(args.seed)

    tokenizer = BPETokenizer.load(args.tokenizer)
    model = load_checkpoint(args.checkpoint)

    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)])
    out_ids = model.generate(
        prompt_ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature
    )
    print(tokenizer.decode(out_ids[0].tolist()))


if __name__ == "__main__":
    main()
