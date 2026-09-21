"""Evaluate a checkpoint's pass rate against data/rl/tasks.jsonl.

Prints a per-task pass/fail table and an aggregate "solved N/M" summary,
and appends that summary to data/eval_history.jsonl so pass rate can be
tracked across training runs instead of eyeballing single samples.

Usage:
    python scripts/evaluate.py --checkpoint data/checkpoint_rl.pt
    python scripts/evaluate.py --checkpoint data/checkpoint_sft.pt --num-samples 4 --temperature 0.8
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from codebot.model import load_checkpoint  # noqa: E402
from codebot.rl import evaluate_model  # noqa: E402
from codebot.tokenizer import BPETokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default=str(ROOT / "data" / "tokenizer.json"))
    parser.add_argument(
        "--checkpoint", default=str(ROOT / "data" / "checkpoint_rl.pt")
    )
    parser.add_argument("--tasks", default=str(ROOT / "data" / "rl" / "tasks.jsonl"))
    parser.add_argument(
        "--log-file", default=str(ROOT / "data" / "eval_history.jsonl")
    )
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    args = parser.parse_args()

    for path in (args.tokenizer, args.checkpoint, args.tasks):
        if not Path(path).exists():
            raise SystemExit(f"{path} not found")

    tokenizer = BPETokenizer.load(args.tokenizer)
    model = load_checkpoint(args.checkpoint)
    tasks = [
        json.loads(line)
        for line in Path(args.tasks).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    summary = evaluate_model(
        model,
        tokenizer,
        tasks,
        num_samples=args.num_samples,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    for result in summary["results"]:
        status = "PASS" if result["pass_rate"] > 0 else "fail"
        rate = f"{result['pass_rate']:.2f}"
        length = f"{result['avg_length']:.0f}" if result["avg_length"] else "-"
        print(f"  [{status}] {result['entry_point']:<28} pass_rate={rate} avg_len={length}")

    print(
        f"\nsolved {summary['solved']}/{summary['num_tasks']} tasks "
        f"(mean pass_rate={summary['mean_pass_rate']:.3f})"
    )

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "num_tasks": summary["num_tasks"],
        "solved": summary["solved"],
        "mean_pass_rate": summary["mean_pass_rate"],
    }
    log_path = Path(args.log_file)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")
    print(f"logged to {log_path}")


if __name__ == "__main__":
    main()
