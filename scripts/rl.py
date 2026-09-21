"""GRPO fine-tuning against an execution-based code-golf reward.

Starts from the SFT checkpoint (scripts/sft.py's output), which also
serves as a frozen KL-reference policy, and rewards short code that
actually passes its tests (data/rl/tasks.jsonl).

Usage:
    python scripts/rl.py --steps 100
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from codebot.model import load_checkpoint, save_checkpoint  # noqa: E402
from codebot.rl import code_golf_reward, grpo_loop, run_tests, sample_completion  # noqa: E402
from codebot.tokenizer import BPETokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def show_sample(model, tokenizer, task, label) -> None:
    prompt_ids = tokenizer.encode(task["prompt"])
    response_ids, _ = sample_completion(model, tokenizer, prompt_ids, max_new_tokens=120)
    eot_id = tokenizer.special_tokens[tokenizer.EOT_TOKEN]
    code = tokenizer.decode([t for t in response_ids if t != eot_id])
    passed = run_tests(code, task["tests"])
    print(f"\n{label} [{task['entry_point']}] (tests {'PASS' if passed else 'FAIL'}):")
    print(code)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default=str(ROOT / "data" / "tokenizer.json"))
    parser.add_argument(
        "--checkpoint", default=str(ROOT / "data" / "checkpoint_sft.pt")
    )
    parser.add_argument("--tasks", default=str(ROOT / "data" / "rl" / "tasks.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "data" / "checkpoint_rl.pt"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=0.02)
    args = parser.parse_args()

    for path in (args.tokenizer, args.checkpoint, args.tasks):
        if not Path(path).exists():
            raise SystemExit(f"{path} not found -- run sft.py first")

    tokenizer = BPETokenizer.load(args.tokenizer)
    model = load_checkpoint(args.checkpoint)
    ref_model = copy.deepcopy(model)
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_model.eval()

    tasks = [
        json.loads(line)
        for line in Path(args.tasks).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"loaded {len(tasks)} RL tasks")

    demo_task = tasks[0]
    show_sample(model, tokenizer, demo_task, "before GRPO")

    grpo_loop(
        model,
        tokenizer,
        tasks,
        reward_fn=code_golf_reward,
        steps=args.steps,
        group_size=args.group_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        lr=args.lr,
        clip_eps=args.clip_eps,
        ref_model=ref_model,
        kl_coef=args.kl_coef,
    )

    show_sample(model, tokenizer, demo_task, "after GRPO")

    save_checkpoint(model, args.out, model.config)
    print(f"\nsaved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
