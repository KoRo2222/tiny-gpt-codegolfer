"""Generate data/corpus, data/sft/examples.jsonl and data/rl/tasks.jsonl
from the single task catalog (src/codebot/data/task_catalog.py).

Usage:
    python scripts/build_datasets.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from codebot.data.task_catalog import TASKS  # noqa: E402
from codebot.rl.reward import run_tests  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    # Self-check: a task with tests must actually pass them. Catches an
    # authoring mistake here, loudly, instead of it silently poisoning
    # the pretraining/SFT/RL data.
    failures = [
        task["entry_point"]
        for task in TASKS
        if task["tests"] and not run_tests(task["code"], task["tests"])
    ]
    if failures:
        raise SystemExit(f"tasks failed their own tests: {failures}")

    corpus_path = ROOT / "data" / "corpus" / "python_snippets.txt"
    corpus_path.write_text(
        "\n".join(task["code"] for task in TASKS), encoding="utf-8"
    )

    sft_path = ROOT / "data" / "sft" / "examples.jsonl"
    sft_lines = [
        json.dumps({"prompt": task["instruction"], "response": task["code"]})
        for task in TASKS
    ]
    sft_path.write_text("\n".join(sft_lines) + "\n", encoding="utf-8")

    rl_path = ROOT / "data" / "rl" / "tasks.jsonl"
    rl_lines = [
        json.dumps(
            {
                "prompt": task["instruction"],
                "entry_point": task["entry_point"],
                "tests": task["tests"],
            }
        )
        for task in TASKS
        if task["tests"]
    ]
    rl_path.write_text("\n".join(rl_lines) + "\n", encoding="utf-8")

    print(f"catalog: {len(TASKS)} functions/classes")
    print(f"corpus  -> {corpus_path}")
    print(f"sft     -> {sft_path} ({len(sft_lines)} examples)")
    print(f"rl      -> {rl_path} ({len(rl_lines)} tasks)")


if __name__ == "__main__":
    main()
