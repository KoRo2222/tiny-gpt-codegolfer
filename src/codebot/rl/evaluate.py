from __future__ import annotations

from typing import Callable

from ..model import TinyGPT
from ..tokenizer import BPETokenizer
from .reward import run_tests
from .sampling import sample_completion

SampleFn = Callable[
    [TinyGPT, BPETokenizer, list[int], int, float], tuple[list[int], list[float]]
]


def evaluate_task(
    model: TinyGPT,
    tokenizer: BPETokenizer,
    task: dict,
    num_samples: int = 1,
    temperature: float = 0.0,
    max_new_tokens: int = 60,
    sample_fn: SampleFn = sample_completion,
) -> dict:
    """pass@num_samples and average length of the passing completions.

    sample_fn is injectable so tests can check the scoring logic against
    a fixed, fake completion instead of a real (stochastic) model.
    """
    prompt_ids = tokenizer.encode(task["prompt"])
    eot_id = tokenizer.special_tokens[tokenizer.EOT_TOKEN]

    passed = 0
    passing_lengths = []
    for _ in range(num_samples):
        response_ids, _ = sample_fn(
            model, tokenizer, prompt_ids, max_new_tokens, temperature
        )
        code = tokenizer.decode([t for t in response_ids if t != eot_id])
        if run_tests(code, task["tests"]):
            passed += 1
            passing_lengths.append(len(code))

    return {
        "entry_point": task["entry_point"],
        "pass_rate": passed / num_samples,
        "avg_length": (
            sum(passing_lengths) / len(passing_lengths) if passing_lengths else None
        ),
    }


def evaluate_model(
    model: TinyGPT,
    tokenizer: BPETokenizer,
    tasks: list[dict],
    num_samples: int = 1,
    temperature: float = 0.0,
    max_new_tokens: int = 60,
    sample_fn: SampleFn = sample_completion,
) -> dict:
    """Score a model against every task, plus an aggregate summary.

    This is the scalar this project has been missing: "solved N/M tasks"
    tracked run over run, instead of eyeballing one before/after sample.
    """
    results = [
        evaluate_task(
            model, tokenizer, task, num_samples, temperature, max_new_tokens, sample_fn
        )
        for task in tasks
    ]
    solved = sum(1 for r in results if r["pass_rate"] > 0)
    mean_pass_rate = sum(r["pass_rate"] for r in results) / len(results)

    return {
        "num_tasks": len(tasks),
        "solved": solved,
        "mean_pass_rate": mean_pass_rate,
        "results": results,
    }
