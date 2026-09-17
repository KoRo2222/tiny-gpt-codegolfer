import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from codebot.model import TinyGPT
from codebot.tokenizer import BPETokenizer
from codebot.rl import (
    grpo_loss,
    grpo_step,
    group_relative_advantages,
    response_log_probs,
    sample_completion,
)

CORPUS = [
    "def add(a, b):\n    return a + b\n",
    "def sub(a, b):\n    return a - b\n",
]


def make_tokenizer():
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=280)
    return tok


def make_tiny_model(vocab_size, max_seq_len=64):
    return TinyGPT(
        vocab_size=vocab_size,
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=2,
        max_seq_len=max_seq_len,
    )


def test_group_relative_advantages_are_zero_mean():
    advantages = group_relative_advantages([0.0, 1.0, 1.0, 2.0])
    assert abs(sum(advantages)) < 1e-6


def test_group_relative_advantages_rank_order_matches_rewards():
    advantages = group_relative_advantages([0.0, 5.0, 2.0])
    assert advantages[1] > advantages[2] > advantages[0]


def test_group_relative_advantages_handles_zero_variance():
    # All identical rewards: every completion is equally (un)helpful, so
    # advantages should collapse near zero, not divide-by-zero/blow up.
    advantages = group_relative_advantages([1.0, 1.0, 1.0])
    assert all(abs(a) < 1.0 for a in advantages)


def test_response_log_probs_length_matches_response():
    tok = make_tokenizer()
    model = make_tiny_model(len(tok.vocab))
    prompt_ids = tok.encode("def add(a, b):\n    ")
    response_ids = tok.encode("return a + b\n")

    log_probs = response_log_probs(model, prompt_ids, response_ids)

    assert log_probs.shape == (len(response_ids),)
    assert (log_probs <= 0).all()


def test_sample_completion_log_probs_match_response_length():
    tok = make_tokenizer()
    model = make_tiny_model(len(tok.vocab))
    prompt_ids = tok.encode("def add(a, b):\n    ")

    response_ids, log_probs = sample_completion(
        model, tok, prompt_ids, max_new_tokens=10
    )

    assert len(response_ids) == len(log_probs)
    assert 0 < len(response_ids) <= 10


def test_grpo_loss_increases_probability_of_a_positive_advantage_response():
    # The core correctness property of the whole update: after one step
    # with advantage > 0, the model must assign that exact response a
    # higher probability than before. Get the sign wrong and RL makes
    # good completions *less* likely instead of more.
    torch.manual_seed(0)
    tok = make_tokenizer()
    model = make_tiny_model(len(tok.vocab))
    prompt_ids = tok.encode("def add(a, b):\n    ")
    response_ids = tok.encode("return a + b\n")
    old_log_probs = response_log_probs(model, prompt_ids, response_ids).tolist()

    before = sum(old_log_probs)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss = grpo_loss(model, prompt_ids, response_ids, old_log_probs, advantage=1.0)
        loss.backward()
        optimizer.step()

    after = response_log_probs(model, prompt_ids, response_ids).sum().item()
    assert after > before


def test_grpo_step_skips_when_group_rewards_are_identical():
    tok = make_tokenizer()
    model = make_tiny_model(len(tok.vocab))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    task = {"prompt": "def add(a, b):\n    ", "tests": ["assert True"]}

    stats = grpo_step(
        model,
        optimizer,
        tok,
        task,
        reward_fn=lambda code, tests: 0.0,  # constant reward -> no signal
        group_size=4,
        max_new_tokens=5,
    )

    assert stats["skipped"] is True


def test_grpo_step_updates_when_rewards_differ():
    tok = make_tokenizer()
    model = make_tiny_model(len(tok.vocab))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    task = {"prompt": "def add(a, b):\n    ", "tests": ["assert True"]}

    calls = {"n": 0}

    def alternating_reward(code, tests):
        calls["n"] += 1
        return float(calls["n"] % 2)  # 0, 1, 0, 1, ...

    stats = grpo_step(
        model,
        optimizer,
        tok,
        task,
        reward_fn=alternating_reward,
        group_size=4,
        max_new_tokens=5,
    )

    assert stats["skipped"] is False
