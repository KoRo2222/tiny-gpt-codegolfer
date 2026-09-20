import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from codebot.model import TinyGPT
from codebot.tokenizer import BPETokenizer
from codebot.rl import evaluate_model, evaluate_task

CORPUS = ["def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n"]


def make_tokenizer():
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=280)
    return tok


def make_tiny_model(vocab_size):
    return TinyGPT(
        vocab_size=vocab_size, d_model=8, n_heads=2, d_ff=16, n_layers=2, max_seq_len=64
    )


def make_fixed_sampler(tokenizer, response_text):
    response_ids = tokenizer.encode(response_text)
    response_ids.append(tokenizer.special_tokens[tokenizer.EOT_TOKEN])

    def fixed_sample(model, tok, prompt_ids, max_new_tokens, temperature):
        return response_ids, [0.0] * len(response_ids)

    return fixed_sample


def test_evaluate_task_scores_a_correct_completion_as_passing():
    tok = make_tokenizer()
    model = make_tiny_model(len(tok.vocab))
    task = {
        "prompt": "# gcd\n",
        "entry_point": "gcd",
        "tests": ["assert gcd(12, 18) == 6"],
    }
    correct_code = "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n"

    result = evaluate_task(
        model, tok, task, num_samples=1, sample_fn=make_fixed_sampler(tok, correct_code)
    )

    assert result["pass_rate"] == 1.0
    assert result["avg_length"] == len(correct_code)


def test_evaluate_task_scores_a_wrong_completion_as_failing():
    tok = make_tokenizer()
    model = make_tiny_model(len(tok.vocab))
    task = {
        "prompt": "# gcd\n",
        "entry_point": "gcd",
        "tests": ["assert gcd(12, 18) == 6"],
    }
    wrong_code = "def gcd(a, b):\n    return a + b\n"

    result = evaluate_task(
        model, tok, task, num_samples=1, sample_fn=make_fixed_sampler(tok, wrong_code)
    )

    assert result["pass_rate"] == 0.0
    assert result["avg_length"] is None


def test_evaluate_model_aggregates_across_tasks():
    tok = make_tokenizer()
    model = make_tiny_model(len(tok.vocab))
    correct_code = "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n"
    wrong_code = "def gcd(a, b):\n    return a + b\n"

    tasks = [
        {"prompt": "# gcd\n", "entry_point": "gcd_ok", "tests": ["assert gcd(12, 18) == 6"]},
        {"prompt": "# gcd\n", "entry_point": "gcd_bad", "tests": ["assert gcd(12, 18) == 6"]},
    ]

    # Alternate between a correct and a wrong sampler per call.
    calls = {"n": 0}

    def alternating_sample(model, tok, prompt_ids, max_new_tokens, temperature):
        calls["n"] += 1
        code = correct_code if calls["n"] % 2 == 1 else wrong_code
        ids = tok.encode(code) + [tok.special_tokens[tok.EOT_TOKEN]]
        return ids, [0.0] * len(ids)

    summary = evaluate_model(model, tok, tasks, num_samples=1, sample_fn=alternating_sample)

    assert summary["num_tasks"] == 2
    assert summary["solved"] == 1
    assert summary["mean_pass_rate"] == 0.5


def test_evaluate_task_pass_rate_reflects_multiple_samples():
    tok = make_tokenizer()
    model = make_tiny_model(len(tok.vocab))
    task = {
        "prompt": "# gcd\n",
        "entry_point": "gcd",
        "tests": ["assert gcd(12, 18) == 6"],
    }
    correct_code = "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n"
    wrong_code = "def gcd(a, b):\n    return a + b\n"

    calls = {"n": 0}

    def half_correct_sample(model, tok, prompt_ids, max_new_tokens, temperature):
        calls["n"] += 1
        code = correct_code if calls["n"] % 2 == 1 else wrong_code
        ids = tok.encode(code) + [tok.special_tokens[tok.EOT_TOKEN]]
        return ids, [0.0] * len(ids)

    result = evaluate_task(
        model, tok, task, num_samples=4, sample_fn=half_correct_sample
    )

    assert result["pass_rate"] == 0.5
