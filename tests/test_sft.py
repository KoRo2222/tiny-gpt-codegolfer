import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from codebot.model import TinyGPT
from codebot.tokenizer import BPETokenizer
from codebot.train import build_example, sft_example_loss, sft_loop
from codebot.train.sft import IGNORE_INDEX

CORPUS = [
    "def add(a, b):\n    return a + b\n",
    "def sub(a, b):\n    return a - b\n",
]


def make_tokenizer():
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=280)
    return tok


def test_build_example_masks_the_prompt_region():
    tok = make_tokenizer()
    prompt = "def add(a, b):\n    "
    response = "return a + b\n"

    input_ids, labels = build_example(tok, prompt, response)

    prompt_len = len(tok.encode(prompt))
    # Every label predicting a token inside the prompt must be ignored.
    assert all(label == IGNORE_INDEX for label in labels[: prompt_len - 1])
    # At least one label in the response region must be a real token id.
    assert any(label != IGNORE_INDEX for label in labels[prompt_len - 1 :])


def test_build_example_last_label_is_eot():
    tok = make_tokenizer()
    input_ids, labels = build_example(tok, "def add(a, b):\n    ", "return a + b\n")

    eot_id = tok.special_tokens[tok.EOT_TOKEN]
    assert labels[-1] == eot_id


def test_build_example_input_and_labels_are_shifted_by_one():
    tok = make_tokenizer()
    prompt, response = "def add(a, b):\n    ", "return a + b\n"
    input_ids, labels = build_example(tok, prompt, response)

    eot_id = tok.special_tokens[tok.EOT_TOKEN]
    full_ids = tok.encode(prompt) + tok.encode(response) + [eot_id]
    assert input_ids == full_ids[:-1]
    assert len(labels) == len(input_ids)


def make_tiny_model(vocab_size):
    return TinyGPT(
        vocab_size=vocab_size, d_model=8, n_heads=2, d_ff=16, n_layers=2, max_seq_len=64
    )


def test_sft_example_loss_is_a_positive_scalar():
    tok = make_tokenizer()
    model = make_tiny_model(len(tok.vocab))
    input_ids, labels = build_example(tok, "def add(a, b):\n    ", "return a + b\n")

    loss = sft_example_loss(model, input_ids, labels)

    assert loss.dim() == 0
    assert loss.item() > 0


def test_sft_loop_reduces_loss_on_a_repeated_example():
    torch.manual_seed(0)
    tok = make_tokenizer()
    model = make_tiny_model(len(tok.vocab))
    examples = [{"prompt": "def add(a, b):\n    ", "response": "return a + b\n"}]

    history = sft_loop(model, tok, examples, epochs=30, lr=5e-3)

    assert history[-1]["loss"] < history[0]["loss"]
