import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from codebot.model import TinyGPT


def make_model():
    return TinyGPT(vocab_size=50, d_model=16, n_heads=4, d_ff=32, max_seq_len=32)


def test_forward_output_shape():
    model = make_model()
    token_ids = torch.randint(0, 50, (2, 6))

    logits = model(token_ids)

    assert logits.shape == (2, 6, 50)


def test_next_token_probs_is_a_probability_distribution():
    model = make_model()
    token_ids = torch.randint(0, 50, (2, 6))

    probs = model.next_token_probs(token_ids)

    assert probs.shape == (2, 50)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-6)
    assert (probs >= 0).all()


def test_logits_at_a_position_are_unaffected_by_later_tokens():
    # The whole point of the causal mask: position i's logits must only
    # depend on tokens <= i. Changing a future token must not change it.
    model = make_model()
    model.eval()
    token_ids = torch.randint(0, 50, (1, 6))

    logits_a = model(token_ids)

    changed = token_ids.clone()
    changed[0, -1] = (changed[0, -1] + 1) % 50
    logits_b = model(changed)

    assert torch.allclose(logits_a[0, :-1], logits_b[0, :-1], atol=1e-5)


def test_gradients_flow_to_embedding():
    model = make_model()
    token_ids = torch.randint(0, 50, (2, 6))

    logits = model(token_ids)
    logits.sum().backward()

    assert model.embedding.token_embedding.weight.grad is not None
