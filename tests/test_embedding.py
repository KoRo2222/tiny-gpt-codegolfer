import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from codebot.model import TokenPositionalEmbedding


def test_output_shape():
    embedding = TokenPositionalEmbedding(vocab_size=50, d_model=16, max_seq_len=32)
    token_ids = torch.randint(0, 50, (2, 5))

    out = embedding(token_ids)

    assert out.shape == (2, 5, 16)


def test_same_token_gets_different_vectors_at_different_positions():
    embedding = TokenPositionalEmbedding(vocab_size=50, d_model=16, max_seq_len=32)
    token_ids = torch.tensor([[7, 7, 7]])

    out = embedding(token_ids)

    assert not torch.allclose(out[0, 0], out[0, 1])
    assert not torch.allclose(out[0, 1], out[0, 2])


def test_matches_token_plus_position_embedding_directly():
    embedding = TokenPositionalEmbedding(vocab_size=50, d_model=16, max_seq_len=32)
    token_ids = torch.tensor([[3, 9]])

    out = embedding(token_ids)

    expected = embedding.token_embedding(token_ids) + embedding.position_embedding(
        torch.arange(2)
    )
    assert torch.allclose(out, expected)
