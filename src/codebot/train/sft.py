from __future__ import annotations

import random
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from ..model import TinyGPT
from ..tokenizer import BPETokenizer

IGNORE_INDEX = -100


def build_example(
    tokenizer: BPETokenizer, prompt: str, response: str
) -> tuple[list[int], list[int]]:
    """Turn one (prompt, response) pair into (input_ids, labels).

    Pretraining scores every position equally, but here we only want the
    model penalized for getting the *response* wrong -- it shouldn't be
    trained to predict the instruction text itself. labels holds the
    usual next-token target, except positions whose target token falls
    inside the prompt are set to IGNORE_INDEX, which F.cross_entropy
    skips.
    """
    prompt_ids = tokenizer.encode(prompt)
    response_ids = tokenizer.encode(response)
    eot_id = tokenizer.special_tokens[tokenizer.EOT_TOKEN]
    ids = prompt_ids + response_ids + [eot_id]

    input_ids = ids[:-1]
    targets = ids[1:]
    # targets[i] is the token at position i+1 in `ids`; only score it if
    # that position falls at or after the response's first token.
    labels = [
        target if (i + 1) >= len(prompt_ids) else IGNORE_INDEX
        for i, target in enumerate(targets)
    ]
    return input_ids, labels


def sft_example_loss(
    model: TinyGPT, input_ids: list[int], labels: list[int], device: str = "cpu"
) -> Tensor:
    x = torch.tensor([input_ids], device=device)
    y = torch.tensor([labels], device=device)
    logits = model(x)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=IGNORE_INDEX
    )


def sft_loop(
    model: TinyGPT,
    tokenizer: BPETokenizer,
    examples: list[dict],
    epochs: int = 5,
    lr: float = 1e-4,
    device: str = "cpu",
    log: Callable[[str], None] = print,
) -> list[dict]:
    """Fine-tune on prompt/response pairs, one example per optimizer step.

    No padding/batching -- examples are short and few enough that batch
    size 1 keeps this simple and avoids needing an attention padding
    mask the model doesn't support yet.
    """
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    max_seq_len = model.embedding.position_embedding.num_embeddings

    history = []
    for epoch in range(1, epochs + 1):
        shuffled = examples[:]
        random.shuffle(shuffled)

        epoch_losses = []
        for example in shuffled:
            input_ids, labels = build_example(
                tokenizer, example["prompt"], example["response"]
            )
            input_ids, labels = input_ids[:max_seq_len], labels[:max_seq_len]

            loss = sft_example_loss(model, input_ids, labels, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        history.append({"epoch": epoch, "loss": avg_loss})
        log(f"epoch {epoch}/{epochs}: loss={avg_loss:.4f}")

    return history
