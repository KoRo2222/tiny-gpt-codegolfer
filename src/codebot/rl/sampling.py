from __future__ import annotations

import torch

from ..model import TinyGPT
from ..tokenizer import BPETokenizer


@torch.no_grad()
def sample_completion(
    model: TinyGPT,
    tokenizer: BPETokenizer,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
) -> tuple[list[int], list[float]]:
    """Sample one completion, recording each sampled token's log-prob.

    Those log-probs are GRPO's "old policy" reference for the importance
    ratio exp(new_logp - old_logp): it's exactly 1.0 the instant this
    returns, and only drifts once the model has since been updated.
    Stops early on <|endoftext|>, same as training expects a response to
    end.
    """
    was_training = model.training
    model.eval()
    max_seq_len = model.embedding.position_embedding.num_embeddings
    eot_id = tokenizer.special_tokens[tokenizer.EOT_TOKEN]

    token_ids = torch.tensor([prompt_ids])
    response_ids: list[int] = []
    log_probs: list[float] = []

    for _ in range(max_new_tokens):
        context = token_ids[:, -max_seq_len:]
        logits = model(context)[:, -1, :]

        if temperature <= 1e-8:
            # Greedy decoding: deterministic, for reproducible eval runs.
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        # log-prob under whichever distribution actually produced next_id
        # -- GRPO's importance ratio is only meaningful if this matches
        # the draw, temperature-scaled or not.
        log_probs.append(torch.log(probs[0, next_id.item()] + 1e-12).item())
        response_ids.append(next_id.item())
        token_ids = torch.cat([token_ids, next_id], dim=1)

        if next_id.item() == eot_id:
            break

    model.train(was_training)
    return response_ids, log_probs
