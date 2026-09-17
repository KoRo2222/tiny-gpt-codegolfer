from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from ..model import TinyGPT
from ..tokenizer import BPETokenizer
from .sampling import sample_completion


def group_relative_advantages(rewards: list[float], eps: float = 1e-4) -> list[float]:
    """Normalize each reward against its own group's mean and spread.

    This is GRPO's central trick versus PPO: instead of a learned value
    network as the baseline, the baseline is just "how did this
    completion do relative to its G siblings sampled from the same
    prompt" -- free to compute, and automatically scaled to how hard or
    variable that particular prompt happens to be.
    """
    n = len(rewards)
    mean = sum(rewards) / n
    variance = sum((r - mean) ** 2 for r in rewards) / n
    std = variance**0.5
    return [(r - mean) / (std + eps) for r in rewards]


def response_log_probs(
    model: TinyGPT, prompt_ids: list[int], response_ids: list[int]
) -> Tensor:
    """Per-token log-probs the current model assigns the sampled response.

    Same shift-by-one + region-selection idea as SFT's build_example,
    but returning per-token log-probs (with grad) instead of a scalar
    loss -- GRPO needs them individually for the ratio and KL terms.
    """
    ids = prompt_ids + response_ids
    x = torch.tensor([ids[:-1]])
    targets = torch.tensor(ids[1:])

    logits = model(x)[0]
    log_probs_all = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs_all.gather(1, targets.unsqueeze(1)).squeeze(1)

    start = len(prompt_ids) - 1
    return token_log_probs[start:]


def grpo_loss(
    model: TinyGPT,
    prompt_ids: list[int],
    response_ids: list[int],
    old_log_probs: list[float],
    advantage: float,
    clip_eps: float = 0.2,
    ref_model: TinyGPT | None = None,
    kl_coef: float = 0.0,
) -> Tensor:
    """Clipped surrogate loss for one sampled completion, GRPO-style.

    ratio > 1 means the current policy now likes this response more than
    it did at sampling time; clipping keeps a single update from moving
    too far on the strength of one (possibly lucky) sample. An optional
    KL term against a frozen reference model discourages drifting so far
    that the model stops looking like the SFT policy it started from.
    """
    new_log_probs = response_log_probs(model, prompt_ids, response_ids)
    old = torch.tensor(old_log_probs)
    ratio = torch.exp(new_log_probs - old)

    adv = torch.full_like(ratio, advantage)
    surrogate = torch.minimum(
        ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    )
    per_token = surrogate

    if ref_model is not None and kl_coef > 0:
        with torch.no_grad():
            ref_log_probs = response_log_probs(ref_model, prompt_ids, response_ids)
        # Schulman's k3 estimator: unbiased and, unlike the naive
        # log-ratio, always >= 0 -- so it behaves like a real divergence.
        log_ratio = ref_log_probs - new_log_probs
        kl = torch.exp(log_ratio) - log_ratio - 1
        per_token = per_token - kl_coef * kl

    return -per_token.mean()


def grpo_step(
    model: TinyGPT,
    optimizer: torch.optim.Optimizer,
    tokenizer: BPETokenizer,
    task: dict,
    reward_fn: Callable[[str, list[str]], float],
    group_size: int = 8,
    max_new_tokens: int = 60,
    temperature: float = 1.0,
    clip_eps: float = 0.2,
    ref_model: TinyGPT | None = None,
    kl_coef: float = 0.0,
) -> dict:
    prompt_ids = tokenizer.encode(task["prompt"])
    # Keep prompt + response inside the model's positional embedding
    # table: sampling itself clips its context window per step, but the
    # loss's single forward pass over the whole sequence can't.
    max_seq_len = model.embedding.position_embedding.num_embeddings
    max_new_tokens = min(max_new_tokens, max(1, max_seq_len - len(prompt_ids) - 1))

    samples = [
        sample_completion(model, tokenizer, prompt_ids, max_new_tokens, temperature)
        for _ in range(group_size)
    ]

    eot_id = tokenizer.special_tokens[tokenizer.EOT_TOKEN]
    rewards = []
    for response_ids, _ in samples:
        code_ids = [t for t in response_ids if t != eot_id]
        code = tokenizer.decode(code_ids)
        rewards.append(reward_fn(code, task["tests"]))

    advantages = group_relative_advantages(rewards)

    if all(a == 0.0 for a in advantages):
        # Every completion scored identically (often all failing) --
        # nothing in this group distinguishes a better response from a
        # worse one, so there's no gradient signal worth taking.
        return {
            "mean_reward": sum(rewards) / len(rewards),
            "loss": 0.0,
            "skipped": True,
        }

    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for (response_ids, old_log_probs), advantage in zip(samples, advantages):
        loss = grpo_loss(
            model,
            prompt_ids,
            response_ids,
            old_log_probs,
            advantage,
            clip_eps=clip_eps,
            ref_model=ref_model,
            kl_coef=kl_coef,
        )
        (loss / group_size).backward()
        total_loss += loss.item()
    optimizer.step()

    return {
        "mean_reward": sum(rewards) / len(rewards),
        "loss": total_loss / group_size,
        "skipped": False,
    }


def grpo_loop(
    model: TinyGPT,
    tokenizer: BPETokenizer,
    tasks: list[dict],
    reward_fn: Callable[[str, list[str]], float],
    steps: int,
    group_size: int = 8,
    max_new_tokens: int = 60,
    temperature: float = 1.0,
    lr: float = 1e-5,
    clip_eps: float = 0.2,
    ref_model: TinyGPT | None = None,
    kl_coef: float = 0.0,
    log: Callable[[str], None] = print,
) -> list[dict]:
    """Cycle through tasks, one GRPO group-update per step."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history = []
    for step in range(1, steps + 1):
        task = tasks[(step - 1) % len(tasks)]
        stats = grpo_step(
            model,
            optimizer,
            tokenizer,
            task,
            reward_fn,
            group_size=group_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            clip_eps=clip_eps,
            ref_model=ref_model,
            kl_coef=kl_coef,
        )
        stats["step"] = step
        stats["task"] = task.get("entry_point", "?")
        history.append(stats)

        flag = " (skipped)" if stats["skipped"] else ""
        log(
            f"step {step}/{steps} [{stats['task']}]: "
            f"mean_reward={stats['mean_reward']:.3f} loss={stats['loss']:.4f}{flag}"
        )

    return history
