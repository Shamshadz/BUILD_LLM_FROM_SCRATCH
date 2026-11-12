import torch


def text_to_token_ids(text, tokenizer):
    """Encode text into a batch of token IDs, allowing GPT-2's end marker."""
    token_ids = tokenizer.encode(
        text, allowed_special={"<|endoftext|>"}
    )
    return torch.tensor(token_ids).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    """Decode token IDs from a single sequence or a batch of one."""
    token_ids = token_ids.squeeze(0)
    return tokenizer.decode(token_ids.tolist())


def softmax_with_temperature(logits, temperature):
    """Convert logits to probabilities after applying temperature scaling."""
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    return torch.softmax(logits / temperature, dim=-1)


def generate(
    model,
    idx,
    max_new_tokens,
    context_size,
    temperature=0.0,
    top_k=None,
    eos_id=None,
):
    """Generate tokens greedily or by sampling, optionally using top-k and EOS."""
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive")

    finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)

    for _ in range(max_new_tokens):
        # Limit the prompt to the model's available context window.
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)[:, -1, :]

        if top_k is not None:
            # Mask logits below the kth-largest value before sampling.
            k = min(top_k, logits.size(-1))
            top_logits = torch.topk(logits, k, dim=-1).values
            min_val = top_logits[:, -1, None]
            logits = logits.masked_fill(logits < min_val, -torch.inf)

        if temperature > 0:
            # Temperature changes the probability distribution used for sampling.
            probs = torch.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            # At zero temperature, choose the highest-logit token deterministically.
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None:
            # Keep completed sequences at EOS while other batch members continue.
            idx_next = torch.where(
                finished[:, None],
                torch.full_like(idx_next, eos_id),
                idx_next,
            )

        idx = torch.cat((idx, idx_next), dim=1)

        if eos_id is not None:
            finished |= idx_next.squeeze(1).eq(eos_id)
            if finished.all():
                break

    return idx
