import torch


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """Greedily append the highest-scoring next token at each generation step."""
    for _ in range(max_new_tokens):
        # Restrict the model input to the context window it can process.
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        # Select the logits for the final position and choose the top token.
        next_token_logits = logits[:, -1, :]
        idx_next = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=-1)

    return idx
