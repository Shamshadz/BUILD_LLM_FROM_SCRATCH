import torch

from .generation import text_to_token_ids, token_ids_to_text
from ch04 import generate_text_simple


def calc_loss_batch(input_batch, target_batch, model, device):
    """Compute next-token cross-entropy loss for one batch."""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Compute mean loss over the loader or its first ``num_batches`` batches."""
    total_batches = len(data_loader)
    if total_batches == 0:
        return float("nan")

    if num_batches is None:
        num_batches = total_batches
    else:
        num_batches = min(num_batches, total_batches)

    total_loss = 0.0
    for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
        if batch_idx >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()

    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """Measure train and validation loss without computing gradients."""
    model.eval()
    try:
        with torch.no_grad():
            train_loss = calc_loss_loader(
                train_loader, model, device, num_batches=eval_iter
            )
            val_loss = calc_loss_loader(
                val_loader, model, device, num_batches=eval_iter
            )
    finally:
        model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    """Generate and print a short sample while restoring training mode."""
    was_training = model.training
    model.eval()
    try:
        context_size = model.pos_emb.weight.shape[0]
        encoded = text_to_token_ids(start_context, tokenizer).to(device)
        with torch.no_grad():
            token_ids = generate_text_simple(
                model=model,
                idx=encoded,
                max_new_tokens=50,
                context_size=context_size,
            )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))
    finally:
        model.train(was_training)


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
):
    """Train a model and periodically record training/validation losses."""
    if eval_freq <= 0:
        raise ValueError("eval_freq must be greater than zero")

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen
