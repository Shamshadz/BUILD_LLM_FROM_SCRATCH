import numpy as np
import torch


def assign(left, right):
    """Convert a NumPy checkpoint array to a parameter after checking its shape."""
    if left.shape != right.shape:
        raise ValueError(
            f"Shape mismatch. Left: {left.shape}, Right: {right.shape}"
        )
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    """Load GPT-2 checkpoint arrays into a Chapter 4 GPTModel."""
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for block_idx, block_params in enumerate(params["blocks"]):
        attention = gpt.trf_blocks[block_idx].att
        q_w, k_w, v_w = np.split(
            block_params["attn"]["c_attn"]["w"], 3, axis=-1
        )
        attention.W_query.weight = assign(attention.W_query.weight, q_w.T)
        attention.W_key.weight = assign(attention.W_key.weight, k_w.T)
        attention.W_value.weight = assign(attention.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            block_params["attn"]["c_attn"]["b"], 3, axis=-1
        )
        attention.W_query.bias = assign(attention.W_query.bias, q_b)
        attention.W_key.bias = assign(attention.W_key.bias, k_b)
        attention.W_value.bias = assign(attention.W_value.bias, v_b)

        attention.out_proj.weight = assign(
            attention.out_proj.weight,
            block_params["attn"]["c_proj"]["w"].T,
        )
        attention.out_proj.bias = assign(
            attention.out_proj.bias, block_params["attn"]["c_proj"]["b"]
        )

        feed_forward = gpt.trf_blocks[block_idx].ff.layers
        feed_forward[0].weight = assign(
            feed_forward[0].weight, block_params["mlp"]["c_fc"]["w"].T
        )
        feed_forward[0].bias = assign(
            feed_forward[0].bias, block_params["mlp"]["c_fc"]["b"]
        )
        feed_forward[2].weight = assign(
            feed_forward[2].weight, block_params["mlp"]["c_proj"]["w"].T
        )
        feed_forward[2].bias = assign(
            feed_forward[2].bias, block_params["mlp"]["c_proj"]["b"]
        )

        norm1 = gpt.trf_blocks[block_idx].norm1
        norm2 = gpt.trf_blocks[block_idx].norm2
        norm1.scale = assign(norm1.scale, block_params["ln_1"]["g"])
        norm1.shift = assign(norm1.shift, block_params["ln_1"]["b"])
        norm2.scale = assign(norm2.scale, block_params["ln_2"]["g"])
        norm2.shift = assign(norm2.shift, block_params["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    # GPT-2 ties the language-model head weights to token embeddings.
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
