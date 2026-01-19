import os
import jax
import jax.numpy as jnp
import numpy as np
from jax import P
from jax._src.mesh import AxisType
from transformers import AutoTokenizer

from bonsai.models.qwen2 import modeling, params
from bonsai.utils import GreedySampler, Sampler


def tokenize(tokenizer, input: list[str], shd: P | None = None):
    pad_idx = tokenizer.pad_token_id
    lines = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": l}], tokenize=False, add_generation_prompt=True
        )
        for l in input
    ]
    lines = [tokenizer.encode(line) for line in lines]
    max_len = max(len(line) for line in lines)  # Right-align, left-padding to the max token length.
    return jnp.array([np.pad(l, (max_len - len(l), 0), constant_values=pad_idx) for l in lines], out_sharding=shd)


def run_model():
    model_ckpt_path = os.path.expanduser("~/.cache/modelscope/hub/models/Qwen/Qwen2-0.5B")
    # config = modeling.ModelConfig.qwen2_0_5b(use_sharding=False)
    # mesh, batch_shd = None, None

    # Enable sharding below if you have multiple devices.
    # model_ckpt_path = snapshot_download("Qwen/Qwen2-7B")
    # config = modeling.ModelConfig.qwen2_7b(use_sharding=True)
    config = modeling.ModelConfig.qwen2_0_5b(use_sharding=True)
    mesh = jax.make_mesh((1, 2), ("fsdp", "tp"), axis_types=(AxisType.Explicit, AxisType.Explicit))
    batch_shd = P("fsdp", None)
    jax.set_mesh(mesh)

    query = [
        "天空为什么是蓝色的？",
    ]

    # Load tokenizer from local path
    # Use the absolute path directly to avoid any model hub lookups
    tokenizer = AutoTokenizer.from_pretrained(
        model_ckpt_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    # Debug: Check tokenizer special tokens
    print(f"Tokenizer special tokens:")
    print(f"  eos_token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"  pad_token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"  bos_token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    if hasattr(tokenizer, 'im_end_id'):
        print(f"  im_end_id: {tokenizer.im_end_id}")
    if hasattr(tokenizer, 'im_start_id'):
        print(f"  im_start_id: {tokenizer.im_start_id}")

    # Try to find the correct EOS token
    # For Qwen2, the chat format uses <|im_end|> as the end token
    im_end_token_id = None
    if hasattr(tokenizer, 'im_end_id'):
        im_end_token_id = tokenizer.im_end_id
    else:
        # Try to encode the token
        try:
            im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
            if im_end_ids:
                im_end_token_id = im_end_ids[0]
                print(f"  <|im_end|> token ID: {im_end_token_id}")
        except:
            pass

    print()
    tokens = tokenize(tokenizer, query, batch_shd)
    batch_size, token_len = tokens.shape

    # Set max generation steps - will stop early if EOS token is generated
    generate_steps = 1024
    model = params.create_model_from_safe_tensors(model_ckpt_path, config, mesh)
    cache = model.init_cache(config, batch_size, token_len, generate_steps)

    key = jax.random.key(0)
    # Option 1: Use Sampler with lower temperature
    sampler = Sampler(temperature=0.7, top_p=0.9, top_k=20)

    # Option 2: Use GreedySampler for deterministic generation
    # sampler = GreedySampler()

    jit_sampler = jax.jit(sampler)

    print(f"Using sampler: {sampler.__class__.__name__}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")
    if im_end_token_id is not None:
        print(f"IM_END token ID: {im_end_token_id} (NOT used for stopping - only EOS)")
    print()

    # prefill - only initialize cache, don't start decoding yet
    logits, cache = modeling.forward(model, cache, tokens, tokenizer.pad_token_id)

    # decode
    tokens_list = []
    finished = jnp.zeros((batch_size,), dtype=jnp.bool_)

    for i in range(generate_steps):
        # CRITICAL: Split key for each step to avoid deterministic sampling
        key, subkey = jax.random.split(key)
        next_tokens = jit_sampler(logits, key=subkey)

        current_token_id = int(next_tokens.squeeze(-1)[0])

        # Only check for actual EOS token (NOT im_end)
        is_eos = (next_tokens.squeeze(-1) == tokenizer.eos_token_id)
        finished = finished | is_eos

        tokens_list.append(next_tokens)

        # Debug: print every 50 steps
        if (i + 1) % 50 == 0:
            eos_flag = " [EOS]" if current_token_id == tokenizer.eos_token_id else ""
            print(f"Step {i+1}: Token {current_token_id}{eos_flag}")

        # Check if generation is finished
        if finished.all():
            print(f"✓ Generation stopped at step {i+1}/{generate_steps} (EOS token {current_token_id} reached)")
            break

        # Continue generation
        logits, cache = modeling.forward(model, cache, next_tokens, tokenizer.pad_token_id)

    all_output_tokens = jax.device_get(jnp.concatenate(tokens_list, axis=-1))
    for i, q in enumerate(query):
        print(f"User:\n {q}")
        seq_tokens = all_output_tokens[i]
        eos_idx = np.where(seq_tokens == tokenizer.eos_token_id)[0]
        if eos_idx.size > 0:
            seq_tokens = seq_tokens[: eos_idx[0]]
        decoded = tokenizer.decode(seq_tokens, skip_special_tokens=True)
        print(f"Answer ({len(seq_tokens)} tokens):\n {decoded}\n\n")


if __name__ == "__main__":
    run_model()


__all__ = ["run_model"]