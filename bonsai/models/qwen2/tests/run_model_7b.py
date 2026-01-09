import os
import time

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
    # Enable JAX memory optimizations
    import os
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.90'

    model_ckpt_path = os.path.expanduser("~/.cache/modelscope/hub/models/Qwen/Qwen2-7B")

    # Disable sharding - run on single GPU
    config = modeling.ModelConfig.qwen2_7b(use_sharding=False)
    mesh, batch_shd = None, None

    print(f"Running without sharding (single GPU mode)")
    print(f"Available devices: {jax.device_count()}")
    print(f"Using device: {jax.devices()[0]}\n")

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

    print()
    tokens = tokenize(tokenizer, query, batch_shd)
    batch_size, token_len = tokens.shape

    # Set max generation steps - reduce to save memory
    generate_steps = 256
    print(f"\nLoading model...")
    model = params.create_model_from_safe_tensors(model_ckpt_path, config, mesh)
    print(f"Model loaded successfully\n")

    cache = model.init_cache(config, batch_size, token_len, generate_steps)

    print(f"Memory settings:")
    print(f"  Batch size: {batch_size}")
    print(f"  Input token length: {token_len}")
    print(f"  Max generation steps: {generate_steps}")
    print(f"  Total cache length: {token_len + generate_steps}\n")

    key = jax.random.key(0)
    # Use Sampler with lower temperature
    sampler = Sampler(temperature=0.7, top_p=0.9, top_k=20)
    jit_sampler = jax.jit(sampler)

    print(f"Using sampler: {sampler.__class__.__name__}")
    print(f"Starting generation...\n")

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

        # Only check for actual EOS token
        is_eos = (next_tokens.squeeze(-1) == tokenizer.eos_token_id)
        finished = finished | is_eos

        tokens_list.append(next_tokens)

        # Check if generation is finished
        if finished.all():
            print(f"✓ Generation stopped at step {i+1}/{generate_steps} (EOS token reached)")
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