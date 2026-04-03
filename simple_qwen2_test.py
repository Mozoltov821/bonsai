#!/usr/bin/env python3
"""简化的Qwen2测试脚本，用于调试乱码问题"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from bonsai.models.qwen2 import modeling, params
from bonsai.utils import GreedySampler

def simple_test():
    print("=== 简化Qwen2测试 ===")

    # 加载模型和tokenizer
    model_path = snapshot_download("Qwen/Qwen2-0.5B")
    config = modeling.ModelConfig.qwen2_0_5b(use_sharding=False)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = params.create_model_from_safe_tensors(model_path, config)

    print(f"Model vocab_size: {config.vocab_size}")
    print(f"Tokenizer vocab_size: {tokenizer.vocab_size}")

    # 简单文本测试（不用chat template）
    test_text = "Hello, how are you?"
    print(f"Input text: {test_text}")

    # 直接tokenize
    tokens = tokenizer.encode(test_text, return_tensors="pt")
    tokens = jnp.array(tokens.numpy())
    print(f"Input tokens: {tokens}")

    # 准备输入
    batch_size = 1
    token_len = tokens.shape[-1]
    tokens = tokens.reshape(batch_size, token_len)

    # 初始化cache
    generate_steps = 20
    cache = model.init_cache(config, batch_size, token_len, generate_steps)

    # 使用greedy sampling
    sampler = GreedySampler()

    # 第一步：prefill
    print("\\n=== Prefill ===")
    logits, cache = modeling.forward(model, cache, tokens, tokenizer.pad_token_id, tokenizer.vocab_size)
    print(f"Prefill logits shape: {logits.shape}")
    print(f"Prefill logits range: [{float(logits.min()):.3f}, {float(logits.max()):.3f}]")

    # 检查logits中的最大值对应的token
    next_token_id = jnp.argmax(logits, axis=-1)
    print(f"Next token ID: {next_token_id}")

    if next_token_id < tokenizer.vocab_size:
        next_token_text = tokenizer.decode([int(next_token_id[0])])
        print(f"Next token text: {repr(next_token_text)}")
    else:
        print(f"ERROR: Token ID {next_token_id} >= vocab_size {tokenizer.vocab_size}")

    next_tokens = sampler(logits, key=jax.random.key(0))
    print(f"Sampled next tokens: {next_tokens}")

    # 生成序列
    print("\\n=== Generation ===")
    tokens_list = [next_tokens]

    for i in range(min(generate_steps, 10)):  # 只生成10个token进行调试
        logits, cache = modeling.forward(model, cache, next_tokens, tokenizer.pad_token_id, tokenizer.vocab_size)
        next_tokens = sampler(logits, key=jax.random.key(i))
        tokens_list.append(next_tokens)

        token_id = int(next_tokens[0, 0])
        print(f"Step {i+1}: token_id={token_id}", end="")

        if token_id < tokenizer.vocab_size:
            token_text = tokenizer.decode([token_id])
            print(f" -> {repr(token_text)}")
        else:
            print(f" -> ERROR: ID >= vocab_size")
            break

        if token_id == tokenizer.eos_token_id:
            print("Hit EOS token")
            break

    # 解码完整输出
    all_output_tokens = jax.device_get(jnp.concatenate(tokens_list, axis=-1))
    generated_sequence = all_output_tokens[0]

    print(f"\\n=== Final Results ===")
    print(f"Generated tokens: {generated_sequence}")

    # 检查是否有无效token
    invalid_tokens = generated_sequence >= tokenizer.vocab_size
    if jnp.any(invalid_tokens):
        print(f"WARNING: Found {jnp.sum(invalid_tokens)} invalid tokens!")
        valid_tokens = generated_sequence[~invalid_tokens]
    else:
        valid_tokens = generated_sequence

    decoded = tokenizer.decode(valid_tokens, skip_special_tokens=True)
    print(f"Decoded text: {repr(decoded)}")

    return decoded

if __name__ == "__main__":
    simple_test()