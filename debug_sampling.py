#!/usr/bin/env python3
"""调试采样过程"""

import jax
import jax.numpy as jnp
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from bonsai.models.qwen2 import modeling, params
from bonsai.utils import GreedySampler

def debug_sampling():
    """调试采样过程中的logits"""
    print("=== 调试采样过程 ===")

    # 加载模型和tokenizer
    model_path = snapshot_download("Qwen/Qwen2-0.5B")
    config = modeling.ModelConfig.qwen2_0_5b(use_sharding=False)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = params.create_model_from_safe_tensors(model_path, config)

    print(f"Model vocab_size: {config.vocab_size}")
    print(f"Tokenizer vocab_size: {tokenizer.vocab_size}")

    # 准备输入
    test_text = "Hello, how are you?"
    tokens = tokenizer.encode(test_text, return_tensors="pt")
    tokens = jnp.array(tokens.numpy()).reshape(1, -1)
    print(f"Input tokens: {tokens}")

    # 初始化cache
    batch_size = 1
    token_len = tokens.shape[-1]
    cache = model.init_cache(config, batch_size, token_len, 10)

    # 第一步：prefill
    print(f"\n=== Prefill调试 ===")
    logits, cache = modeling.forward(model, cache, tokens, tokenizer.pad_token_id, tokenizer.vocab_size)

    print(f"Raw logits shape: {logits.shape}")
    print(f"Raw logits range: [{float(logits.min()):.3f}, {float(logits.max()):.3f}]")

    # 检查logits中的具体值
    logits_np = jax.device_get(logits[0])  # 取第一个batch
    print(f"Logits numpy shape: {logits_np.shape}")

    # 检查是否有-inf值
    inf_mask = jnp.isinf(logits_np)
    num_inf = jnp.sum(inf_mask)
    print(f"Number of -inf values: {num_inf}")
    print(f"Total logits: {len(logits_np)}")

    # 检查有效vocab range内的logits
    valid_range = logits_np[:tokenizer.vocab_size]
    invalid_range = logits_np[tokenizer.vocab_size:]

    print(f"Valid range ({tokenizer.vocab_size}): [{float(valid_range.min()):.3f}, {float(valid_range.max()):.3f}]")
    print(f"Invalid range ({len(invalid_range)}): [{float(invalid_range.min()):.3f}, {float(invalid_range.max()):.3f}]")

    # 检查invalid range是否都是-inf
    invalid_is_inf = jnp.all(jnp.isinf(invalid_range))
    print(f"All invalid tokens are -inf: {invalid_is_inf}")

    # 找出top 10 tokens
    top_indices = jnp.argsort(logits_np)[-10:][::-1]
    print(f"\nTop 10 logits:")
    for i, idx in enumerate(top_indices):
        logit_val = logits_np[idx]
        is_valid = idx < tokenizer.vocab_size
        if is_valid:
            token_text = tokenizer.decode([int(idx)])
            print(f"  {i+1}. ID {idx} (valid): '{token_text}' -> {float(logit_val):.3f}")
        else:
            print(f"  {i+1}. ID {idx} (INVALID): -> {float(logit_val):.3f}")

    # 手动测试argmax
    print(f"\n=== Argmax测试 ===")
    argmax_result = jnp.argmax(logits_np)
    print(f"Argmax result: {argmax_result}")
    print(f"Is argmax valid: {argmax_result < tokenizer.vocab_size}")

    # 使用GreedySampler
    print(f"\n=== GreedySampler测试 ===")
    sampler = GreedySampler()
    sampled = sampler(logits, key=jax.random.key(0))
    print(f"Sampled token: {sampled}")
    print(f"Is sampled valid: {sampled[0, 0] < tokenizer.vocab_size}")

    # 如果采样的token无效，找出问题
    if sampled[0, 0] >= tokenizer.vocab_size:
        print(f"\n⚠️ 采样了无效token {sampled[0, 0]}")
        print(f"对应的logit值: {logits_np[sampled[0, 0]]}")

        # 检查有效范围内最高的token
        valid_logits = logits_np[:tokenizer.vocab_size]
        valid_argmax = jnp.argmax(valid_logits)
        print(f"有效范围内最高token: {valid_argmax} -> {float(valid_logits[valid_argmax]):.3f}")
        token_text = tokenizer.decode([int(valid_argmax)])
        print(f"对应文本: '{token_text}'")

if __name__ == "__main__":
    debug_sampling()