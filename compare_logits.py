#!/usr/bin/env python3
"""对比JAX和PyTorch版本的logits输出"""

import torch
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

from bonsai.models.qwen2 import modeling, params

def compare_logits():
    print("=== 对比JAX和PyTorch logits ===")

    # 准备相同的输入
    model_path = snapshot_download("Qwen/Qwen2-0.5B")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    test_text = "Hello, how are you?"
    inputs = tokenizer(test_text, return_tensors="pt")
    tokens_torch = inputs["input_ids"]
    tokens_jax = jnp.array(tokens_torch.numpy())

    print(f"Input: {test_text}")
    print(f"Tokens: {tokens_torch}")

    # PyTorch版本
    print("\\n=== PyTorch 版本 ===")
    torch_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    torch_model.eval()

    with torch.no_grad():
        torch_outputs = torch_model(tokens_torch)
        torch_logits = torch_outputs.logits[0, -1, :tokenizer.vocab_size]  # 只取有效vocab范围

    print(f"Logits shape: {torch_logits.shape}")
    print(f"Logits range: [{torch_logits.min().item():.3f}, {torch_logits.max().item():.3f}]")

    # Top 5 tokens
    torch_top_indices = torch.topk(torch_logits, 5).indices
    torch_top_values = torch.topk(torch_logits, 5).values
    print("Top 5 tokens:")
    for i, (idx, val) in enumerate(zip(torch_top_indices, torch_top_values)):
        token_text = tokenizer.decode([idx])
        print(f"  {i+1}. ID {idx}: {repr(token_text)} (logit: {val.item():.3f})")

    # JAX版本
    print("\\n=== JAX 版本 ===")
    config = modeling.ModelConfig.qwen2_0_5b(use_sharding=False)
    jax_model = params.create_model_from_safe_tensors(model_path, config)

    batch_size, token_len = tokens_jax.shape
    cache = jax_model.init_cache(config, batch_size, token_len, 10)

    jax_logits, _ = modeling.forward(jax_model, cache, tokens_jax, tokenizer.pad_token_id, tokenizer.vocab_size)
    jax_logits = jax_logits[0, :tokenizer.vocab_size]  # 只取有效vocab范围

    print(f"Logits shape: {jax_logits.shape}")
    print(f"Logits range: [{float(jax_logits.min()):.3f}, {float(jax_logits.max()):.3f}]")

    # Top 5 tokens
    jax_top_indices = jnp.argsort(jax_logits)[-5:][::-1]
    jax_top_values = jax_logits[jax_top_indices]
    print("Top 5 tokens:")
    for i, (idx, val) in enumerate(zip(jax_top_indices, jax_top_values)):
        token_text = tokenizer.decode([int(idx)])
        print(f"  {i+1}. ID {idx}: {repr(token_text)} (logit: {float(val):.3f})")

    # 数值对比
    print("\\n=== 数值对比 ===")
    torch_logits_np = torch_logits.numpy()
    jax_logits_np = np.array(jax_logits)

    diff = np.abs(torch_logits_np - jax_logits_np)
    print(f"最大差异: {diff.max():.6f}")
    print(f"平均差异: {diff.mean():.6f}")
    print(f"差异标准差: {diff.std():.6f}")

    # 检查是否高度相关
    correlation = np.corrcoef(torch_logits_np, jax_logits_np)[0, 1]
    print(f"相关性: {correlation:.6f}")

    if correlation < 0.9:
        print("⚠️  WARNING: 低相关性表明模型输出显著不同")
        # 找出差异最大的几个位置
        worst_indices = np.argsort(diff)[-10:][::-1]
        print("差异最大的10个token:")
        for idx in worst_indices:
            torch_val, jax_val = torch_logits_np[idx], jax_logits_np[idx]
            token_text = tokenizer.decode([idx])
            print(f"  Token {idx} {repr(token_text)}: PyTorch={float(torch_val):.3f}, JAX={float(jax_val):.3f}, diff={float(diff[idx]):.3f}")

if __name__ == "__main__":
    compare_logits()