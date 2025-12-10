#!/usr/bin/env python3
"""对比LayerNorm实现差异"""

import torch
import jax.numpy as jnp
import numpy as np
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from bonsai.models.qwen2 import modeling, params

def debug_layernorm():
    """对比layernorm差异"""
    print("=== 对比LayerNorm实现 ===")

    model_path = snapshot_download("Qwen/Qwen2-0.5B")

    # 准备相同的输入
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    test_text = "Hello"
    inputs = tokenizer(test_text, return_tensors="pt")
    tokens_torch = inputs["input_ids"]

    # === PyTorch版本 ===
    torch_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    torch_model.eval()

    with torch.no_grad():
        torch_embeddings = torch_model.model.embed_tokens(tokens_torch)
        print(f"Input shape: {torch_embeddings.shape}")
        print(f"Input range: [{torch_embeddings.min().item():.6f}, {torch_embeddings.max().item():.6f}]")

        # PyTorch layernorm
        layer_0 = torch_model.model.layers[0]
        torch_normalized = layer_0.input_layernorm(torch_embeddings)
        print(f"\nPyTorch LayerNorm:")
        print(f"  Type: {type(layer_0.input_layernorm)}")
        print(f"  Output shape: {torch_normalized.shape}")
        print(f"  Output range: [{torch_normalized.min().item():.6f}, {torch_normalized.max().item():.6f}]")
        print(f"  First token norm: {torch_normalized[0, 0, :5]}")

    # === JAX版本 ===
    config = modeling.ModelConfig.qwen2_0_5b(use_sharding=False)
    jax_model = params.create_model_from_safe_tensors(model_path, config)

    # JAX embeddings (相同输入)
    tokens_jax = jnp.array(tokens_torch.numpy())
    jax_embeddings = jax_model.embedder.embedding[...][tokens_jax[0]]
    jax_embeddings = jax_embeddings[None, :]  # Add batch dimension

    print(f"\nJAX Input:")
    print(f"  Shape: {jax_embeddings.shape}")
    print(f"  Range: [{float(jax_embeddings.min()):.6f}, {float(jax_embeddings.max()):.6f}]")

    # JAX RMSNorm
    layer_0_jax = jax_model.layers[0]
    jax_normalized = layer_0_jax.input_layernorm(jax_embeddings)
    print(f"\nJAX RMSNorm:")
    print(f"  Type: {type(layer_0_jax.input_layernorm)}")
    print(f"  Output shape: {jax_normalized.shape}")
    print(f"  Output range: [{float(jax_normalized.min()):.6f}, {float(jax_normalized.max()):.6f}]")
    print(f"  First token norm: {jax_normalized[0, 0, :5]}")

    # 对比
    torch_norm_np = torch_normalized.numpy()
    jax_norm_np = jnp.array(jax_normalized)
    diff = jnp.abs(torch_norm_np - jax_norm_np)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())

    print(f"\n=== LayerNorm对比 ===")
    print(f"最大差异: {max_diff:.6f}")
    print(f"平均差异: {mean_diff:.6f}")
    print(f"相对差异: {max_diff / float(jnp.abs(torch_norm_np).max()):.6f}")

    if max_diff > 0.01:
        print("⚠️ LayerNorm实现差异很大！")

        # 检查配置参数
        print(f"\nLayerNorm参数检查:")
        print(f"  JAX norm_eps: {config.norm_eps}")

        # 尝试手动实现RMSNorm来对比
        print(f"\n手动RMSNorm验证:")
        manual_rms_norm = manual_rmsnorm(torch_embeddings, torch_model.model.layers[0].input_layernorm.weight, config.norm_eps)
        manual_diff = jnp.abs(manual_rms_norm - jax_norm_np).max()
        print(f"  手动实现与JAX的差异: {float(manual_diff):.6f}")

    else:
        print("✅ LayerNorm实现基本一致")

    return max_diff

def manual_rmsnorm(x, weight, eps):
    """手动实现RMSNorm"""
    x_np = x.numpy()
    weight_np = weight.detach().numpy()

    # RMSNorm: x / sqrt(mean(x^2) + eps) * weight
    rms = np.sqrt(np.mean(x_np**2, axis=-1, keepdims=True) + eps)
    normalized = x_np / rms * weight_np
    return normalized

if __name__ == "__main__":
    debug_layernorm()