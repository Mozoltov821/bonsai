#!/usr/bin/env python3
"""专注于QKV投影的对比"""

import torch
import jax.numpy as jnp
import numpy as np
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from bonsai.models.qwen2 import modeling, params

def debug_qkv_simple():
    """专注对比QKV投影"""
    print("=== 专注对比QKV投影 ===")

    model_path = snapshot_download("Qwen/Qwen2-0.5B")

    # 使用相同的简单输入
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    test_text = "Hello"
    inputs = tokenizer(test_text, return_tensors="pt")
    tokens_torch = inputs["input_ids"]
    tokens_jax = jnp.array(tokens_torch.numpy())

    print(f"Input: {test_text}")
    print(f"Tokens: {tokens_torch}")

    # === PyTorch ===
    print(f"\n=== PyTorch版本 ===")
    torch_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    torch_model.eval()

    with torch.no_grad():
        # Embeddings
        torch_embeddings = torch_model.model.embed_tokens(tokens_torch)
        print(f"Embeddings: {torch_embeddings.shape}, range: [{torch_embeddings.min().item():.6f}, {torch_embeddings.max().item():.6f}]")

        # 第一层
        layer_0 = torch_model.model.layers[0]

        # Input LayerNorm
        torch_normed = layer_0.input_layernorm(torch_embeddings)
        print(f"Input norm: {torch_normed.shape}, range: [{torch_normed.min().item():.6f}, {torch_normed.max().item():.6f}]")
        print(f"Input norm[0,0,:10]: {torch_normed[0,0,:10]}")

        # 手动QKV投影
        torch_q = torch.nn.functional.linear(torch_normed, layer_0.self_attn.q_proj.weight, layer_0.self_attn.q_proj.bias)
        torch_k = torch.nn.functional.linear(torch_normed, layer_0.self_attn.k_proj.weight, layer_0.self_attn.k_proj.bias)
        torch_v = torch.nn.functional.linear(torch_normed, layer_0.self_attn.v_proj.weight, layer_0.self_attn.v_proj.bias)

        print(f"Q proj: {torch_q.shape}, range: [{torch_q.min().item():.6f}, {torch_q.max().item():.6f}]")
        print(f"K proj: {torch_k.shape}, range: [{torch_k.min().item():.6f}, {torch_k.max().item():.6f}]")
        print(f"V proj: {torch_v.shape}, range: [{torch_v.min().item():.6f}, {torch_v.max().item():.6f}]")
        print(f"Q[0,0,:10]: {torch_q[0,0,:10]}")
        print(f"K[0,0,:10]: {torch_k[0,0,:10]}")
        print(f"V[0,0,:10]: {torch_v[0,0,:10]}")

    # === JAX ===
    print(f"\n=== JAX版本 ===")
    config = modeling.ModelConfig.qwen2_0_5b(use_sharding=False)
    jax_model = params.create_model_from_safe_tensors(model_path, config)

    # Embeddings
    jax_embeddings = jax_model.embedder.embedding[...][tokens_jax[0]]
    jax_embeddings = jax_embeddings[None, :]
    print(f"Embeddings: {jax_embeddings.shape}, range: [{float(jax_embeddings.min()):.6f}, {float(jax_embeddings.max()):.6f}]")

    # 第一层
    jax_layer_0 = jax_model.layers[0]

    # Input LayerNorm
    jax_normed = jax_layer_0.input_layernorm(jax_embeddings)
    print(f"Input norm: {jax_normed.shape}, range: [{float(jax_normed.min()):.6f}, {float(jax_normed.max()):.6f}]")
    print(f"Input norm[0,0,:10]: {jax_normed[0,0,:10]}")

    # 测试直接QKV投影（绕过bias）
    jax_q_direct = jax_layer_0.attn.q_proj(jax_normed)
    jax_k_direct = jax_layer_0.attn.k_proj(jax_normed)
    jax_v_direct = jax_layer_0.attn.v_proj(jax_normed)

    print(f"Q proj (direct): {jax_q_direct.shape}, range: [{float(jax_q_direct.min()):.6f}, {float(jax_q_direct.max()):.6f}]")
    print(f"K proj (direct): {jax_k_direct.shape}, range: [{float(jax_k_direct.min()):.6f}, {float(jax_k_direct.max()):.6f}]")
    print(f"V proj (direct): {jax_v_direct.shape}, range: [{float(jax_v_direct.min()):.6f}, {float(jax_v_direct.max()):.6f}]")

    # 手动应用bias来验证
    q_bias_val = jax_layer_0.attn.q_bias.value if hasattr(jax_layer_0.attn.q_bias, 'value') else jax_layer_0.attn.q_bias
    k_bias_val = jax_layer_0.attn.k_bias.value if hasattr(jax_layer_0.attn.k_bias, 'value') else jax_layer_0.attn.k_bias
    v_bias_val = jax_layer_0.attn.v_bias.value if hasattr(jax_layer_0.attn.v_bias, 'value') else jax_layer_0.attn.v_bias

    jax_q = jax_q_direct + q_bias_val
    jax_k = jax_k_direct + k_bias_val
    jax_v = jax_v_direct + v_bias_val

    print(f"Q proj (with bias): {jax_q.shape}, range: [{float(jax_q.min()):.6f}, {float(jax_q.max()):.6f}]")
    print(f"K proj (with bias): {jax_k.shape}, range: [{float(jax_k.min()):.6f}, {float(jax_k.max()):.6f}]")
    print(f"V proj (with bias): {jax_v.shape}, range: [{float(jax_v.min()):.6f}, {float(jax_v.max()):.6f}]")
    print(f"Q[0,0,:10]: {jax_q[0,0,:10]}")
    print(f"K[0,0,:10]: {jax_k[0,0,:10]}")
    print(f"V[0,0,:10]: {jax_v[0,0,:10]}")

    # === 详细对比 ===
    print(f"\n=== 详细对比 ===")

    # 对比embeddings
    emb_diff = jnp.abs(torch_embeddings.numpy() - jnp.array(jax_embeddings)).max()
    print(f"Embeddings差异: {float(emb_diff):.8f}")

    # 对比input norm
    norm_diff = jnp.abs(torch_normed.numpy() - jnp.array(jax_normed)).max()
    print(f"Input norm差异: {float(norm_diff):.8f}")

    # 对比QKV
    q_diff = jnp.abs(torch_q.numpy() - jnp.array(jax_q)).max()
    k_diff = jnp.abs(torch_k.numpy() - jnp.array(jax_k)).max()
    v_diff = jnp.abs(torch_v.numpy() - jnp.array(jax_v)).max()

    print(f"Q投影差异: {float(q_diff):.8f}")
    print(f"K投影差异: {float(k_diff):.8f}")
    print(f"V投影差异: {float(v_diff):.8f}")

    # 检查bias
    print(f"\n=== Bias检查 ===")
    q_bias = layer_0.self_attn.q_proj.bias
    k_bias = layer_0.self_attn.k_proj.bias
    v_bias = layer_0.self_attn.v_proj.bias

    if q_bias is not None:
        print(f"PyTorch Q bias: {q_bias.shape}, range: [{q_bias.min().item():.6f}, {q_bias.max().item():.6f}]")
    else:
        print("PyTorch Q bias: None")

    if k_bias is not None:
        print(f"PyTorch K bias: {k_bias.shape}, range: [{k_bias.min().item():.6f}, {k_bias.max().item():.6f}]")
    else:
        print("PyTorch K bias: None")

    if v_bias is not None:
        print(f"PyTorch V bias: {v_bias.shape}, range: [{v_bias.min().item():.6f}, {v_bias.max().item():.6f}]")
    else:
        print("PyTorch V bias: None")

    # 检查最大差异
    max_diff = max(q_diff, k_diff, v_diff)
    if max_diff > 1e-5:
        print(f"\n⚠️  发现QKV投影显著差异！最大差异: {float(max_diff):.8f}")
        print("这可能是乱码问题的根源")

        # 找出差异最大的位置
        if q_diff == max_diff:
            print("Q投影差异最大")
            q_diff_matrix = jnp.abs(torch_q.numpy() - jnp.array(jax_q))
            max_pos = jnp.unravel_index(jnp.argmax(q_diff_matrix), q_diff_matrix.shape)
            print(f"最大差异位置: {max_pos}")
            print(f"PyTorch值: {torch_q.numpy()[max_pos]:.6f}")
            print(f"JAX值: {float(jax_q[max_pos]):.6f}")

        elif k_diff == max_diff:
            print("K投影差异最大")
        elif v_diff == max_diff:
            print("V投影差异最大")
    else:
        print(f"\n✅ QKV投影基本匹配，最大差异: {float(max_diff):.8f}")
        print("问题可能在后续的attention计算步骤")

if __name__ == "__main__":
    debug_qkv_simple()