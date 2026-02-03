#!/usr/bin/env python3
"""对比JAX和PyTorch模型的中间层输出"""

import torch
import jax.numpy as jnp
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from bonsai.models.qwen2 import modeling, params

def debug_intermediate_outputs():
    """对比中间层输出找出差异"""
    print("=== 对比中间层输出 ===")

    model_path = snapshot_download("Qwen/Qwen2-0.5B")

    # 准备相同的输入
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    test_text = "Hello, how are you?"
    inputs = tokenizer(test_text, return_tensors="pt")
    tokens_torch = inputs["input_ids"]
    tokens_jax = jnp.array(tokens_torch.numpy())

    print(f"Input: {test_text}")
    print(f"Tokens: {tokens_torch}")

    # === PyTorch版本 ===
    print(f"\n=== PyTorch版本 ===")
    torch_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    torch_model.eval()

    # 从PyTorch模型提取embeddings
    with torch.no_grad():
        # 手动执行embedding
        torch_embeddings = torch_model.model.embed_tokens(tokens_torch)  # [1, seq_len, emb_dim]
        print(f"Embeddings shape: {torch_embeddings.shape}")
        print(f"Embeddings range: [{torch_embeddings.min().item():.3f}, {torch_embeddings.max().item():.3f}]")
        print(f"Embeddings[0,0,:5]: {torch_embeddings[0,0,:5]}")

        # 获取第一层的输入（embeddings）
        hidden_states = torch_embeddings
        layer_0 = torch_model.model.layers[0]

        # 第一层的input layernorm
        normed_input = layer_0.input_layernorm(hidden_states)
        print(f"\nLayer 0 input norm:")
        print(f"  Shape: {normed_input.shape}")
        print(f"  Range: [{normed_input.min().item():.3f}, {normed_input.max().item():.3f}]")
        print(f"  [0,0,:5]: {normed_input[0,0,:5]}")

        # 注意力层的Q投影（手动计算）
        q_out = torch.nn.functional.linear(normed_input, layer_0.self_attn.q_proj.weight)
        print(f"\nLayer 0 Q projection:")
        print(f"  Shape: {q_out.shape}")
        print(f"  Range: [{q_out.min().item():.3f}, {q_out.max().item():.3f}]")
        print(f"  [0,0,:5]: {q_out[0,0,:5]}")

    # === JAX版本 ===
    print(f"\n=== JAX版本 ===")
    config = modeling.ModelConfig.qwen2_0_5b(use_sharding=False)
    jax_model = params.create_model_from_safe_tensors(model_path, config)

    # JAX embeddings
    jax_embeddings = jax_model.embedder.embedding[...][tokens_jax[0]]  # [seq_len, emb_dim]
    jax_embeddings = jax_embeddings[None, :]  # [1, seq_len, emb_dim]
    print(f"Embeddings shape: {jax_embeddings.shape}")
    print(f"Embeddings range: [{float(jax_embeddings.min()):.3f}, {float(jax_embeddings.max()):.3f}]")
    print(f"Embeddings[0,0,:5]: {jax_embeddings[0,0,:5]}")

    # 对比embeddings
    torch_emb_np = torch_embeddings.numpy()
    jax_emb_np = jnp.array(jax_embeddings)
    emb_diff = jnp.abs(torch_emb_np - jax_emb_np).max()
    print(f"\nEmbeddings差异: {float(emb_diff):.6f}")

    # JAX第一层input layernorm
    # 注意：需要模拟layernorm的计算
    # 这里需要访问JAX模型的第一层结构
    try:
        layer_0_jax = jax_model.layers[0]
        # 手动执行input layernorm
        # 注意：这需要了解layernorm的具体实现
        print(f"\nJAX Layer 0结构:")
        print(f"  有input_layernorm: {hasattr(layer_0_jax, 'input_layernorm')}")
        if hasattr(layer_0_jax, 'input_layernorm'):
            # 这需要具体的JAX layernorm实现细节
            print(f"  input_layernorm类型: {type(layer_0_jax.input_layernorm)}")

    except Exception as e:
        print(f"JAX层访问错误: {e}")

    print(f"\n=== 对比总结 ===")
    if emb_diff < 1e-6:
        print("✅ Embeddings匹配良好")
    else:
        print(f"⚠️ Embeddings有差异: {float(emb_diff):.6f}")

if __name__ == "__main__":
    debug_intermediate_outputs()