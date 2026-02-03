#!/usr/bin/env python3
"""逐步对比JAX和PyTorch的完整forward过程"""

import torch
import jax.numpy as jnp
import numpy as np
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from bonsai.models.qwen2 import modeling, params

def debug_full_forward():
    """逐步对比完整的forward过程"""
    print("=== 逐步对比完整forward过程 ===")

    model_path = snapshot_download("Qwen/Qwen2-0.5B")

    # 使用简单输入进行调试
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    test_text = "Hello"
    inputs = tokenizer(test_text, return_tensors="pt")
    tokens_torch = inputs["input_ids"]
    tokens_jax = jnp.array(tokens_torch.numpy())

    print(f"Input: {test_text}")
    print(f"Tokens: {tokens_torch}")
    print(f"Token shape: {tokens_torch.shape}")

    # === PyTorch完整forward ===
    print(f"\n=== PyTorch forward ===")
    torch_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    torch_model.eval()

    with torch.no_grad():
        # Step 1: Embeddings
        torch_embeddings = torch_model.model.embed_tokens(tokens_torch)
        print(f"1. Embeddings: {torch_embeddings.shape}, range: [{torch_embeddings.min().item():.6f}, {torch_embeddings.max().item():.6f}]")

        # Step 2: 第一层
        layer_0 = torch_model.model.layers[0]
        hidden_states = torch_embeddings

        # Step 2.1: Input LayerNorm
        normed_input = layer_0.input_layernorm(hidden_states)
        print(f"2.1 Input LayerNorm: {normed_input.shape}, range: [{normed_input.min().item():.6f}, {normed_input.max().item():.6f}]")

        # Step 2.2: Attention QKV projections
        q_proj = layer_0.self_attn.q_proj(normed_input)
        k_proj = layer_0.self_attn.k_proj(normed_input)
        v_proj = layer_0.self_attn.v_proj(normed_input)
        print(f"2.2 Q proj: {q_proj.shape}, range: [{q_proj.min().item():.6f}, {q_proj.max().item():.6f}]")
        print(f"2.2 K proj: {k_proj.shape}, range: [{k_proj.min().item():.6f}, {k_proj.max().item():.6f}]")
        print(f"2.2 V proj: {v_proj.shape}, range: [{v_proj.min().item():.6f}, {v_proj.max().item():.6f}]")

        # Step 2.3: Reshape for attention heads
        bsz, seq_len, _ = q_proj.shape
        num_heads = 14
        num_kv_heads = 2
        head_dim = 64

        query_states = q_proj.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        key_states = k_proj.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = v_proj.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        print(f"2.3 Query states: {query_states.shape}")
        print(f"2.3 Key states: {key_states.shape}")
        print(f"2.3 Value states: {value_states.shape}")

        # Step 2.4: 完整的attention（包括RoPE）
        # 创建必要的attention参数
        attention_mask = torch.ones((1, 1), dtype=torch.bool)
        position_ids = torch.arange(0, tokens_torch.shape[-1]).unsqueeze(0)

        attn_output = layer_0.self_attn(
            hidden_states=normed_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            position_embeddings=None
        )[0]
        print(f"2.4 Attention output: {attn_output.shape}, range: [{attn_output.min().item():.6f}, {attn_output.max().item():.6f}]")

        # Step 2.5: 残差连接
        hidden_states = hidden_states + attn_output
        print(f"2.5 After attn residual: {hidden_states.shape}, range: [{hidden_states.min().item():.6f}, {hidden_states.max().item():.6f}]")

        # Step 2.6: Post attention LayerNorm
        post_norm = layer_0.post_attention_layernorm(hidden_states)
        print(f"2.6 Post attn norm: {post_norm.shape}, range: [{post_norm.min().item():.6f}, {post_norm.max().item():.6f}]")

        # Step 2.7: MLP
        mlp_output = layer_0.mlp(post_norm)
        print(f"2.7 MLP output: {mlp_output.shape}, range: [{mlp_output.min().item():.6f}, {mlp_output.max().item():.6f}]")

        # Step 2.8: 第二个残差连接
        hidden_states = hidden_states + mlp_output
        print(f"2.8 After MLP residual: {hidden_states.shape}, range: [{hidden_states.min().item():.6f}, {hidden_states.max().item():.6f}]")

        # 完整forward到最后
        full_output = torch_model(tokens_torch)
        final_logits = full_output.logits[:, -1, :tokenizer.vocab_size]
        print(f"Final logits: {final_logits.shape}, range: [{final_logits.min().item():.6f}, {final_logits.max().item():.6f}]")

    # === JAX forward ===
    print(f"\n=== JAX forward ===")
    config = modeling.ModelConfig.qwen2_0_5b(use_sharding=False)
    jax_model = params.create_model_from_safe_tensors(model_path, config)

    # Step 1: Embeddings
    jax_embeddings = jax_model.embedder.embedding[...][tokens_jax[0]]
    jax_embeddings = jax_embeddings[None, :]  # Add batch dimension
    print(f"1. Embeddings: {jax_embeddings.shape}, range: [{float(jax_embeddings.min()):.6f}, {float(jax_embeddings.max()):.6f}]")

    # 对比embeddings
    emb_diff = jnp.abs(torch_embeddings.numpy() - jnp.array(jax_embeddings)).max()
    print(f"   Embeddings差异: {float(emb_diff):.8f}")

    # Step 2: 第一层
    jax_layer_0 = jax_model.layers[0]
    jax_hidden = jax_embeddings

    # Step 2.1: Input LayerNorm
    jax_normed_input = jax_layer_0.input_layernorm(jax_hidden)
    print(f"2.1 Input LayerNorm: {jax_normed_input.shape}, range: [{float(jax_normed_input.min()):.6f}, {float(jax_normed_input.max()):.6f}]")

    # 对比input layernorm
    norm_diff = jnp.abs(normed_input.numpy() - jnp.array(jax_normed_input)).max()
    print(f"   Input LayerNorm差异: {float(norm_diff):.8f}")

    # Step 2.2: QKV投影
    jax_q = jax_layer_0.attn.q_proj(jax_normed_input)
    jax_k = jax_layer_0.attn.k_proj(jax_normed_input)
    jax_v = jax_layer_0.attn.v_proj(jax_normed_input)
    print(f"2.2 Q proj: {jax_q.shape}, range: [{float(jax_q.min()):.6f}, {float(jax_q.max()):.6f}]")
    print(f"2.2 K proj: {jax_k.shape}, range: [{float(jax_k.min()):.6f}, {float(jax_k.max()):.6f}]")
    print(f"2.2 V proj: {jax_v.shape}, range: [{float(jax_v.min()):.6f}, {float(jax_v.max()):.6f}]")

    # 对比QKV投影
    q_diff = jnp.abs(q_proj.numpy() - jnp.array(jax_q)).max()
    k_diff = jnp.abs(k_proj.numpy() - jnp.array(jax_k)).max()
    v_diff = jnp.abs(v_proj.numpy() - jnp.array(jax_v)).max()
    print(f"   Q投影差异: {float(q_diff):.8f}")
    print(f"   K投影差异: {float(k_diff):.8f}")
    print(f"   V投影差异: {float(v_diff):.8f}")

    # 如果QKV投影有显著差异，这就是问题所在
    if max(q_diff, k_diff, v_diff) > 1e-5:
        print(f"\n⚠️  发现QKV投影有显著差异！")
        print(f"   这可能是乱码问题的根源")

        # 详细分析第一个差异
        if q_diff > 1e-5:
            print(f"\nQ投影详细分析:")
            print(f"PyTorch Q[0,0,:5]: {q_proj[0,0,:5]}")
            print(f"JAX Q[0,0,:5]: {jax_q[0,0,:5]}")

    # 检查完整的attention计算
    batch_size = 1
    token_len = tokens_jax.shape[-1]
    cache = jax_model.init_cache(config, batch_size, token_len, 10)

    # 创建segment_ids
    segment_ids = jnp.ones_like(tokens_jax)

    # JAX attention
    jax_attn_out = jax_layer_0.attn(jax_normed_input, cache.layers[0], segment_ids)
    print(f"2.4 JAX Attention output: {jax_attn_out.shape}, range: [{float(jax_attn_out.min()):.6f}, {float(jax_attn_out.max()):.6f}]")

    # 对比attention输出
    attn_diff = jnp.abs(attn_output.numpy() - jnp.array(jax_attn_out)).max()
    print(f"   Attention输出差异: {float(attn_diff):.8f}")

    if attn_diff > 0.01:
        print(f"\n⚠️  Attention输出差异很大！")
        print(f"   这确认了乱码问题在attention层")

    # 完整JAX forward
    jax_logits, _ = modeling.forward(jax_model, cache, tokens_jax, tokenizer.pad_token_id, tokenizer.vocab_size)
    jax_final_logits = jax_logits[0, :tokenizer.vocab_size]
    print(f"JAX Final logits: {jax_final_logits.shape}, range: [{float(jax_final_logits.min()):.6f}, {float(jax_final_logits.max()):.6f}]")

    # 最终对比
    final_diff = jnp.abs(final_logits.numpy() - jnp.array(jax_final_logits)).max()
    print(f"\n=== 最终对比 ===")
    print(f"最终logits差异: {float(final_diff):.6f}")

    # 相关性检查
    correlation = np.corrcoef(final_logits.numpy(), jnp.array(jax_final_logits))[0, 1]
    print(f"最终logits相关性: {correlation:.6f}")

    if correlation < 0.9:
        print("❌ 确认：JAX和PyTorch输出显著不同")
    else:
        print("✅ JAX和PyTorch输出基本一致")

if __name__ == "__main__":
    debug_full_forward()