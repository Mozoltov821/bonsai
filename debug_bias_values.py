#!/usr/bin/env python3
"""检查JAX模型中bias的实际值"""

import jax.numpy as jnp
from huggingface_hub import snapshot_download
from bonsai.models.qwen2 import modeling, params

def check_bias_values():
    """检查JAX模型加载后的bias值"""
    print("=== 检查JAX模型bias值 ===")

    model_path = snapshot_download("Qwen/Qwen2-0.5B")
    config = modeling.ModelConfig.qwen2_0_5b(use_sharding=False)
    jax_model = params.create_model_from_safe_tensors(model_path, config)

    # 检查第一层的bias值
    layer_0 = jax_model.layers[0].attn

    print(f"\n第一层attention bias值:")
    print(f"q_bias is None: {layer_0.q_bias is None}")
    print(f"k_bias is None: {layer_0.k_bias is None}")
    print(f"v_bias is None: {layer_0.v_bias is None}")
    print(f"o_bias is None: {layer_0.o_bias is None}")

    if layer_0.q_bias is not None:
        q_bias_val = layer_0.q_bias.value if hasattr(layer_0.q_bias, 'value') else layer_0.q_bias
        print(f"Q bias shape: {q_bias_val.shape}")
        print(f"Q bias range: [{float(q_bias_val.min()):.6f}, {float(q_bias_val.max()):.6f}]")
        print(f"Q bias[:10]: {q_bias_val[:10]}")

    if layer_0.k_bias is not None:
        k_bias_val = layer_0.k_bias.value if hasattr(layer_0.k_bias, 'value') else layer_0.k_bias
        print(f"K bias shape: {k_bias_val.shape}")
        print(f"K bias range: [{float(k_bias_val.min()):.6f}, {float(k_bias_val.max()):.6f}]")
        print(f"K bias[:10]: {k_bias_val[:10]}")

    if layer_0.v_bias is not None:
        v_bias_val = layer_0.v_bias.value if hasattr(layer_0.v_bias, 'value') else layer_0.v_bias
        print(f"V bias shape: {v_bias_val.shape}")
        print(f"V bias range: [{float(v_bias_val.min()):.6f}, {float(v_bias_val.max()):.6f}]")
        print(f"V bias[:10]: {v_bias_val[:10]}")

    # 测试一个简单的前向传播
    print(f"\n=== 测试前向传播中的bias应用 ===")

    # 创建一个简单的输入
    test_input = jnp.ones((1, 1, 896)) * 0.1  # 小的均匀输入

    # 执行QKV投影
    q_linear = layer_0.q_proj(test_input)
    k_linear = layer_0.k_proj(test_input)
    v_linear = layer_0.v_proj(test_input)

    print(f"Q线性输出 (before bias): {q_linear.shape}, range: [{float(q_linear.min()):.6f}, {float(q_linear.max()):.6f}]")
    print(f"K线性输出 (before bias): {k_linear.shape}, range: [{float(k_linear.min()):.6f}, {float(k_linear.max()):.6f}]")
    print(f"V线性输出 (before bias): {v_linear.shape}, range: [{float(v_linear.min()):.6f}, {float(v_linear.max()):.6f}]")

    # 手动应用bias
    if layer_0.q_bias is not None:
        q_bias_val = layer_0.q_bias.value if hasattr(layer_0.q_bias, 'value') else layer_0.q_bias
        q_with_bias = q_linear + q_bias_val
        print(f"Q加bias后: {q_with_bias.shape}, range: [{float(q_with_bias.min()):.6f}, {float(q_with_bias.max()):.6f}]")

    if layer_0.k_bias is not None:
        k_bias_val = layer_0.k_bias.value if hasattr(layer_0.k_bias, 'value') else layer_0.k_bias
        k_with_bias = k_linear + k_bias_val
        print(f"K加bias后: {k_with_bias.shape}, range: [{float(k_with_bias.min()):.6f}, {float(k_with_bias.max()):.6f}]")

    if layer_0.v_bias is not None:
        v_bias_val = layer_0.v_bias.value if hasattr(layer_0.v_bias, 'value') else layer_0.v_bias
        v_with_bias = v_linear + v_bias_val
        print(f"V加bias后: {v_with_bias.shape}, range: [{float(v_with_bias.min()):.6f}, {float(v_with_bias.max()):.6f}]")

if __name__ == "__main__":
    check_bias_values()