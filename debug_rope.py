#!/usr/bin/env python3
"""对比RoPE实现差异"""

import torch
import jax.numpy as jnp
import numpy as np
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from bonsai.models.qwen2 import modeling, params

def debug_rope():
    """对比RoPE实现"""
    print("=== 对比RoPE实现 ===")

    model_path = snapshot_download("Qwen/Qwen2-0.5B")

    # 准备测试参数
    batch_size = 1
    seq_len = 6
    head_dim = 64  # Qwen2-0.5B的head_dim
    rope_theta = 1000000  # Qwen2使用的theta

    # 创建位置序列
    positions = jnp.arange(seq_len)[None, :]  # [1, seq_len]
    print(f"位置序列: {positions}")

    # === JAX RoPE实现 ===
    print(f"\n=== JAX RoPE ===")
    jax_sin, jax_cos = modeling._generate_pos_embeddings(positions, head_dim, rope_theta)
    print(f"Sin shape: {jax_sin.shape}")
    print(f"Cos shape: {jax_cos.shape}")
    print(f"Sin range: [{float(jax_sin.min()):.6f}, {float(jax_sin.max()):.6f}]")
    print(f"Cos range: [{float(jax_cos.min()):.6f}, {float(jax_cos.max()):.6f}]")
    print(f"Sin[0,0,:5]: {jax_sin[0, 0, :5]}")
    print(f"Cos[0,0,:5]: {jax_cos[0, 0, :5]}")

    # === PyTorch RoPE实现 ===
    print(f"\n=== PyTorch RoPE ===")

    # 手动实现PyTorch版本的RoPE来对比
    def pytorch_rope(positions, head_dim, theta=10000):
        """手动实现PyTorch风格的RoPE"""
        positions = torch.tensor(np.array(positions), dtype=torch.float32)

        # 计算频率
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))

        # 计算位置编码
        freqs = torch.einsum('bi,j->bij', positions, inv_freq)

        # 计算sin和cos
        sin = torch.sin(freqs)
        cos = torch.cos(freqs)

        return sin, cos

    torch_sin, torch_cos = pytorch_rope(positions, head_dim, rope_theta)
    print(f"Sin shape: {torch_sin.shape}")
    print(f"Cos shape: {torch_cos.shape}")
    print(f"Sin range: [{torch_sin.min().item():.6f}, {torch_sin.max().item():.6f}]")
    print(f"Cos range: [{torch_cos.min().item():.6f}, {torch_cos.max().item():.6f}]")
    print(f"Sin[0,0,:5]: {torch_sin[0, 0, :5]}")
    print(f"Cos[0,0,:5]: {torch_cos[0, 0, :5]}")

    # === 对比 ===
    print(f"\n=== RoPE对比 ===")
    torch_sin_np = torch_sin.numpy()
    torch_cos_np = torch_cos.numpy()
    jax_sin_np = np.array(jax_sin)
    jax_cos_np = np.array(jax_cos)

    sin_diff = np.abs(torch_sin_np - jax_sin_np).max()
    cos_diff = np.abs(torch_cos_np - jax_cos_np).max()

    print(f"Sin最大差异: {sin_diff:.8f}")
    print(f"Cos最大差异: {cos_diff:.8f}")

    if sin_diff < 1e-6 and cos_diff < 1e-6:
        print("✅ RoPE实现一致")
    else:
        print("⚠️ RoPE实现有差异")

    # === 测试apply_rope函数 ===
    print(f"\n=== 测试apply_rope ===")
    # 创建一个测试query tensor
    test_query = jnp.ones((batch_size, seq_len, 1, head_dim)) * 0.1  # [B, T, H, D]
    print(f"Test query shape: {test_query.shape}")

    # 应用JAX RoPE
    jax_rotated = modeling.apply_rope(test_query, jax_sin, jax_cos)
    print(f"JAX rotated shape: {jax_rotated.shape}")
    print(f"JAX rotated range: [{float(jax_rotated.min()):.6f}, {float(jax_rotated.max()):.6f}]")
    print(f"JAX rotated[0,0,0,:5]: {jax_rotated[0, 0, 0, :5]}")

    # 手动实现PyTorch风格的apply_rope
    def pytorch_apply_rope(x, sin, cos):
        """手动实现PyTorch风格的RoPE应用"""
        x = torch.tensor(x, dtype=torch.float32)
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]

        # 调整sin/cos的维度来匹配x
        sin = sin[:, :, None, :]  # [B, T, 1, head_dim//2]
        cos = cos[:, :, None, :]

        rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return rotated

    pytorch_rotated = pytorch_apply_rope(test_query, torch_sin, torch_cos)
    print(f"PyTorch rotated shape: {pytorch_rotated.shape}")
    print(f"PyTorch rotated range: [{pytorch_rotated.min().item():.6f}, {pytorch_rotated.max().item():.6f}]")
    print(f"PyTorch rotated[0,0,0,:5]: {pytorch_rotated[0, 0, 0, :5]}")

    # 对比apply_rope结果
    apply_rope_diff = np.abs(pytorch_rotated.numpy() - np.array(jax_rotated)).max()
    print(f"\napply_rope最大差异: {apply_rope_diff:.8f}")

    if apply_rope_diff < 1e-6:
        print("✅ apply_rope实现一致")
    else:
        print("⚠️ apply_rope实现有差异")

if __name__ == "__main__":
    debug_rope()