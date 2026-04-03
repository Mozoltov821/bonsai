#!/usr/bin/env python3
"""测试权重转换的正确性"""

import torch
import numpy as np
import safetensors
from etils import epath
from huggingface_hub import snapshot_download
from transformers import AutoConfig

def test_weight_transforms():
    """测试权重转换逻辑"""
    print("=== 测试权重转换 ===")

    model_path = snapshot_download("Qwen/Qwen2-0.5B")
    config = AutoConfig.from_pretrained(model_path)

    emb_dim = config.hidden_size  # 896
    num_heads = config.num_attention_heads  # 14
    num_kv_heads = config.num_key_value_heads  # 2
    head_dim = config.hidden_size // config.num_attention_heads  # 64

    print(f"模型配置: emb_dim={emb_dim}, num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")

    files = list(epath.Path(model_path).expanduser().glob("*.safetensors"))

    with safetensors.safe_open(files[0], framework="pt") as sf:
        # 获取第一层的权重
        q_weight = sf.get_tensor("model.layers.0.self_attn.q_proj.weight").float().numpy()
        k_weight = sf.get_tensor("model.layers.0.self_attn.k_proj.weight").float().numpy()
        v_weight = sf.get_tensor("model.layers.0.self_attn.v_proj.weight").float().numpy()
        o_weight = sf.get_tensor("model.layers.0.self_attn.o_proj.weight").float().numpy()

        print(f"\n原始权重形状:")
        print(f"  q_proj: {q_weight.shape}")
        print(f"  k_proj: {k_weight.shape}")
        print(f"  v_proj: {v_weight.shape}")
        print(f"  o_proj: {o_weight.shape}")

        # 手动执行转换逻辑
        def apply_transform(tensor, permute, reshape, reshape_first):
            """应用转换规则"""
            print(f"    输入形状: {tensor.shape}")

            if reshape_first and reshape is not None:
                print(f"    先reshape: {tensor.shape} -> {reshape}")
                tensor = tensor.reshape(reshape)
                print(f"    reshape后: {tensor.shape}")

            if permute:
                print(f"    permute{permute}: {tensor.shape} -> ", end="")
                tensor = tensor.transpose(permute)
                print(f"{tensor.shape}")

            if not reshape_first and reshape is not None:
                print(f"    后reshape: {tensor.shape} -> {reshape}")
                tensor = tensor.reshape(reshape)
                print(f"    reshape后: {tensor.shape}")

            return tensor

        # 测试Q权重转换
        print(f"\n=== Q权重转换 ===")
        q_transform = ((2, 0, 1), (num_heads, head_dim, emb_dim), True)
        permute, reshape, reshape_first = q_transform
        q_converted = apply_transform(q_weight.copy(), permute, reshape, reshape_first)
        q_expected = (emb_dim, num_heads, head_dim)  # (896, 14, 64)
        print(f"  期望形状: {q_expected}")
        print(f"  实际形状: {q_converted.shape}")
        print(f"  ✅ 形状匹配: {q_converted.shape == q_expected}")

        # 测试K权重转换
        print(f"\n=== K权重转换 ===")
        kv_transform = ((1, 0), (emb_dim, num_kv_heads, head_dim), False)
        permute, reshape, reshape_first = kv_transform
        k_converted = apply_transform(k_weight.copy(), permute, reshape, reshape_first)
        kv_expected = (emb_dim, num_kv_heads, head_dim)  # (896, 2, 64)
        print(f"  期望形状: {kv_expected}")
        print(f"  实际形状: {k_converted.shape}")
        print(f"  ✅ 形状匹配: {k_converted.shape == kv_expected}")

        # 测试V权重转换
        print(f"\n=== V权重转换 ===")
        v_converted = apply_transform(v_weight.copy(), permute, reshape, reshape_first)
        print(f"  期望形状: {kv_expected}")
        print(f"  实际形状: {v_converted.shape}")
        print(f"  ✅ 形状匹配: {v_converted.shape == kv_expected}")

        # 测试O权重转换
        print(f"\n=== O权重转换 ===")
        o_transform = ((1, 0), (num_heads, head_dim, emb_dim), False)
        permute, reshape, reshape_first = o_transform
        o_converted = apply_transform(o_weight.copy(), permute, reshape, reshape_first)
        o_expected = (num_heads, head_dim, emb_dim)  # (14, 64, 896)
        print(f"  期望形状: {o_expected}")
        print(f"  实际形状: {o_converted.shape}")
        print(f"  ✅ 形状匹配: {o_converted.shape == o_expected}")

        # 检查数值范围
        print(f"\n=== 数值范围检查 ===")
        print(f"Q权重: [{q_converted.min():.3f}, {q_converted.max():.3f}]")
        print(f"K权重: [{k_converted.min():.3f}, {k_converted.max():.3f}]")
        print(f"V权重: [{v_converted.min():.3f}, {v_converted.max():.3f}]")
        print(f"O权重: [{o_converted.min():.3f}, {o_converted.max():.3f}]")

if __name__ == "__main__":
    test_weight_transforms()