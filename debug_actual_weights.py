#!/usr/bin/env python3
"""调试实际加载的权重"""

import jax
import jax.numpy as jnp
import torch
import safetensors
from etils import epath
from huggingface_hub import snapshot_download
from transformers import AutoConfig

from bonsai.models.qwen2 import modeling, params

def debug_actual_weights():
    """调试实际加载的JAX权重与PyTorch权重的对比"""
    print("=== 调试实际权重加载 ===")

    # 加载模型
    model_path = snapshot_download("Qwen/Qwen2-0.5B")
    config = modeling.ModelConfig.qwen2_0_5b(use_sharding=False)
    jax_model = params.create_model_from_safe_tensors(model_path, config)

    # 从JAX模型提取第一层的attention权重
    jax_q_weight = jax_model.layers[0].attn.q_proj.kernel[...]
    jax_k_weight = jax_model.layers[0].attn.k_proj.kernel[...]
    jax_v_weight = jax_model.layers[0].attn.v_proj.kernel[...]
    jax_o_weight = jax_model.layers[0].attn.o_proj.kernel[...]

    print(f"\nJAX模型第一层attention权重形状:")
    print(f"  q_proj: {jax_q_weight.shape}")
    print(f"  k_proj: {jax_k_weight.shape}")
    print(f"  v_proj: {jax_v_weight.shape}")
    print(f"  o_proj: {jax_o_weight.shape}")

    # 从原始safetensors加载PyTorch权重
    files = list(epath.Path(model_path).expanduser().glob("*.safetensors"))
    with safetensors.safe_open(files[0], framework="pt") as sf:
        torch_q_weight = sf.get_tensor("model.layers.0.self_attn.q_proj.weight").float().numpy()
        torch_k_weight = sf.get_tensor("model.layers.0.self_attn.k_proj.weight").float().numpy()
        torch_v_weight = sf.get_tensor("model.layers.0.self_attn.v_proj.weight").float().numpy()
        torch_o_weight = sf.get_tensor("model.layers.0.self_attn.o_proj.weight").float().numpy()

    print(f"\nPyTorch原始权重形状:")
    print(f"  q_proj: {torch_q_weight.shape}")
    print(f"  k_proj: {torch_k_weight.shape}")
    print(f"  v_proj: {torch_v_weight.shape}")
    print(f"  o_proj: {torch_o_weight.shape}")

    # 手动转换PyTorch权重以进行对比
    emb_dim = config.emb_dim
    num_heads = config.num_heads
    num_kv_heads = config.num_kv_heads
    head_dim = config.head_dim

    def convert_q_weight(w):
        """Q权重转换: (896,896) -> reshape(14,64,896) -> permute(2,0,1) -> (896,14,64)"""
        # return w.reshape(num_heads, head_dim, emb_dim).transpose(2, 0, 1)
        return w.T

    def convert_kv_weight(w):
        """K/V权重转换: (128,896) -> transpose(1,0) -> (896,128) -> reshape -> (896,2,64)"""
        # return w.T.reshape(emb_dim, num_kv_heads, head_dim)
        return w.T

    def convert_o_weight(w):
        """O权重转换: (896,896) -> transpose(1,0) -> (896,896) -> reshape -> (14,64,896)"""
        # return w.T.reshape(num_heads, head_dim, emb_dim)
        return w.T

    # 转换PyTorch权重
    converted_q = convert_q_weight(torch_q_weight)
    converted_k = convert_kv_weight(torch_k_weight)
    converted_v = convert_kv_weight(torch_v_weight)
    converted_o = convert_o_weight(torch_o_weight)

    print(f"\n转换后的PyTorch权重形状:")
    print(f"  q_proj: {converted_q.shape}")
    print(f"  k_proj: {converted_k.shape}")
    print(f"  v_proj: {converted_v.shape}")
    print(f"  o_proj: {converted_o.shape}")

    # 对比JAX和转换后的PyTorch权重
    def compare_weights(jax_w, torch_w, name):
        """对比权重"""
        jax_np = jax.device_get(jax_w)
        diff = jax_np - torch_w
        max_diff = abs(diff).max()
        mean_diff = abs(diff).mean()
        print(f"\n{name}权重对比:")
        print(f"  形状匹配: {jax_w.shape == torch_w.shape}")
        print(f"  最大差异: {max_diff:.6f}")
        print(f"  平均差异: {mean_diff:.6f}")
        print(f"  JAX范围: [{float(jax_np.min()):.3f}, {float(jax_np.max()):.3f}]")
        print(f"  PyTorch范围: [{float(torch_w.min()):.3f}, {float(torch_w.max()):.3f}]")

        # 如果差异很大，说明转换有问题
        if max_diff > 1e-5:
            print(f"  ⚠️ 权重差异很大！")
        else:
            print(f"  ✅ 权重匹配良好")

    compare_weights(jax_q_weight, converted_q, "Q")
    compare_weights(jax_k_weight, converted_k, "K")
    compare_weights(jax_v_weight, converted_v, "V")
    compare_weights(jax_o_weight, converted_o, "O")

if __name__ == "__main__":
    debug_actual_weights()