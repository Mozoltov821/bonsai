#!/usr/bin/env python3
"""调试其他权重（embeddings, lm_head等）"""

import jax
import torch
import safetensors
from etils import epath
from huggingface_hub import snapshot_download

from bonsai.models.qwen2 import modeling, params

def debug_other_weights():
    """调试embeddings和lm_head权重"""
    print("=== 调试其他权重 ===")

    # 加载模型
    model_path = snapshot_download("Qwen/Qwen2-0.5B")
    config = modeling.ModelConfig.qwen2_0_5b(use_sharding=False)
    jax_model = params.create_model_from_safe_tensors(model_path, config)

    # 从原始safetensors加载PyTorch权重
    files = list(epath.Path(model_path).expanduser().glob("*.safetensors"))
    with safetensors.safe_open(files[0], framework="pt") as sf:
        torch_embeddings = sf.get_tensor("model.embed_tokens.weight").float().numpy()

        # 尝试加载lm_head权重，如果不存在则可能使用权重绑定
        try:
            torch_lm_head = sf.get_tensor("lm_head.weight").float().numpy()
            has_separate_lm_head = True
        except:
            torch_lm_head = None
            has_separate_lm_head = False
            print("lm_head.weight不存在，可能使用权重绑定")

        # 检查是否存在final norm权重
        try:
            torch_final_norm = sf.get_tensor("model.norm.weight").float().numpy()
        except Exception as e:
            print(f"Final norm权重加载失败: {e}")
            torch_final_norm = None

    print(f"\nPyTorch原始权重形状:")
    print(f"  embeddings: {torch_embeddings.shape}")
    if has_separate_lm_head:
        print(f"  lm_head: {torch_lm_head.shape}")
    else:
        print(f"  lm_head: 使用权重绑定")
    if torch_final_norm is not None:
        print(f"  final_norm: {torch_final_norm.shape}")

    # 从JAX模型提取权重
    jax_embeddings = jax_model.embedder.embedding[...]
    jax_lm_head = jax_model.lm_head.w[...]

    # 检查是否存在final norm权重
    try:
        jax_final_norm = jax_model.final_norm.scale[...]
    except Exception as e:
        print(f"JAX final norm权重访问失败: {e}")
        jax_final_norm = None

    print(f"\nJAX模型权重形状:")
    print(f"  embeddings: {jax_embeddings.shape}")
    print(f"  lm_head: {jax_lm_head.shape}")
    if jax_final_norm is not None:
        print(f"  final_norm: {jax_final_norm.shape}")

    # 对比权重的函数
    def compare_weights(jax_w, torch_w, name):
        jax_np = jax.device_get(jax_w)

        # 对于lm_head，需要transpose
        if name == "lm_head":
            torch_w = torch_w.T  # (vocab, emb) -> (emb, vocab)

        diff = jax_np - torch_w
        max_diff = abs(diff).max()
        mean_diff = abs(diff).mean()
        print(f"\n{name}权重对比:")
        print(f"  形状匹配: {jax_w.shape == torch_w.shape}")
        print(f"  最大差异: {max_diff:.6f}")
        print(f"  平均差异: {mean_diff:.6f}")
        print(f"  JAX范围: [{float(jax_np.min()):.3f}, {float(jax_np.max()):.3f}]")
        print(f"  PyTorch范围: [{float(torch_w.min()):.3f}, {float(torch_w.max()):.3f}]")

        if max_diff > 1e-5:
            print(f"  ⚠️ 权重差异很大！")
            return False
        else:
            print(f"  ✅ 权重匹配良好")
            return True

    # 对比各个权重
    embeddings_ok = compare_weights(jax_embeddings, torch_embeddings, "embeddings")

    if has_separate_lm_head:
        lm_head_ok = compare_weights(jax_lm_head, torch_lm_head, "lm_head")
    else:
        # 检查JAX的lm_head是否等于embeddings.T
        expected_lm_head = torch_embeddings.T  # (vocab, emb) -> (emb, vocab)
        lm_head_ok = compare_weights(jax_lm_head, expected_lm_head, "lm_head(绑定)")

    if torch_final_norm is not None and jax_final_norm is not None:
        final_norm_ok = compare_weights(jax_final_norm, torch_final_norm, "final_norm")
    else:
        print("\n⚠️ Final norm权重未找到或无法比较")
        final_norm_ok = True

    # 检查config中的tie_word_embeddings设置
    print(f"\n=== 配置检查 ===")
    print(f"config.tie_word_embeddings: {config.tie_word_embeddings}")

    if config.tie_word_embeddings:
        # 检查是否embeddings和lm_head权重被绑定
        expected_lm_head = jax_embeddings.T  # embeddings转置应该等于lm_head
        lm_head_diff = abs(jax_lm_head - expected_lm_head).max()
        print(f"embeddings.T与lm_head的差异: {lm_head_diff:.6f}")
        if lm_head_diff < 1e-5:
            print("✅ 权重绑定正确")
        else:
            print("⚠️ 权重绑定可能有问题")

    return embeddings_ok and lm_head_ok and final_norm_ok

if __name__ == "__main__":
    debug_other_weights()