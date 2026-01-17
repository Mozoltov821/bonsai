#!/usr/bin/env python3
"""
逐层对比 JAX 和 PyTorch 版本的 MiMo Audio 模型

使用方法：
    python -m bonsai.models.mimo_audio.test.test_layer_comparison
"""

import os
import sys
import json
from typing import Dict, Any, Tuple
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

# 固定配置
MODEL_PATH = os.path.expanduser("~/.cache/modelscope/hub/models/XiaomiMiMo/MiMo-Audio-7B-Instruct")
BATCH_SIZE = 1
NUM_GROUPS = 4
ATOL = 1e-2  # 绝对误差容忍度（bfloat16 精度）
RTOL = 1e-2  # 相对误差容忍度

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)


@dataclass
class ComparisonResult:
    """对比结果"""
    layer_name: str
    max_diff: float
    mean_diff: float
    relative_error: float
    passed: bool
    jax_shape: tuple
    torch_shape: tuple
    jax_stats: Dict[str, float]
    torch_stats: Dict[str, float]


class LayerComparator:
    """逐层对比工具"""

    def __init__(self):
        self.results: list[ComparisonResult] = []
        self.jax_model = None
        self.torch_model = None

    def _print(self, message: str, level: str = "INFO"):
        """打印信息"""
        prefix = {
            "INFO": "ℹ️ ",
            "SUCCESS": "✅",
            "ERROR": "❌",
            "WARNING": "⚠️ ",
            "SECTION": "\n" + "=" * 80 + "\n",
        }.get(level, "")
        print(f"{prefix} {message}")

    def load_jax_model(self) -> bool:
        """加载 JAX 模型"""
        self._print("加载 JAX 模型", "SECTION")

        try:
            from bonsai.models.mimo_audio.modeling import MiMoAudioConfig, MiMoAudioArguments
            from bonsai.models.mimo_audio.params import create_model_with_weights
            from transformers import AutoTokenizer

            # 加载配置
            config_path = os.path.join(MODEL_PATH, "config.json")
            with open(config_path) as f:
                config_dict = json.load(f)

            # 创建配置
            config_kwargs = {k: v for k, v in config_dict.items()
                           if k in MiMoAudioConfig.__dataclass_fields__}
            config = MiMoAudioConfig(**config_kwargs)

            # 从 tokenizer 获取 special token IDs
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            args = MiMoAudioArguments(
                model_name_or_path=MODEL_PATH,
                sosp_idx=tokenizer.convert_tokens_to_ids("<|sosp|>"),
                eosp_idx=tokenizer.convert_tokens_to_ids("<|eosp|>"),
                sostm_idx=tokenizer.convert_tokens_to_ids("<|sostm|>"),
                eostm_idx=tokenizer.convert_tokens_to_ids("<|eostm|>"),
                eot_idx=tokenizer.convert_tokens_to_ids("<|eot|>"),
                empty_idx=tokenizer.convert_tokens_to_ids("<|empty|>"),
            )

            # 加载模型
            self.jax_model = create_model_with_weights(
                model_path=MODEL_PATH,
                config=config,
                args=args,
                rngs=nnx.Rngs(42),
                dtype=jnp.bfloat16,
                mesh=None,
            )

            self._print(f"JAX 模型加载成功", "SUCCESS")
            self._print(f"  - 词表大小: {config.vocab_size}")
            self._print(f"  - 隐藏层大小: {config.hidden_size}")
            self._print(f"  - 层数: {config.num_hidden_layers}")
            self._print(f"  - 音频通道: {config.audio_channels}")

            return True

        except Exception as e:
            self._print(f"JAX 模型加载失败: {type(e).__name__}: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return False

    def load_torch_model(self) -> bool:
        """加载 PyTorch 模型"""
        self._print("加载 PyTorch 模型", "SECTION")

        try:
            import torch
            from transformers import AutoTokenizer

            # 导入本地的 PyTorch MiMo Audio 实现
            pytorch_src_dir = os.path.join(project_root, "bonsai/models/mimo_audio/pytorch/src")
            if pytorch_src_dir not in sys.path:
                sys.path.insert(0, pytorch_src_dir)

            self._print(f"PyTorch 源码路径: {pytorch_src_dir}")

            from mimo_audio import MiMoAudioForCausalLM, MiMoAudioConfig, MiMoAudioArguments

            # 加载配置
            config_path = os.path.join(MODEL_PATH, "config.json")
            with open(config_path) as f:
                config_dict = json.load(f)

            # 创建配置
            config = MiMoAudioConfig(**config_dict)

            # 从 tokenizer 获取 special token IDs
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            args = MiMoAudioArguments(
                model_name_or_path=MODEL_PATH,
                sosp_idx=tokenizer.convert_tokens_to_ids("<|sosp|>"),
                eosp_idx=tokenizer.convert_tokens_to_ids("<|eosp|>"),
                sostm_idx=tokenizer.convert_tokens_to_ids("<|sostm|>"),
                eostm_idx=tokenizer.convert_tokens_to_ids("<|eostm|>"),
                eot_idx=tokenizer.convert_tokens_to_ids("<|eot|>"),
                empty_idx=tokenizer.convert_tokens_to_ids("<|empty|>"),
            )

            # 创建模型并加载权重
            self.torch_model = MiMoAudioForCausalLM(config, args)

            # 加载权重
            from safetensors import safe_open
            safetensors_files = []
            index_path = os.path.join(MODEL_PATH, "model.safetensors.index.json")

            if os.path.exists(index_path):
                with open(index_path) as f:
                    index = json.load(f)
                safetensors_files = sorted(set(index["weight_map"].values()))
            else:
                safetensors_files = ["model.safetensors"]

            state_dict = {}
            for shard_file in safetensors_files:
                shard_path = os.path.join(MODEL_PATH, shard_file)
                with safe_open(shard_path, framework="pt") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)

            # 加载到模型
            missing, unexpected = self.torch_model.load_state_dict(state_dict, strict=False)
            if missing:
                self._print(f"  缺少的权重: {len(missing)} 个", "WARNING")
            if unexpected:
                self._print(f"  多余的权重: {len(unexpected)} 个", "WARNING")

            self.torch_model.eval()
            self.torch_model = self.torch_model.to(torch.bfloat16)

            self._print(f"PyTorch 模型加载成功", "SUCCESS")
            self._print(f"  - 词表大小: {self.torch_model.config.vocab_size}")
            self._print(f"  - 隐藏层大小: {self.torch_model.config.hidden_size}")
            self._print(f"  - 层数: {self.torch_model.config.num_hidden_layers}")
            self._print(f"  - 音频通道: {self.torch_model.audio_channels}")

            return True

        except Exception as e:
            self._print(f"PyTorch 模型加载失败: {type(e).__name__}: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return False

    def create_test_input(self) -> Tuple[Any, Any]:
        """创建测试输入"""
        audio_channels = self.jax_model.audio_channels
        group_size = self.jax_model.group_size

        # 创建随机输入（使用固定种子保证一致性）
        np.random.seed(42)
        input_shape = (BATCH_SIZE, audio_channels + 1, NUM_GROUPS * group_size)

        # 使用小范围的随机整数（避免越界）
        input_array = np.random.randint(0, 1000, size=input_shape, dtype=np.int32)

        # JAX 输入
        jax_input = jnp.array(input_array)

        # PyTorch 输入 - 使用相同的形状
        import torch
        torch_input = torch.from_numpy(input_array).long()

        self._print(f"\n创建测试输入:")
        self._print(f"  - 批次大小: {BATCH_SIZE}")
        self._print(f"  - 音频通道: {audio_channels}")
        self._print(f"  - 分组大小: {group_size}")
        self._print(f"  - 分组数量: {NUM_GROUPS}")
        self._print(f"  - JAX 输入形状: {jax_input.shape}")
        self._print(f"  - PyTorch 输入形状: {torch_input.shape}")

        return jax_input, torch_input

    def compare_arrays(
        self,
        jax_array: jnp.ndarray,
        torch_array: Any,
        name: str,
    ) -> ComparisonResult:
        """对比两个数组"""
        import torch

        # 转换为 numpy
        jax_np = np.array(jax_array.astype(jnp.float32))
        torch_np = torch_array.detach().cpu().float().numpy()

        # 计算统计信息
        jax_stats = {
            "mean": float(jax_np.mean()),
            "std": float(jax_np.std()),
            "min": float(jax_np.min()),
            "max": float(jax_np.max()),
        }

        torch_stats = {
            "mean": float(torch_np.mean()),
            "std": float(torch_np.std()),
            "min": float(torch_np.min()),
            "max": float(torch_np.max()),
        }

        # 计算差异
        if jax_np.shape != torch_np.shape:
            self._print(f"  ❌ 形状不匹配: JAX {jax_np.shape} vs PyTorch {torch_np.shape}", "ERROR")
            return ComparisonResult(
                layer_name=name,
                max_diff=float('inf'),
                mean_diff=float('inf'),
                relative_error=float('inf'),
                passed=False,
                jax_shape=jax_np.shape,
                torch_shape=torch_np.shape,
                jax_stats=jax_stats,
                torch_stats=torch_stats,
            )

        diff = np.abs(jax_np - torch_np)
        max_diff = float(np.max(diff))
        mean_diff = float(np.mean(diff))

        # 计算相对误差
        denominator = np.maximum(np.abs(torch_np), 1e-8)
        relative_error = float(np.mean(diff / denominator))

        # 判断是否通过
        passed = np.allclose(jax_np, torch_np, atol=ATOL, rtol=RTOL)

        return ComparisonResult(
            layer_name=name,
            max_diff=max_diff,
            mean_diff=mean_diff,
            relative_error=relative_error,
            passed=passed,
            jax_shape=jax_np.shape,
            torch_shape=torch_np.shape,
            jax_stats=jax_stats,
            torch_stats=torch_stats,
        )

    def compare_embeddings(self, jax_input: jnp.ndarray, torch_input: Any) -> bool:
        """对比 embeddings 层"""
        self._print("\n对比 Embeddings", "SECTION")

        try:
            import torch

            # JAX: 获取文本 embeddings
            text_input_ids = jax_input[:, 0, ::self.jax_model.group_size]  # [B, T_groups]
            jax_text_embeds = self.jax_model.model.embedder.embedding[text_input_ids]

            # PyTorch: 获取文本 embeddings
            with torch.no_grad():
                text_token_ids = torch_input[:, 0, ::self.jax_model.group_size]
                torch_text_embeds = self.torch_model.model.embed_tokens(text_token_ids)

            # 对比
            result = self.compare_arrays(jax_text_embeds, torch_text_embeds, "text_embeddings")
            self.results.append(result)

            self._print(f"{'✅' if result.passed else '❌'} Text Embeddings:")
            self._print(f"  - 最大差异: {result.max_diff:.6f}")
            self._print(f"  - 平均差异: {result.mean_diff:.6f}")
            self._print(f"  - 相对误差: {result.relative_error:.6f}")

            return result.passed

        except Exception as e:
            self._print(f"Embeddings 对比失败: {type(e).__name__}: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return False

    def compare_speech_embeddings(self, jax_input: jnp.ndarray) -> bool:
        """对比语音 embeddings"""
        self._print("\n对比语音 Embeddings", "SECTION")

        try:
            import torch

            # 提取语音 token IDs
            batch_size = jax_input.shape[0]
            audio_channels = self.jax_model.audio_channels
            group_size = self.jax_model.group_size

            speech_input_ids = jax_input[:, 1:, :].reshape(
                batch_size, audio_channels, -1, group_size
            ).transpose(0, 2, 1, 3)  # [B, T_groups, audio_channels, group_size]

            # 逐通道对比 embeddings
            all_passed = True

            for ch in range(min(3, audio_channels)):  # 只对比前3个通道
                # JAX
                jax_ch_ids = speech_input_ids[:, :, ch, :]
                jax_ch_embeds = self.jax_model.speech_embeddings[ch](jax_ch_ids)

                # PyTorch
                with torch.no_grad():
                    torch_ch_ids = torch.from_numpy(np.array(jax_ch_ids)).long()
                    torch_ch_embeds = self.torch_model.speech_embeddings[ch](torch_ch_ids)

                # 对比
                result = self.compare_arrays(
                    jax_ch_embeds,
                    torch_ch_embeds,
                    f"speech_embedding_channel_{ch}"
                )
                self.results.append(result)

                self._print(f"  {'✅' if result.passed else '❌'} 通道 {ch}:")
                self._print(f"    - 最大差异: {result.max_diff:.6f}")
                self._print(f"    - 平均差异: {result.mean_diff:.6f}")

                all_passed = all_passed and result.passed

            return all_passed

        except Exception as e:
            self._print(f"语音 Embeddings 对比失败: {type(e).__name__}: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return False

    def compare_forward_pass(self, jax_input: jnp.ndarray, torch_input: Any) -> bool:
        """对比完整的前向传播"""
        self._print("\n对比前向传播", "SECTION")

        try:
            import torch

            batch_size = jax_input.shape[0]
            num_groups = jax_input.shape[2] // self.jax_model.group_size

            # JAX 前向传播
            jax_cache = self.jax_model.model.init_cache(
                self.jax_model.qwen2_config,
                batch_size,
                num_groups,
                generate_steps=0,
                dtype=jnp.bfloat16,
            )

            jax_text_logits, jax_local_hidden = self.jax_model.forward(
                jax_input,
                jax_cache,
                pad_id=0,
            )

            # PyTorch 前向传播
            with torch.no_grad():
                # 创建 attention_mask 和 position_ids
                attention_mask = torch.ones(batch_size, num_groups, dtype=torch.long)
                position_ids = torch.arange(num_groups, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

                # 调用 forward
                torch_output = self.torch_model.forward(
                    input_ids=torch_input,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    cache_position=None,
                )

                torch_text_logits = torch_output.text_logits
                torch_local_hidden = torch_output.local_hidden_states

            # 对比文本 logits
            result_logits = self.compare_arrays(
                jax_text_logits,
                torch_text_logits,
                "text_logits"
            )
            self.results.append(result_logits)

            self._print(f"\n{'✅' if result_logits.passed else '❌'} 文本 Logits:")
            self._print(f"  - 形状: JAX {result_logits.jax_shape} vs PyTorch {result_logits.torch_shape}")
            self._print(f"  - 最大差异: {result_logits.max_diff:.6f}")
            self._print(f"  - 平均差异: {result_logits.mean_diff:.6f}")

            # 对比 local hidden states
            result_hidden = self.compare_arrays(
                jax_local_hidden,
                torch_local_hidden,
                "local_hidden_states"
            )
            self.results.append(result_hidden)

            self._print(f"\n{'✅' if result_hidden.passed else '❌'} Local Hidden States:")
            self._print(f"  - 形状: JAX {result_hidden.jax_shape} vs PyTorch {result_hidden.torch_shape}")
            self._print(f"  - 最大差异: {result_hidden.max_diff:.6f}")
            self._print(f"  - 平均差异: {result_hidden.mean_diff:.6f}")

            # 打印 top-k predictions 对比
            if not np.isnan(result_logits.jax_stats['mean']):
                self._print(f"\n  Top-5 Predictions 对比:")

                # JAX top-5
                jax_probs = jax.nn.softmax(jax_text_logits[0, 0].astype(jnp.float32))
                jax_top_indices = jnp.argsort(jax_probs)[-5:][::-1]
                jax_top_probs = jax_probs[jax_top_indices]

                self._print(f"    JAX:")
                for i, (idx, prob) in enumerate(zip(jax_top_indices, jax_top_probs)):
                    self._print(f"      {i+1}. Token {int(idx)}: {float(prob):.6f}")

                # PyTorch top-5
                torch_probs = torch.softmax(torch_text_logits[0, 0].float(), dim=-1)
                torch_top_probs, torch_top_indices = torch.topk(torch_probs, 5)

                self._print(f"    PyTorch:")
                for i, (idx, prob) in enumerate(zip(torch_top_indices, torch_top_probs)):
                    self._print(f"      {i+1}. Token {int(idx)}: {float(prob):.6f}")
            else:
                self._print(f"\n  ⚠️ JAX 输出包含 NaN，跳过 top-k 对比", "WARNING")

            return result_logits.passed and result_hidden.passed

        except Exception as e:
            self._print(f"前向传播对比失败: {type(e).__name__}: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return False

    def print_summary(self):
        """打印对比总结"""
        self._print("\n对比总结", "SECTION")

        if not self.results:
            self._print("没有对比结果", "WARNING")
            return

        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)

        self._print(f"通过: {passed_count}/{total_count} 层")

        # 详细结果表格
        self._print(f"\n{'层名称':<30} {'最大差异':<15} {'平均差异':<15} {'相对误差':<15} {'状态':<10}")
        self._print("-" * 85)

        for result in self.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            self._print(
                f"{result.layer_name:<30} "
                f"{result.max_diff:<15.6f} "
                f"{result.mean_diff:<15.6f} "
                f"{result.relative_error:<15.6f} "
                f"{status:<10}"
            )

        # 失败的层详细信息
        failed_layers = [r for r in self.results if not r.passed]
        if failed_layers:
            self._print(f"\n失败的层详细信息:", "WARNING")
            for result in failed_layers:
                self._print(f"\n  {result.layer_name}:")
                self._print(f"    JAX 形状: {result.jax_shape}")
                self._print(f"    PyTorch 形状: {result.torch_shape}")
                self._print(f"    JAX 统计: {result.jax_stats}")
                self._print(f"    PyTorch 统计: {result.torch_stats}")
                self._print(f"    最大差异: {result.max_diff:.6f} (阈值: {ATOL})")
                self._print(f"    相对误差: {result.relative_error:.6f} (阈值: {RTOL})")

        if passed_count == total_count:
            self._print("\n🎉 所有层对比通过！", "SUCCESS")
        else:
            self._print(f"\n⚠️  {total_count - passed_count} 层对比失败", "WARNING")

    def run_comparison(self) -> bool:
        """运行完整对比"""
        self._print("开始逐层对比", "SECTION")
        self._print(f"模型路径: {MODEL_PATH}")

        # 1. 加载模型
        if not self.load_jax_model():
            return False

        if not self.load_torch_model():
            return False

        # 2. 创建测试输入
        jax_input, torch_input = self.create_test_input()

        # 3. 对比各个部分
        tests = [
            ("Embeddings", lambda: self.compare_embeddings(jax_input, torch_input)),
            ("语音 Embeddings", lambda: self.compare_speech_embeddings(jax_input)),
            ("前向传播", lambda: self.compare_forward_pass(jax_input, torch_input)),
        ]

        all_passed = True
        for test_name, test_func in tests:
            try:
                passed = test_func()
                all_passed = all_passed and passed
            except Exception as e:
                self._print(f"{test_name} 测试失败: {e}", "ERROR")
                all_passed = False

        # 4. 打印总结
        self.print_summary()

        return all_passed


def main():
    """主函数"""
    comparator = LayerComparator()
    success = comparator.run_comparison()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
