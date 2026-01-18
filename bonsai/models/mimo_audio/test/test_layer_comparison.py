"""
MiMo Audio 模型层级一致性验证脚本

该脚本用于验证 PyTorch 版本和 JAX 版本的 MiMo Audio 模型的逐层输入输出一致性。

主要功能:
1. 加载 JAX 和 PyTorch 模型(使用相同权重)
2. 生成相同的测试输入
3. 捕获每一层的中间输出
4. 逐层比较数值差异
5. 生成详细的对比报告

使用方法:
    python -m bonsai.models.mimo_audio.test.test_layer_comparison

    # 或指定模型路径
    python -m bonsai.models.mimo_audio.test.test_layer_comparison --model_path /path/to/model
"""

import os
import sys
import json
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
from flax import nnx

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))


@dataclass
class ComparisonConfig:
    """对比配置"""
    # 容差设置 (考虑 bfloat16 精度)
    rtol: float = 1e-2  # 相对容差 1%
    atol: float = 1e-3  # 绝对容差

    # 输入配置
    batch_size: int = 2
    num_groups: int = 4
    audio_channels: int = 8
    group_size: int = 4
    seed: int = 42

    # 验证范围
    compare_embeddings: bool = True
    compare_main_transformer: bool = True
    compare_local_transformer: bool = True
    compare_input_local_transformer: bool = True
    compare_projections: bool = True

    # 输出选项
    save_activations: bool = True
    output_dir: str = "comparison_results"
    verbose: bool = True

    # 模型路径
    model_path: str = os.path.expanduser(
        "~/.cache/modelscope/hub/models/XiaomiMiMo/MiMo-Audio-7B-Instruct"
    )


@dataclass
class ComparisonResult:
    """单层对比结果"""
    layer_name: str
    passed: bool
    max_abs_diff: float = 0.0
    max_rel_diff: float = 0.0
    mean_abs_diff: float = 0.0
    jax_stats: Optional[Dict] = None
    torch_stats: Optional[Dict] = None
    jax_shape: Optional[Tuple] = None
    torch_shape: Optional[Tuple] = None
    error_type: Optional[str] = None


class TestInputGenerator:
    """测试输入生成器"""

    def __init__(self, config: ComparisonConfig):
        self.config = config
        # 语音词汇表大小（对应每个音频通道）
        self.speech_vocab_sizes = [1025, 1025, 129, 129, 129, 129, 129, 129]
        # 文本词汇表大小
        self.text_vocab_size = 151680

    def generate_input_ids(self) -> Tuple[jnp.ndarray, torch.Tensor]:
        """
        生成相同的测试 input_ids

        为每个通道生成合适范围内的 token IDs：
        - 通道 0 (文本): 0 到 text_vocab_size
        - 通道 1-8 (音频): 根据 speech_vocab_sizes 生成

        Returns:
            (jax_input, torch_input): JAX 和 PyTorch 格式的输入
        """
        np.random.seed(self.config.seed)

        B = self.config.batch_size
        T = self.config.num_groups * self.config.group_size

        # 初始化输出数组
        input_ids_np = np.zeros(
            (B, self.config.audio_channels + 1, T),
            dtype=np.int32
        )

        # 为每个批次生成 token
        for b in range(B):
            # 通道 0: 文本 tokens (范围 0 到 text_vocab_size-1)
            input_ids_np[b, 0, :] = np.random.randint(
                0, self.text_vocab_size, size=T, dtype=np.int32
            )

            # 通道 1-8: 语音 tokens (根据各自的词汇表大小)
            for ch in range(self.config.audio_channels):
                vocab_size = self.speech_vocab_sizes[ch]
                input_ids_np[b, ch + 1, :] = np.random.randint(
                    0, vocab_size, size=T, dtype=np.int32
                )

        # 转换为两个框架的格式
        jax_input = jnp.array(input_ids_np)
        torch_input = torch.from_numpy(input_ids_np)

        return jax_input, torch_input

    def compute_checksum(self, arr: np.ndarray) -> str:
        """计算数组的校验和"""
        return hashlib.md5(arr.tobytes()).hexdigest()[:8]


class TensorComparator:
    """张量对比器"""

    def __init__(self, config: ComparisonConfig):
        self.config = config

    def compare(
        self,
        jax_tensor: jnp.ndarray,
        torch_tensor: Union[torch.Tensor, np.ndarray],
        layer_name: str
    ) -> ComparisonResult:
        """
        对比两个张量

        Args:
            jax_tensor: JAX 张量
            torch_tensor: PyTorch 张量或 NumPy 数组
            layer_name: 层名称

        Returns:
            ComparisonResult: 对比结果
        """
        # 转换为 NumPy float32
        if isinstance(torch_tensor, torch.Tensor):
            torch_np = torch_tensor.detach().cpu().float().numpy()
        else:
            torch_np = np.array(torch_tensor, dtype=np.float32)

        jax_np = np.array(jax_tensor, dtype=np.float32)

        # 形状检查
        if jax_np.shape != torch_np.shape:
            return ComparisonResult(
                layer_name=layer_name,
                passed=False,
                error_type='shape_mismatch',
                jax_shape=jax_np.shape,
                torch_shape=torch_np.shape
            )

        # 计算差异
        abs_diff = np.abs(jax_np - torch_np)

        max_abs_diff = float(np.max(abs_diff))
        mean_abs_diff = float(np.mean(abs_diff))

        # 计算相对误差 - 只在值足够大时才有意义
        # 使用 scale 来避免除以接近 0 的数
        scale = np.maximum(np.abs(jax_np), np.abs(torch_np))
        # 只计算 scale > 0.01 的位置的相对误差
        valid_mask = scale > 0.01
        if np.any(valid_mask):
            rel_diff = abs_diff[valid_mask] / scale[valid_mask]
            max_rel_diff = float(np.max(rel_diff))
        else:
            # 如果所有值都很小，相对误差没有意义
            max_rel_diff = 0.0

        # 判断是否通过
        # 使用更智能的逻辑：
        # 1. 如果绝对差异很小（< atol），直接通过
        # 2. 否则检查相对误差
        passed = (
            max_abs_diff <= self.config.atol or
            (max_rel_diff <= self.config.rtol and max_abs_diff < 10.0)  # 相对误差小且绝对差异不太大
        )

        return ComparisonResult(
            layer_name=layer_name,
            passed=passed,
            max_abs_diff=max_abs_diff,
            max_rel_diff=max_rel_diff,
            mean_abs_diff=mean_abs_diff,
            jax_stats=self._compute_stats(jax_np),
            torch_stats=self._compute_stats(torch_np),
            jax_shape=jax_np.shape,
            torch_shape=torch_np.shape
        )

    @staticmethod
    def _compute_stats(arr: np.ndarray) -> Dict:
        """计算张量统计信息"""
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'has_nan': bool(np.any(np.isnan(arr))),
            'has_inf': bool(np.any(np.isinf(arr)))
        }


class ModelLoader:
    """模型加载器"""

    def __init__(self, config: ComparisonConfig):
        self.config = config
        self.model_path = config.model_path

    def load_jax_model(self):
        """加载 JAX 模型"""
        from bonsai.models.mimo_audio.params import create_model_with_weights
        from bonsai.models.mimo_audio.modeling import MiMoAudioConfig, MiMoAudioArguments

        # 加载配置
        config_path = os.path.join(self.model_path, "config.json")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        # JAX MiMoAudioConfig 是 @dataclass，只接受定义的字段
        # 使用白名单方式：只提取 JAX MiMoAudioConfig 支持的字段
        from dataclasses import fields
        jax_supported_fields = {f.name for f in fields(MiMoAudioConfig)}

        # 从 config.json 中提取 JAX 支持的字段
        mimo_config_fields = {k: v for k, v in config_dict.items()
                              if k in jax_supported_fields}

        # 创建 MiMo 配置（只使用 JAX 支持的字段，无默认值）
        mimo_config = MiMoAudioConfig(**mimo_config_fields)

        # 创建参数（使用默认值，因为这些字段可能不在 config.json 中）
        args = MiMoAudioArguments(
            model_name_or_path=self.model_path,
            sosp_idx=config_dict.get('sosp_idx', 151649),
            eosp_idx=config_dict.get('eosp_idx', 151650),
            sostm_idx=config_dict.get('sostm_idx', 151651),
            eostm_idx=config_dict.get('eostm_idx', 151652),
            eot_idx=config_dict.get('eot_idx', 151643),
            empty_idx=config_dict.get('empty_idx', 151648)
        )

        # 加载模型
        model = create_model_with_weights(
            model_path=self.model_path,
            config=mimo_config,
            args=args,
            rngs=nnx.Rngs(self.config.seed),
            mesh=None
        )

        return model, mimo_config, args

    def load_pytorch_model(self):
        """加载 PyTorch 模型"""
        # 直接导入 PyTorch 实现 - 使用绝对路径
        from bonsai.models.mimo_audio.pytorch.src.mimo_audio.modeling_mimo_audio import (
            MiMoAudioForCausalLM, MiMoAudioConfig, MiMoAudioArguments
        )

        # 加载配置
        config_path = os.path.join(self.model_path, "config.json")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        # PyTorch MiMoAudioConfig 继承自 PretrainedConfig，可能也有特有字段
        # 过滤掉已知的不兼容字段
        torch_incompatible_fields = {
            'add_input_local_transformer', 'add_speech_sosp_eosp'
        }
        config_dict_filtered = {k: v for k, v in config_dict.items()
                                if k not in torch_incompatible_fields}

        # 创建配置（直接从 config.json，无默认值）
        config = MiMoAudioConfig(**config_dict_filtered)

        # 创建参数（使用默认值，因为这些字段可能不在 config.json 中）
        args = MiMoAudioArguments(
            model_name_or_path=self.model_path,
            sosp_idx=config_dict.get('sosp_idx', 151649),
            eosp_idx=config_dict.get('eosp_idx', 151650),
            sostm_idx=config_dict.get('sostm_idx', 151651),
            eostm_idx=config_dict.get('eostm_idx', 151652),
            eot_idx=config_dict.get('eot_idx', 151643),
            empty_idx=config_dict.get('empty_idx', 151648)
        )

        # 创建模型
        model = MiMoAudioForCausalLM(config, args)

        # 加载权重
        import safetensors.torch

        # 检查是否有分片的权重文件
        weights_path = os.path.join(self.model_path, "model.safetensors")
        if not os.path.exists(weights_path):
            # 尝试加载分片权重
            import glob
            shard_files = sorted(glob.glob(os.path.join(self.model_path, "model-*.safetensors")))
            if shard_files:
                state_dict = {}
                for shard_file in shard_files:
                    state_dict.update(safetensors.torch.load_file(shard_file))
            else:
                raise FileNotFoundError(f"未找到权重文件: {weights_path}")
        else:
            state_dict = safetensors.torch.load_file(weights_path)

        model.load_state_dict(state_dict, strict=False)

        # 设置为评估模式
        model.eval()
        model = model.to(torch.bfloat16)

        return model, config, args

    def verify_weight_consistency(self, jax_model, torch_model) -> bool:
        """验证权重一致性"""
        print("  检查嵌入层权重...")

        # 比较文本嵌入 - 使用新的 API
        jax_text_emb = np.array(jax_model.model.embedder.embedding[...])
        torch_text_emb = torch_model.model.embed_tokens.weight.detach().cpu().float().numpy()

        diff = np.abs(jax_text_emb - torch_text_emb).max()
        if diff > 1e-5:
            print(f"  ⚠ 文本嵌入权重不一致: max_diff={diff:.6f}")
            return False

        print("  检查第一层注意力权重...")
        # 比较第一层的 Q 权重 - 使用 q_proj 而不是 wq
        jax_q = np.array(jax_model.model.layers[0].attn.q_proj.kernel[...])
        torch_q = torch_model.model.layers[0].self_attn.q_proj.weight.detach().cpu().float().numpy()

        diff = np.abs(jax_q.T - torch_q).max()  # JAX 使用转置
        if diff > 1e-5:
            print(f"  ⚠ 第一层 Q 权重不一致: max_diff={diff:.6f}")
            return False

        print("  ✅ 嵌入层权重一致")
        print("  ✅ 第一层注意力权重一致")
        return True

    def compare_configs(self, jax_config, torch_config) -> bool:
        """比对两个配置是否一致"""
        print("\n" + "=" * 70)
        print("配置对比")
        print("=" * 70)

        # 关键配置字段
        key_fields = [
            'vocab_size', 'hidden_size', 'num_hidden_layers',
            'num_attention_heads', 'num_key_value_heads',
            'intermediate_size', 'max_position_embeddings',
            'rope_theta', 'head_dim', 'group_size', 'audio_channels',
            'local_dim', 'local_layers', 'local_attn_heads',
            'local_ffn_dim', 'input_local_layers', 'input_local_dim',
            'input_full_attention', 'speech_vocab_size',
            'speech_zeroemb_idx', 'delay_pattern'
        ]

        all_match = True

        for field in key_fields:
            jax_val = getattr(jax_config, field, None)
            torch_val = getattr(torch_config, field, None)

            if jax_val != torch_val:
                print(f"❌ {field:30s}: JAX={jax_val}, PyTorch={torch_val}")
                all_match = False
            else:
                print(f"✅ {field:30s}: {jax_val}")

        print("=" * 70)

        if all_match:
            print("✅ 所有配置一致\n")
        else:
            print("❌ 配置不一致！请检查上述差异\n")

        return all_match


class JAXLayerCapture:
    """JAX 层捕获器"""

    def __init__(self, model, config: ComparisonConfig):
        self.model = model
        self.config = config
        self.activations = {}

    def capture(self, name: str, value: jnp.ndarray):
        """捕获激活值"""
        self.activations[name] = {
            'value': value,
            'shape': value.shape,
            'dtype': str(value.dtype)
        }

    def forward_with_capture(
        self,
        input_ids: jnp.ndarray,
        pad_id: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
        """
        执行前向传播并捕获中间层输出

        Args:
            input_ids: [B, audio_channels+1, T]
            pad_id: padding ID

        Returns:
            (text_logits, local_hidden, activations)
        """
        B = input_ids.shape[0]

        # 1. 提取文本和语音 tokens
        text_input_ids = input_ids[:, 0, ::self.model.group_size]  # [B, T_groups]
        speech_input_ids = input_ids[:, 1:, :].reshape(
            B, self.model.audio_channels, -1, self.model.group_size
        ).transpose(0, 2, 1, 3)  # [B, T_groups, audio_channels, group_size]

        is_speech = text_input_ids == self.model.args.empty_idx

        # 2. 文本嵌入
        text_embeds = self.model.model.embedder(text_input_ids)
        self.capture('text_embeddings', text_embeds)

        # 3. 语音嵌入 (累加所有通道)
        speech_embeds = jnp.zeros(
            (B, is_speech.shape[1], self.model.group_size, self.model.config.input_local_dim),
            dtype=jnp.bfloat16
        )

        for idx in range(self.model.audio_channels):
            cur_empty = self.model.speech_empty_ids[idx]
            cur_embed = self.model.speech_embeddings[idx]
            cur_speech_ids = speech_input_ids[:, :, idx, :]
            cur_speech_embeds = cur_embed(cur_speech_ids)

            # Mask 空 tokens
            cur_mask = cur_speech_ids == cur_empty
            cur_speech_embeds = cur_speech_embeds * ~cur_mask[..., None]

            speech_embeds = speech_embeds + cur_speech_embeds
            self.capture(f'speech_embedding_ch{idx}', cur_speech_embeds)

        # Mask 非语音位置
        speech_embeds = speech_embeds * is_speech[:, :, None, None]
        self.capture('speech_embeddings_combined', speech_embeds)

        # 4. Input Local Transformer
        if self.config.compare_input_local_transformer:
            B_orig, T_groups, group_size, hidden_size = speech_embeds.shape
            speech_embeds_flat = speech_embeds.reshape(B_orig * T_groups, group_size, hidden_size).astype(jnp.bfloat16)
            segment_ids = jnp.ones((B_orig * T_groups, group_size), dtype=jnp.int32)

            # 初始化 cache
            cache = self.model.input_local_transformer.init_cache(
                self.model.input_local_qwen2_config,
                B_orig * T_groups,
                group_size,
                0,
                jnp.bfloat16
            )

            x = speech_embeds_flat
            for i, layer in enumerate(self.model.input_local_transformer.layers):
                x = layer(x, cache[i], segment_ids)
                self.capture(f'input_local_layer_{i}_output', x)

            x = self.model.input_local_transformer.final_norm(x)
            speech_embeds = x.reshape(B_orig, T_groups, group_size, hidden_size)

        speech_embeds = speech_embeds * is_speech[:, :, None, None]
        self.capture('input_local_transformer_output', speech_embeds)

        # 5. Speech group downcast
        T_groups = speech_embeds.shape[1]
        speech_grouped_embeds = self.model.speech_group_downcast(
            speech_embeds.reshape(B, T_groups, -1)
        )
        self.capture('speech_group_downcast_output', speech_grouped_embeds)

        # 6. 组合嵌入
        combined_embeds = jnp.where(
            is_speech[:, :, None],
            speech_grouped_embeds,
            text_embeds
        )
        self.capture('combined_embeddings', combined_embeds)

        # 7. 主 Transformer
        cache = self.model.model.init_cache(
            self.model.qwen2_config,
            B,
            T_groups,
            0,
            jnp.bfloat16
        )

        segment_ids = jnp.ones((B, T_groups), dtype=jnp.int32)
        x = combined_embeds

        for i, layer in enumerate(self.model.model.layers):
            x = layer(x, cache[i], segment_ids)
            if self.config.compare_main_transformer:
                self.capture(f'main_layer_{i}_output', x)

        x = self.model.model.final_norm(x)
        self.capture('final_norm_output', x)

        # 8. 文本 logits
        text_logits = self.model.model.lm_head(x[:, -1:, :])
        self.capture('text_logits', text_logits)

        # 9. Local transformer 准备
        local_hidden = self.model.hidden_states_downcast(x[:, -1:, :])
        self.capture('hidden_states_downcast_output', local_hidden)

        # 10. Local transformer (如果需要)
        if self.config.compare_local_transformer:
            local_cache = self.model.local_transformer.init_cache(
                self.model.local_qwen2_config,
                B,
                1,
                0,
                jnp.bfloat16
            )

            local_x = local_hidden
            local_segment_ids = jnp.ones((B, 1), dtype=jnp.int32)

            for i, layer in enumerate(self.model.local_transformer.layers):
                local_x = layer(local_x, local_cache[i], local_segment_ids)
                self.capture(f'local_layer_{i}_output', local_x)

            local_x = self.model.local_transformer.final_norm(local_x)

            # LM heads
            for i in range(self.model.audio_channels):
                lm_head_out = self.model.local_transformer_lm_heads[i](local_x)
                self.capture(f'local_lm_head_ch{i}', lm_head_out)

        return text_logits, local_hidden, self.activations


class PyTorchHookManager:
    """PyTorch Hook 管理器"""

    def __init__(self, model, config: ComparisonConfig):
        self.model = model
        self.config = config
        self.hooks = {}
        self.hook_handles = []
        self.activations = {}

    def register_hooks(self):
        """注册所有 hooks"""
        # 文本嵌入
        if self.config.compare_embeddings:
            self._register_hook('text_embeddings', self.model.model.embed_tokens)

        # 语音嵌入
        if self.config.compare_embeddings:
            for i, emb in enumerate(self.model.speech_embeddings):
                # 需要特殊处理，因为 embedding 不会自动触发 hook
                pass

        # Input local transformer
        if self.config.compare_input_local_transformer and self.model.input_local_transformer:
            for i, layer in enumerate(self.model.input_local_transformer.layers):
                self._register_hook(f'input_local_layer_{i}_output', layer)

        # 主 transformer
        if self.config.compare_main_transformer:
            for i, layer in enumerate(self.model.model.layers):
                self._register_hook(f'main_layer_{i}_output', layer)

        # Local transformer
        if self.config.compare_local_transformer:
            for i, layer in enumerate(self.model.local_transformer.layers):
                self._register_hook(f'local_layer_{i}_output', layer)

    def _register_hook(self, name: str, module: nn.Module):
        """注册单个 hook"""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                value = output[0]
            else:
                value = output

            self.activations[name] = {
                'value': value.detach().cpu().float().numpy(),
                'shape': tuple(value.shape),
                'dtype': str(value.dtype)
            }

        handle = module.register_forward_hook(hook_fn)
        self.hook_handles.append(handle)

    def remove_hooks(self):
        """移除所有 hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def capture_manual(self, name: str, value: torch.Tensor):
        """手动捕获激活值"""
        self.activations[name] = {
            'value': value.detach().cpu().float().numpy(),
            'shape': tuple(value.shape),
            'dtype': str(value.dtype)
        }


class LayerComparisonRunner:
    """层级对比执行器"""

    def __init__(self, config: ComparisonConfig):
        self.config = config
        self.loader = ModelLoader(config)
        self.input_gen = TestInputGenerator(config)
        self.comparator = TensorComparator(config)

        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)

    def run(self) -> Dict:
        """执行完整的层级对比"""
        print("=" * 70)
        print("MiMo Audio 层级一致性验证")
        print("=" * 70)

        # 1. 加载模型
        print("\n[1/7] 加载 JAX 模型...")
        start_time = time.time()
        jax_model, jax_config, jax_args = self.loader.load_jax_model()
        jax_load_time = time.time() - start_time
        print(f"✅ JAX 模型加载成功 ({jax_load_time:.1f}s)")

        print("\n[2/7] 加载 PyTorch 模型...")
        start_time = time.time()
        torch_model, torch_config, torch_args = self.loader.load_pytorch_model()
        torch_load_time = time.time() - start_time
        print(f"✅ PyTorch 模型加载成功 ({torch_load_time:.1f}s)")

        # 2. 比对配置
        print("\n[3/7] 比对配置...")
        if not self.loader.compare_configs(jax_config, torch_config):
            print("⚠️ 警告：配置不一致，测试结果可能不可靠")
            response = input("是否继续？(y/n): ")
            if response.lower() != 'y':
                print("测试中止")
                return {}

        # 3. 验证权重一致性
        print("\n[4/7] 验证权重一致性...")
        if self.loader.verify_weight_consistency(jax_model, torch_model):
            print("✅ 权重验证通过")
        else:
            print("⚠ 权重可能不一致，但继续测试")

        # 4. 生成测试输入
        print("\n[5/7] 生成测试输入...")
        jax_input, torch_input = self.input_gen.generate_input_ids()
        checksum = self.input_gen.compute_checksum(np.array(jax_input))
        print(f"✅ 输入形状: {jax_input.shape}")
        print(f"✅ 校验和: 0x{checksum}")

        # 5. 运行 JAX 前向传播
        print("\n[6/7] 运行 JAX 前向传播...")
        start_time = time.time()
        jax_capture = JAXLayerCapture(jax_model, self.config)
        jax_logits, jax_local, jax_activations = jax_capture.forward_with_capture(
            jax_input, pad_id=0
        )
        jax_runtime = time.time() - start_time
        print(f"✅ 捕获 {len(jax_activations)} 个 JAX 层输出 ({jax_runtime:.1f}s)")

        # 6. 运行 PyTorch 前向传播
        print("\n[7/7] 运行 PyTorch 前向传播...")
        start_time = time.time()
        torch_activations = self._run_pytorch_forward(
            torch_model, torch_input, torch_args
        )
        torch_runtime = time.time() - start_time
        print(f"✅ 捕获 {len(torch_activations)} 个 PyTorch 层输出 ({torch_runtime:.1f}s)")

        # 7. 逐层对比
        print("\n[8/8] 逐层对比...")
        results = []

        for layer_name in sorted(jax_activations.keys()):
            if layer_name not in torch_activations:
                print(f"⚠ {layer_name}: PyTorch 中缺失")
                continue

            jax_act = jax_activations[layer_name]['value']
            torch_act = torch_activations[layer_name]['value']

            result = self.comparator.compare(jax_act, torch_act, layer_name)
            results.append(result)

            status = "✅" if result.passed else "❌"
            # 显示更详细的信息
            print(f"{status} {layer_name}: "
                  f"max_abs={result.max_abs_diff:.6f}, "
                  f"max_rel={result.max_rel_diff:.2%}, "
                  f"mean_abs={result.mean_abs_diff:.6f}")

        # 7. 生成报告
        report = self._generate_report(results, {
            'jax_load_time': jax_load_time,
            'torch_load_time': torch_load_time,
            'jax_runtime': jax_runtime,
            'torch_runtime': torch_runtime
        })

        # 8. 保存结果
        self._save_results(report, jax_activations, torch_activations)

        return report

    def _run_pytorch_forward(
        self,
        model,
        input_ids: torch.Tensor,
        args
    ) -> Dict:
        """运行 PyTorch 前向传播并使用 hooks 捕获激活值"""
        activations = {}
        hook_handles = []

        def create_hook(name):
            """创建 hook 函数"""
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    value = output[0]
                else:
                    value = output
                activations[name] = {
                    'value': value.detach().cpu().float().numpy(),
                    'shape': tuple(value.shape)
                }
            return hook_fn

        # 注册 hooks
        with torch.no_grad():
            # 嵌入层 hooks
            hook_handles.append(model.model.embed_tokens.register_forward_hook(
                create_hook('text_embeddings')
            ))

            # 语音嵌入层 hooks
            for i, emb in enumerate(model.speech_embeddings):
                hook_handles.append(emb.register_forward_hook(
                    create_hook(f'speech_embedding_ch{i}')
                ))

            # Input local transformer hooks
            if model.input_local_transformer and self.config.compare_input_local_transformer:
                for i, layer in enumerate(model.input_local_transformer.layers):
                    hook_handles.append(layer.register_forward_hook(
                        create_hook(f'input_local_layer_{i}_output')
                    ))

            # Speech group downcast hook
            hook_handles.append(model.speech_group_downcast.register_forward_hook(
                create_hook('speech_group_downcast_output')
            ))

            # Input local transformer norm hook (用于捕获 input_local_transformer_output)
            if model.input_local_transformer and self.config.compare_input_local_transformer:
                hook_handles.append(model.input_local_transformer.norm.register_forward_hook(
                    create_hook('input_local_transformer_output')
                ))

            # 主 transformer hooks
            if self.config.compare_main_transformer:
                for i, layer in enumerate(model.model.layers):
                    hook_handles.append(layer.register_forward_hook(
                        create_hook(f'main_layer_{i}_output')
                    ))

            # Final norm hook
            hook_handles.append(model.model.norm.register_forward_hook(
                create_hook('final_norm_output')
            ))

            # LM head hook
            hook_handles.append(model.lm_head.register_forward_hook(
                create_hook('text_logits')
            ))

            # Hidden states downcast hook
            hook_handles.append(model.hidden_states_downcast.register_forward_hook(
                create_hook('hidden_states_downcast_output')
            ))

            # Local transformer hooks
            if self.config.compare_local_transformer:
                for i, layer in enumerate(model.local_transformer.layers):
                    hook_handles.append(layer.register_forward_hook(
                        create_hook(f'local_layer_{i}_output')
                    ))

                # Local LM heads
                for i, lm_head in enumerate(model.local_transformer_lm_heads):
                    hook_handles.append(lm_head.register_forward_hook(
                        create_hook(f'local_lm_head_ch{i}')
                    ))

            # 准备输入
            B, _, T = input_ids.shape
            T_groups = T // model.group_size

            # 创建 attention_mask 和 position_ids
            attention_mask = torch.ones((B, T_groups), dtype=torch.bool, device=input_ids.device)
            position_ids = torch.arange(T_groups, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(B, -1)

            # 执行主模型前向传播
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                cache_position=None
            )

            # 如果需要比较 local transformer，手动运行它
            if self.config.compare_local_transformer and output.local_hidden_states is not None:
                local_hidden = output.local_hidden_states  # [B, 1, local_dim]

                # 手动运行 local transformer 一步
                local_output = model.local_transformer(
                    inputs_embeds=local_hidden,
                    return_dict=True,
                    use_cache=False
                )
                local_x = local_output.last_hidden_state

                # 运行所有 LM heads 以触发 hooks
                for i, lm_head in enumerate(model.local_transformer_lm_heads):
                    _ = lm_head(local_x)

            # 移除所有 hooks
            for handle in hook_handles:
                handle.remove()

        return activations

    def _generate_report(self, results: List[ComparisonResult], timings: Dict) -> Dict:
        """生成报告"""
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed

        print("\n" + "=" * 70)
        print("验证结果汇总")
        print("=" * 70)
        print(f"\n总计层数: {total}")
        print(f"通过: {passed} ({100*passed/total:.1f}%)")
        print(f"失败: {failed} ({100*failed/total:.1f}%)")

        if failed > 0:
            print("\n失败的层（按绝对差异排序）:")
            failed_results = [r for r in results if not r.passed]
            failed_results.sort(key=lambda r: r.max_abs_diff, reverse=True)

            for i, r in enumerate(failed_results[:10], 1):  # 只显示前 10 个最差的
                print(f"  {i}. {r.layer_name}")
                print(f"     最大绝对差异: {r.max_abs_diff:.6f}")
                print(f"     最大相对差异: {r.max_rel_diff:.2%}")
                print(f"     平均绝对差异: {r.mean_abs_diff:.6f}")

            if len(failed_results) > 10:
                print(f"  ... 还有 {len(failed_results) - 10} 个失败的层")

            # 计算统计信息
            avg_abs_diff = sum(r.max_abs_diff for r in failed_results) / len(failed_results)
            avg_rel_diff = sum(r.max_rel_diff for r in failed_results) / len(failed_results)

            print(f"\n失败层的平均差异:")
            print(f"  平均最大绝对差异: {avg_abs_diff:.6f}")
            print(f"  平均最大相对差异: {avg_rel_diff:.2%}")
        else:
            print("\n✅ 所有层验证通过！PyTorch 和 JAX 实现一致。")

        return {
            'config': asdict(self.config),
            'summary': {
                'total_layers': total,
                'passed': passed,
                'failed': failed,
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
                **timings
            },
            'results': [asdict(r) for r in results]
        }

    def _save_results(self, report: Dict, jax_activations: Dict, torch_activations: Dict):
        """保存结果"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')

        # 保存 JSON 报告
        json_path = os.path.join(self.config.output_dir, f'report_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n详细报告: {json_path}")

        # 保存激活值
        if self.config.save_activations:
            act_path = os.path.join(self.config.output_dir, f'activations_{timestamp}.npz')
            np.savez_compressed(
                act_path,
                **{
                    f'jax_{k}': v['value'] for k, v in jax_activations.items()
                },
                **{
                    f'torch_{k}': v['value'] for k, v in torch_activations.items()
                }
            )
            print(f"激活值: {act_path}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='MiMo Audio 层级一致性验证')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型路径 (默认使用 ModelScope 缓存)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='批次大小')
    parser.add_argument('--num_groups', type=int, default=4,
                        help='组数')
    parser.add_argument('--rtol', type=float, default=1e-2,
                        help='相对容差')
    parser.add_argument('--atol', type=float, default=1e-3,
                        help='绝对容差')
    parser.add_argument('--output_dir', type=str, default='comparison_results',
                        help='输出目录')
    parser.add_argument('--no_save_activations', action='store_true',
                        help='不保存激活值')

    args = parser.parse_args()

    # 创建配置
    config = ComparisonConfig(
        model_path=args.model_path or ComparisonConfig.model_path,
        batch_size=args.batch_size,
        num_groups=args.num_groups,
        rtol=args.rtol,
        atol=args.atol,
        output_dir=args.output_dir,
        save_activations=not args.no_save_activations
    )

    # 运行对比
    runner = LayerComparisonRunner(config)
    runner.run()


if __name__ == '__main__':
    main()
