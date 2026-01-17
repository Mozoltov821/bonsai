"""
MiMo Audio 模型端到端测试脚本。

该脚本测试完整的流程：
1. 音频 tokenizer 加载和推理
2. 主模型加载和推理
3. 音频处理：波形 -> mel频谱 -> tokens -> 重建
4. 文本转语音生成
5. 语音转文本处理

使用方法：
    python -m bonsai.models.mimo_audio.test.test_end_to_end
"""

import os
import sys
import json
import time
import gc
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

# 将测试目录添加到路径
sys.path.insert(0, os.path.dirname(__file__))


class EndToEndTester:
    """MiMo Audio 模型端到端测试器"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        初始化端到端测试器。

        参数：
            model_path: 主模型路径。如果为 None，使用默认的 ModelScope 缓存。
            tokenizer_path: 音频 tokenizer 路径。如果为 None，使用默认的 ModelScope 缓存。
            verbose: 是否打印详细信息。
        """
        self.verbose = verbose
        self.model_path = model_path or os.path.expanduser(
            "~/.cache/modelscope/hub/models/XiaomiMiMo/MiMo-Audio-7B-Instruct"
        )
        self.tokenizer_path = tokenizer_path or os.path.expanduser(
            "~/.cache/modelscope/hub/models/XiaomiMiMo/MiMo-Audio-Tokenizer"
        )

        self.tokenizer_config = None
        self.tokenizer_model = None
        self.main_model = None
        self.mel_extractor = None

        self.test_results = {}

    def _print(self, message: str, level: str = "INFO"):
        """如果启用详细模式，打印消息。"""
        if self.verbose:
            prefix = {
                "INFO": "ℹ️ ",
                "SUCCESS": "✅",
                "ERROR": "❌",
                "WARNING": "⚠️ ",
                "SECTION": "\n" + "=" * 70 + "\n",
            }.get(level, "")
            print(f"{prefix} {message}")

    def _clear_memory(self):
        """清理 Python 垃圾以释放内存（保留 JAX 编译缓存以提升性能）。"""
        # ✅ 不清理 JAX 编译缓存，以避免重新编译开销
        # jax.clear_caches()  # 保留编译缓存以提升推理速度
        # 强制垃圾回收
        gc.collect()
        self._print("已清理内存 (GC only, 保留编译缓存)", "INFO")

    def test_tokenizer_loading(self) -> bool:
        """测试 1: 加载音频 tokenizer 并验证结构"""
        self._print("测试 1: 音频 Tokenizer 加载", "SECTION")

        try:
            if not os.path.exists(self.tokenizer_path):
                self._print(f"Tokenizer 路径未找到: {self.tokenizer_path}", "ERROR")
                return False

            # 加载配置
            config_path = os.path.join(self.tokenizer_path, "config.json")
            with open(config_path) as f:
                config_dict = json.load(f)

            from bonsai.models.mimo_audio.mimo_audio_tokenizer import MiMoAudioTokenizerConfig
            from bonsai.models.mimo_audio.mimo_audio_tokenizer_params import load_tokenizer_weights_from_safetensors

            # ✅ 禁用 sharding 以提升单卡推理速度（避免多卡通信开销）
            config_dict['use_sharding'] = False
            self.tokenizer_config = MiMoAudioTokenizerConfig(**config_dict)

            self._print("正在加载 tokenizer 配置...")
            self._print(f"  - 编码器层数: {self.tokenizer_config.encoder_layers}")
            self._print(f"  - 解码器层数: {self.tokenizer_config.decoder_layers}")
            self._print(f"  - 量化器数量: {self.tokenizer_config.num_quantizers}")
            self._print(f"  - 采样率: {self.tokenizer_config.sampling_rate} Hz")
            self._print("✅ 已禁用 sharding（单卡推理模式）")

            # 加载模型权重（不使用 mesh）
            safetensors_path = os.path.join(self.tokenizer_path, "model.safetensors")
            start_time = time.time()

            self.tokenizer_model = load_tokenizer_weights_from_safetensors(
                config=self.tokenizer_config,
                safetensors_path=safetensors_path,
                dtype=jnp.float32,  # ✅ Tokenizer必须用float32：quantizer和ISTFT需要
                mesh=None,  # 不使用 sharding
                rngs=nnx.Rngs(0),
            )

            load_time = time.time() - start_time
            self._print(f"Tokenizer 加载完成，耗时 {load_time:.2f}秒", "SUCCESS")

            self.test_results["tokenizer_loading"] = {
                "success": True,
                "load_time": load_time,
            }

            return True

        except Exception as e:
            self._print(f"Tokenizer 加载失败: {type(e).__name__}: {e}", "ERROR")
            self.test_results["tokenizer_loading"] = {
                "success": False,
                "error": str(e),
            }
            return False

    def test_tokenizer_forward(self) -> bool:
        """测试 2: 使用虚拟输入测试 tokenizer 前向传播"""
        self._print("测试 2: Tokenizer 前向传播", "SECTION")

        if self.tokenizer_model is None:
            self._print("Tokenizer 未加载，跳过测试", "WARNING")
            return False

        try:
            # Create dummy mel spectrogram input
            batch_size = 2
            mel_len = 200
            n_mels = self.tokenizer_config.n_mels

            self._print(f"创建虚拟输入: batch={batch_size}, mel_len={mel_len}, n_mels={n_mels}")

            dummy_mels = jax.random.normal(
                jax.random.key(42),
                (batch_size, mel_len, n_mels),
                dtype=jnp.bfloat16,
            )
            dummy_lengths = jnp.array([mel_len, mel_len])

            # Test encoding
            self._print("Testing encoder...")
            start_time = time.time()
            encoder_output = self.tokenizer_model.encode(
                dummy_mels, dummy_lengths, use_quantizer=True
            )
            encode_time = time.time() - start_time

            self._print(f"  - 编码器输出形状: {encoder_output.hidden_states.shape}")
            self._print(f"  - 代码 shape: {encoder_output.codes.shape}")
            self._print(f"  - Output lengths: {encoder_output.output_lengths}")
            self._print(f"  - 编码耗时: {encode_time:.3f}s")

            # Test decoding
            self._print("Testing decoder...")
            start_time = time.time()
            decoded_output = self.tokenizer_model.decode(encoder_output.codes)
            decode_time = time.time() - start_time

            self._print(f"  - 解码器输出形状: {decoded_output.shape}")
            self._print(f"  - 解码耗时: {decode_time:.3f}s")

            self.test_results["tokenizer_forward"] = {
                "success": True,
                "encode_time": encode_time,
                "decode_time": decode_time,
                "encoder_output_shape": encoder_output.hidden_states.shape,
                "codes_shape": encoder_output.codes.shape,
            }

            self._print("Tokenizer forward pass 测试通过", "SUCCESS")

            return True

        except Exception as e:
            self._print(f"Tokenizer forward pass 失败: {type(e).__name__}: {e}", "ERROR")
            self.test_results["tokenizer_forward"] = {
                "success": False,
                "error": str(e),
            }
            import traceback
            traceback.print_exc()
            return False

    def test_mel_extraction(self) -> bool:
        """测试 3: 测试 mel 频谱提取"""
        self._print("测试 3: Mel 频谱提取", "SECTION")

        try:
            from bonsai.models.mimo_audio.mimo_audio_tokenizer import MelSpectrogram

            # Create mel extractor
            self.mel_extractor = MelSpectrogram(
                sample_rate=self.tokenizer_config.sampling_rate,
                n_fft=self.tokenizer_config.nfft,
                hop_length=self.tokenizer_config.hop_length,
                win_length=self.tokenizer_config.window_size,
                f_min=self.tokenizer_config.fmin,
                f_max=self.tokenizer_config.fmax or self.tokenizer_config.sampling_rate // 2,
                n_mels=self.tokenizer_config.n_mels,
            )

            self._print("已创建 mel 提取器")

            # Generate dummy audio (1 second of sine wave)
            sample_rate = self.tokenizer_config.sampling_rate
            duration = 1.0
            frequency = 440.0  # A4 note

            t = jnp.linspace(0, duration, int(sample_rate * duration))
            waveform = jnp.sin(2 * jnp.pi * frequency * t).astype(jnp.float32)

            self._print(f"已生成测试音频: {waveform.shape}")

            # Extract mel spectrogram
            start_time = time.time()
            mel_spec = self.mel_extractor(waveform)
            mel_time = time.time() - start_time

            self._print(f"  - Mel 频谱 shape: {mel_spec.shape}")
            self._print(f"  - Mel 提取耗时: {mel_time:.3f}s")

            # Test with batch
            batch_waveform = jnp.stack([waveform, waveform], axis=0)
            batch_mel = self.mel_extractor(batch_waveform)

            self._print(f"  - 批次 mel 形状: {batch_mel.shape}")

            self.test_results["mel_extraction"] = {
                "success": True,
                "mel_time": mel_time,
                "mel_shape": mel_spec.shape,
                "batch_mel_shape": batch_mel.shape,
            }

            self._print("Mel extraction 测试通过", "SUCCESS")
            return True

        except Exception as e:
            self._print(f"Mel extraction 失败: {type(e).__name__}: {e}", "ERROR")
            self.test_results["mel_extraction"] = {
                "success": False,
                "error": str(e),
            }
            return False

    def test_audio_reconstruction(self) -> bool:
        """测试 4: 测试完整音频重建流程"""
        self._print("测试 4: 音频重建流程", "SECTION")

        if self.tokenizer_model is None or self.mel_extractor is None:
            self._print("前置条件未满足，跳过测试", "WARNING")
            return False

        try:
            # Generate test audio
            sample_rate = self.tokenizer_config.sampling_rate
            duration = 2.0
            frequency = 440.0

            t = jnp.linspace(0, duration, int(sample_rate * duration))
            original_waveform = jnp.sin(2 * jnp.pi * frequency * t).astype(jnp.float32)

            self._print(f"原始波形: {original_waveform.shape}")
            self._print(f"  - 采样率: {sample_rate} Hz")
            self._print(f"  - 时长: {duration} s")
            self._print(f"  - 频率: {frequency} Hz")

            # Extract mel spectrogram
            mel_spec = self.mel_extractor(original_waveform)
            self._print(f"Mel 频谱: {mel_spec.shape}")

            # Add batch dimension and transpose to (batch, time, mels)
            mel_input = mel_spec.T[jnp.newaxis, :, :].astype(jnp.bfloat16)
            mel_lengths = jnp.array([mel_input.shape[1]])

            self._print(f"Mel 输入形状: {mel_input.shape}")

            # Encode to tokens
            start_time = time.time()
            encoder_output = self.tokenizer_model.encode(
                mel_input, mel_lengths, use_quantizer=True
            )
            encode_time = time.time() - start_time

            self._print(f"  - 编码 to tokens: {encoder_output.codes.shape}")
            self._print(f"  - 编码耗时: {encode_time:.3f}s")

            # Print some token statistics
            codes_np = np.array(encoder_output.codes)
            self._print(f"  - Token 统计:")
            self._print(f"    最小 token: {codes_np.min()}")
            self._print(f"    最大 token: {codes_np.max()}")
            self._print(f"    唯一 tokens 数: {len(np.unique(codes_np))}")

            # Decode back to audio
            start_time = time.time()
            reconstructed_audio = self.tokenizer_model.decode(encoder_output.codes)
            decode_time = time.time() - start_time

            self._print(f"  - 重建音频: {reconstructed_audio.shape}")
            self._print(f"  - 解码耗时: {decode_time:.3f}s")

            # Calculate reconstruction metrics
            original_len = original_waveform.shape[0]
            recon_len = reconstructed_audio.shape[-1]

            self._print(f"  - 原始长度: {original_len} 采样点 ({original_len/sample_rate:.2f}s)")
            self._print(f"  - 重建长度: {recon_len} 采样点 ({recon_len/sample_rate:.2f}s)")
            self._print(f"  - 长度比率: {recon_len / original_len:.2f}x")


            import soundfile as sf
            output_dir = "test_outputs"
            os.makedirs(output_dir, exist_ok=True)

            # Save original audio
            original_path = os.path.join(output_dir, "original_audio.wav")
            sf.write(original_path, np.array(original_waveform), sample_rate)
            self._print(f"  - 已保存原始音频: {original_path}")

            # Save reconstructed audio
            reconstructed_path = os.path.join(output_dir, "reconstructed_audio.wav")
            reconstructed_np = np.array(reconstructed_audio[0, 0, :])
            sf.write(reconstructed_path, reconstructed_np, sample_rate)

            self.test_results["audio_reconstruction"] = {
                "success": True,
                "encode_time": encode_time,
                "decode_time": decode_time,
                "original_length": int(original_len),
                "reconstructed_length": int(recon_len),
            }

            self._print("Audio reconstruction 测试通过", "SUCCESS")

            # Clear memory after reconstruction
            self._clear_memory()
            return True

        except Exception as e:
            self._print(f"Audio reconstruction 失败: {type(e).__name__}: {e}", "ERROR")
            self.test_results["audio_reconstruction"] = {
                "success": False,
                "error": str(e),
            }
            import traceback
            traceback.print_exc()
            return False

    def test_main_model_loading(self) -> bool:
        """测试 5: 加载 MiMo Audio 主模型"""
        self._print("测试 5: 主模型加载", "SECTION")

        try:
            if not os.path.exists(self.model_path):
                self._print(f"Main model path not found: {self.model_path}", "WARNING")
                self._print("Skipping main model tests", "WARNING")
                return False

            from bonsai.models.mimo_audio.modeling import MiMoAudioConfig, MiMoAudioArguments
            from bonsai.models.mimo_audio.params import create_model_with_weights

            # Load config
            config_path = os.path.join(self.model_path, "config.json")
            with open(config_path) as f:
                config_dict = json.load(f)

            self._print("Loading main model config...")
            self._print(f"  - 模型类型: {config_dict.get('model_type')}")
            self._print(f"  - 隐藏层大小: {config_dict.get('hidden_size')}")
            self._print(f"  - 词表大小: {config_dict.get('vocab_size')}")
            self._print(f"  - 层数: {config_dict.get('num_hidden_layers')}")

            # Create config without sharding (单卡推理模式)
            config_kwargs = {k: v for k, v in config_dict.items() if k in MiMoAudioConfig.__dataclass_fields__}
            config = MiMoAudioConfig(**config_kwargs)
            self._print("✅ 已禁用模型 sharding（单卡推理模式）")

            # ✅ 关键修复：从tokenizer动态获取special token IDs
            # config.json中的IDs可能是错误的或过时的
            # 必须使用tokenizer的实际IDs
            from transformers import AutoTokenizer
            temp_tokenizer = AutoTokenizer.from_pretrained(self.model_path)

            # Create arguments with correct token IDs from tokenizer
            args = MiMoAudioArguments(
                model_name_or_path=self.model_path,
                sosp_idx=temp_tokenizer.convert_tokens_to_ids("<|sosp|>"),
                eosp_idx=temp_tokenizer.convert_tokens_to_ids("<|eosp|>"),
                sostm_idx=temp_tokenizer.convert_tokens_to_ids("<|sostm|>"),
                eostm_idx=temp_tokenizer.convert_tokens_to_ids("<|eostm|>"),
                eot_idx=temp_tokenizer.convert_tokens_to_ids("<|eot|>"),
                empty_idx=temp_tokenizer.convert_tokens_to_ids("<|empty|>"),
            )

            self._print(f"✅ 从tokenizer获取的正确token IDs:")
            self._print(f"  - SOSTM: {args.sostm_idx}")
            self._print(f"  - EOSTM: {args.eostm_idx}")
            self._print(f"  - Empty: {args.empty_idx}")

            # Load model without sharding
            start_time = time.time()
            self.main_model = create_model_with_weights(
                model_path=self.model_path,
                config=config,
                args=args,
                rngs=nnx.Rngs(0),
                mesh=None,  # 不使用 sharding
            )
            load_time = time.time() - start_time

            self._print(f"Main model loaded in {load_time:.2f}s", "SUCCESS")

            # Verify model structure
            self._print("模型结构:")
            self._print(f"  - 主 Transformer: {self.main_model.model is not None}")
            self._print(f"  - 局部 Transformer: {self.main_model.local_transformer is not None}")
            self._print(f"  - 输入局部 Transformer: {self.main_model.input_local_transformer is not None}")
            self._print(f"  - 音频通道: {len(self.main_model.speech_embeddings)}")

            self.test_results["main_model_loading"] = {
                "success": True,
                "load_time": load_time,
            }

            # Clear memory after loading main model
            self._clear_memory()
            return True

        except Exception as e:
            self._print(f"Main model loading 失败: {type(e).__name__}: {e}", "ERROR")
            self.test_results["main_model_loading"] = {
                "success": False,
                "error": str(e),
            }
            import traceback
            traceback.print_exc()
            return False

    def test_main_model_forward(self) -> bool:
        """测试 6: 测试主模型前向传播"""
        self._print("测试 6: 主模型前向传播", "SECTION")

        if self.main_model is None:
            self._print("主模型未加载，跳过测试", "WARNING")
            return False

        try:
            # Create dummy input
            batch_size = 1
            audio_channels = self.main_model.audio_channels
            group_size = self.main_model.group_size
            num_groups = 4

            # 形状: [B, audio_channels + 1, T * group_size]
            input_shape = (batch_size, audio_channels + 1, num_groups * group_size)

            self._print(f"创建虚拟输入: {input_shape}")
            self._print(f"  - 批次大小: {batch_size}")
            self._print(f"  - 音频通道: {audio_channels}")
            self._print(f"  - 分组大小: {group_size}")
            self._print(f"  - 分组数量: {num_groups}")

            # Create random input tokens
            vocab_size = self.main_model.config.vocab_size
            input_ids = jax.random.randint(
                jax.random.key(123),
                input_shape,
                minval=0,
                maxval=min(vocab_size, 1000),
            )

            # Print input sample
            self._print(f"\n输入 tokens (first group, first 5 tokens per channel):")
            for ch in range(min(3, audio_channels + 1)):  # Show first 3 channels
                channel_tokens = input_ids[0, ch, :5]
                self._print(f"  通道 {ch}: {channel_tokens}")

            # Initialize cache
            cache = self.main_model.model.init_cache(
                self.main_model.qwen2_config,
                batch_size,
                num_groups,
                generate_steps=0,
                dtype=jnp.bfloat16,
            )

            self._print("\n正在执行前向传播...")
            start_time = time.time()

            text_logits, local_hidden_states = self.main_model.forward(
                input_ids, cache, pad_id=0
            )

            forward_time = time.time() - start_time

            self._print(f"  - 文本 logits shape: {text_logits.shape}")
            self._print(f"  - 局部隐藏状态 shape: {local_hidden_states.shape}")
            self._print(f"  - 前向传播耗时: {forward_time:.3f}s")

            # Print output statistics
            self._print(f"\n输出统计:")
            self._print(f"  文本 logits:")
            self._print(f"    均值: {float(jnp.mean(text_logits)):.4f}")
            self._print(f"    标准差: {float(jnp.std(text_logits)):.4f}")
            self._print(f"    最小值: {float(jnp.min(text_logits)):.4f}")
            self._print(f"    最大值: {float(jnp.max(text_logits)):.4f}")

            # Get top predictions
            probs = jax.nn.softmax(text_logits[0, 0], axis=-1)
            top_k = 5
            top_probs, top_indices = jax.lax.top_k(probs, top_k)

            self._print(f"\n  前 {top_k} 预测的 tokens:")
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                self._print(f"    {i+1}. Token {int(idx)}: {float(prob):.6f}")

            self._print(f"\n  局部隐藏状态:")
            self._print(f"    均值: {float(jnp.mean(local_hidden_states)):.4f}")
            self._print(f"    标准差: {float(jnp.std(local_hidden_states)):.4f}")
            self._print(f"    最小值: {float(jnp.min(local_hidden_states)):.4f}")
            self._print(f"    最大值: {float(jnp.max(local_hidden_states)):.4f}")

            # Save outputs to file
            try:
                output_dir = "test_outputs"
                os.makedirs(output_dir, exist_ok=True)

                output_path = os.path.join(output_dir, "model_outputs.npz")
                np.savez(
                    output_path,
                    text_logits=np.array(text_logits),
                    local_hidden_states=np.array(local_hidden_states),
                    input_ids=np.array(input_ids),
                )
                self._print(f"\n  - 已保存输出到: {output_path}")

            except Exception as e:
                self._print(f"  - 保存失败 outputs: {e}", "WARNING")

            self.test_results["main_model_forward"] = {
                "success": True,
                "forward_time": forward_time,
                "text_logits_shape": text_logits.shape,
                "local_hidden_states_shape": local_hidden_states.shape,
            }

            self._print("\nMain model forward pass 测试通过", "SUCCESS")

            # Clear memory after main model forward pass
            self._clear_memory()
            return True

        except Exception as e:
            self._print(f"Main model forward pass 失败: {type(e).__name__}: {e}", "ERROR")
            self.test_results["main_model_forward"] = {
                "success": False,
                "error": str(e),
            }
            import traceback
            traceback.print_exc()
            return False

    def test_input_output_shapes(self) -> bool:
        """测试 7: 详细的输入/输出形状验证"""
        self._print("测试 7: 输入/输出形状验证", "SECTION")

        if self.main_model is None:
            self._print("主模型未加载，跳过测试", "WARNING")
            return False

        try:
            audio_channels = self.main_model.audio_channels
            group_size = self.main_model.group_size
            vocab_size = self.main_model.config.vocab_size

            self._print(f"模型配置:")
            self._print(f"  - 音频通道: {audio_channels}")
            self._print(f"  - 分组大小: {group_size}")
            self._print(f"  - 词表大小: {vocab_size}")
            self._print(f"  - 隐藏层大小: {self.main_model.config.hidden_size}")
            self._print(f"  - 局部维度: {self.main_model.config.local_dim}")

            # Test 1: Single batch, single group
            self._print("\n测试 7.1: 单批次单组")
            batch_size = 1
            num_groups = 1
            input_ids = jax.random.randint(
                jax.random.key(1),
                (batch_size, audio_channels + 1, num_groups * group_size),
                minval=0,
                maxval=1000,
            )
            self._print(f"  输入形状: {input_ids.shape}")
            self._print(f"  预期: (1, {audio_channels + 1}, {num_groups * group_size})")

            cache = self.main_model.model.init_cache(
                self.main_model.qwen2_config, batch_size, num_groups, 0, jnp.bfloat16
            )
            text_logits, local_hidden = self.main_model.forward(input_ids, cache, pad_id=0)

            self._print(f"  文本 logits shape: {text_logits.shape}")
            self._print(f"  预期: (1, 1, {vocab_size})")
            self._print(f"  Local hidden shape: {local_hidden.shape}")
            self._print(f"  预期: (1, 1, {self.main_model.config.local_dim})")

            assert text_logits.shape == (1, 1, vocab_size), f"文本 logits shape mismatch"
            assert local_hidden.shape == (1, 1, self.main_model.config.local_dim), f"Local hidden shape mismatch"

            # Test 2: Multiple groups
            self._print("\n测试 7.2: 单批次多组")
            num_groups = 5
            input_ids = jax.random.randint(
                jax.random.key(2),
                (batch_size, audio_channels + 1, num_groups * group_size),
                minval=0,
                maxval=1000,
            )
            self._print(f"  输入形状: {input_ids.shape}")

            cache = self.main_model.model.init_cache(
                self.main_model.qwen2_config, batch_size, num_groups, 0, jnp.bfloat16
            )
            text_logits, local_hidden = self.main_model.forward(input_ids, cache, pad_id=0)

            self._print(f"  文本 logits shape: {text_logits.shape}")
            self._print(f"  Local hidden shape: {local_hidden.shape}")

            assert text_logits.shape == (1, 1, vocab_size), f"文本 logits shape mismatch"
            assert local_hidden.shape == (1, 1, self.main_model.config.local_dim), f"Local hidden shape mismatch"

            # Test 3: Batch processing
            self._print("\n测试 7.3: 批处理")
            batch_size = 3
            num_groups = 2
            input_ids = jax.random.randint(
                jax.random.key(3),
                (batch_size, audio_channels + 1, num_groups * group_size),
                minval=0,
                maxval=1000,
            )
            self._print(f"  输入形状: {input_ids.shape}")

            cache = self.main_model.model.init_cache(
                self.main_model.qwen2_config, batch_size, num_groups, 0, jnp.bfloat16
            )
            text_logits, local_hidden = self.main_model.forward(input_ids, cache, pad_id=0)

            self._print(f"  文本 logits shape: {text_logits.shape}")
            self._print(f"  预期: ({batch_size}, 1, {vocab_size})")
            self._print(f"  Local hidden shape: {local_hidden.shape}")
            self._print(f"  预期: ({batch_size}, 1, {self.main_model.config.local_dim})")

            assert text_logits.shape == (batch_size, 1, vocab_size), f"文本 logits shape mismatch"
            assert local_hidden.shape == (batch_size, 1, self.main_model.config.local_dim), f"Local hidden shape mismatch"

            self.test_results["input_output_shapes"] = {
                "success": True,
                "tests_passed": 3,
            }

            self._print("\n输入/output shape validation 通过", "SUCCESS")

            # Clear memory after shape validation
            self._clear_memory()
            return True

        except Exception as e:
            self._print(f"输入/output shape validation 失败: {type(e).__name__}: {e}", "ERROR")
            self.test_results["input_output_shapes"] = {
                "success": False,
                "error": str(e),
            }
            import traceback
            traceback.print_exc()
            return False

    def test_tokenizer_various_lengths(self) -> bool:
        """测试 8: 测试 tokenizer 处理各种输入长度"""
        self._print("测试 8: Tokenizer 多种输入长度", "SECTION")

        if self.tokenizer_model is None:
            self._print("Tokenizer 未加载，跳过测试", "WARNING")
            return False

        try:
            n_mels = self.tokenizer_config.n_mels
            test_lengths = [50, 100, 200, 500]

            results = {}
            for mel_len in test_lengths:
                self._print(f"\n测试 mel_len={mel_len}")

                # Create input
                batch_size = 2
                mels = jax.random.normal(
                    jax.random.key(mel_len),
                    (batch_size, mel_len, n_mels),
                    dtype=jnp.bfloat16,
                )
                lengths = jnp.array([mel_len, mel_len])

                # Encode
                start_time = time.time()
                enc_output = self.tokenizer_model.encode(mels, lengths, use_quantizer=True)
                encode_time = time.time() - start_time

                # Decode
                start_time = time.time()
                dec_output = self.tokenizer_model.decode(enc_output.codes)
                decode_time = time.time() - start_time

                self._print(f"  - 输入: {mels.shape}")
                self._print(f"  - 编码: {enc_output.hidden_states.shape}")
                self._print(f"  - 代码: {enc_output.codes.shape}")
                self._print(f"  - 解码: {dec_output.shape}")
                self._print(f"  - 编码耗时: {encode_time:.3f}s")
                self._print(f"  - 解码耗时: {decode_time:.3f}s")

                results[mel_len] = {
                    "input_shape": mels.shape,
                    "encoded_shape": enc_output.hidden_states.shape,
                    "codes_shape": enc_output.codes.shape,
                    "decoded_shape": dec_output.shape,
                    "encode_time": encode_time,
                    "decode_time": decode_time,
                }

            self.test_results["tokenizer_various_lengths"] = {
                "success": True,
                "results": results,
            }

            self._print("\nTokenizer various lengths 测试通过", "SUCCESS")

            # Clear memory after various lengths test
            self._clear_memory()
            return True

        except Exception as e:
            self._print(f"Tokenizer various lengths 测试失败: {type(e).__name__}: {e}", "ERROR")
            self.test_results["tokenizer_various_lengths"] = {
                "success": False,
                "error": str(e),
            }
            import traceback
            traceback.print_exc()
            return False

    def test_output_distributions(self) -> bool:
        """测试 9: 分析输出分布"""
        self._print("测试 9: 输出分布分析", "SECTION")

        if self.main_model is None:
            self._print("主模型未加载，跳过测试", "WARNING")
            return False

        try:
            batch_size = 2
            num_groups = 3
            audio_channels = self.main_model.audio_channels
            group_size = self.main_model.group_size

            # Create input
            input_ids = jax.random.randint(
                jax.random.key(999),
                (batch_size, audio_channels + 1, num_groups * group_size),
                minval=0,
                maxval=1000,
            )

            cache = self.main_model.model.init_cache(
                self.main_model.qwen2_config, batch_size, num_groups, 0, jnp.bfloat16
            )
            text_logits, local_hidden = self.main_model.forward(input_ids, cache, pad_id=0)

            self._print("文本 logits 统计:")
            self._print(f"  - 形状: {text_logits.shape}")
            self._print(f"  - 均值: {float(jnp.mean(text_logits)):.4f}")
            self._print(f"  - 标准差: {float(jnp.std(text_logits)):.4f}")
            self._print(f"  - 最小值: {float(jnp.min(text_logits)):.4f}")
            self._print(f"  - 最大值: {float(jnp.max(text_logits)):.4f}")

            # Check for NaN or Inf
            has_nan = jnp.any(jnp.isnan(text_logits))
            has_inf = jnp.any(jnp.isinf(text_logits))

            self._print(f"  - 包含 NaN: {has_nan}")
            self._print(f"  - 包含 Inf: {has_inf}")

            if has_nan or has_inf:
                self._print("警告：输出中发现 NaN 或 Inf", "WARNING")

            self._print("\n局部隐藏状态统计:")
            self._print(f"  - 形状: {local_hidden.shape}")
            self._print(f"  - 均值: {float(jnp.mean(local_hidden)):.4f}")
            self._print(f"  - 标准差: {float(jnp.std(local_hidden)):.4f}")
            self._print(f"  - 最小值: {float(jnp.min(local_hidden)):.4f}")
            self._print(f"  - 最大值: {float(jnp.max(local_hidden)):.4f}")

            # Apply softmax to logits
            probs = jax.nn.softmax(text_logits, axis=-1)
            top_k = 10
            top_probs, top_indices = jax.lax.top_k(probs[0, 0], top_k)

            self._print(f"\n前 {top_k} 概率:")
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                self._print(f"  {i+1}. Token {int(idx)}: {float(prob):.4f}")

            self.test_results["output_distributions"] = {
                "success": True,
                "has_nan": bool(has_nan),
                "has_inf": bool(has_inf),
                "logits_mean": float(jnp.mean(text_logits)),
                "logits_std": float(jnp.std(text_logits)),
            }

            self._print("\nOutput distribution analysis 通过", "SUCCESS")

            # Clear memory after distribution analysis
            self._clear_memory()
            return True

        except Exception as e:
            self._print(f"Output distribution analysis 失败: {type(e).__name__}: {e}", "ERROR")
            self.test_results["output_distributions"] = {
                "success": False,
                "error": str(e),
            }
            import traceback
            traceback.print_exc()
            return False

    def print_summary(self):
        """打印测试总结"""
        self._print("测试总结", "SECTION")

        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result.get("success", False))

        for test_name, result in self.test_results.items():
            status = "PASS" if result.get("success", False) else "FAIL"
            status_icon = "✅" if status == "PASS" else "❌"
            self._print(f"{status_icon} {test_name}: {status}")

            if result.get("success") and self.verbose:
                if "load_time" in result:
                    self._print(f"    加载耗时: {result['load_time']:.2f}秒")
                if "encode_time" in result:
                    self._print(f"    编码耗时: {result['encode_time']:.3f}秒")
                if "decode_time" in result:
                    self._print(f"    解码耗时: {result['decode_time']:.3f}秒")
                if "forward_time" in result:
                    self._print(f"    前向传播耗时: {result['forward_time']:.3f}秒")

        self._print(f"\n{passed_tests}/{total_tests} 测试通过")

        if passed_tests == total_tests:
            self._print("所有测试通过！", "SUCCESS")
            return 0
        else:
            self._print(f"{total_tests - passed_tests} 个测试失败", "ERROR")
            return 1

    @staticmethod
    def insert_between(tokens: list, group_size: int, fill_value: int) -> list:
        """
        在tokens之间插入填充值以满足group_size要求。

        例如：group_size=4时，[A, B, C] -> [A, fill, fill, fill, B, fill, fill, fill, C, fill, fill, fill]

        Args:
            tokens: 原始token列表
            group_size: 分组大小
            fill_value: 填充值（通常使用pad_token_id）

        Returns:
            插入填充后的token列表
        """
        if group_size <= 1:
            return tokens

        result = []
        for token in tokens:
            result.append(token)
            # 在每个token后添加(group_size - 1)个填充值
            result.extend([fill_value] * (group_size - 1))

        return result

    def test_full_inference(self) -> bool:
        """测试 10: 完整推理流程（文本+音频生成）"""
        self._print("测试 10: 完整推理流程", "SECTION")

        # 如果模型未加载，先加载
        if self.tokenizer_model is None:
            self._print("Tokenizer 未加载，正在加载...", "INFO")
            if not self.test_tokenizer_loading():
                self._print("Tokenizer 加载失败，跳过测试", "ERROR")
                return False

        if self.main_model is None:
            self._print("主模型未加载，正在加载...", "INFO")
            if not self.test_main_model_loading():
                self._print("主模型加载失败，跳过测试", "ERROR")
                return False

        # ✅ 导入 JIT 编译的函数以提升推理速度
        from bonsai.models.mimo_audio.modeling import forward_jit, local_forward_jit
        self._print("✅ 使用 JIT 编译加速推理", "INFO")

        # ✅ 预热 JIT 编译：使用小输入触发编译
        self._print("正在预热 JIT 编译（第一次调用会触发编译）...", "INFO")
        warmup_cache = self.main_model.model.init_cache(
            self.main_model.qwen2_config,
            batch_size=1,
            token_len=1,
            generate_steps=0,
            dtype=jnp.bfloat16,
        )
        warmup_input = jnp.zeros(
            (1, self.main_model.audio_channels + 1, self.main_model.group_size),
            dtype=jnp.int32
        )
        warmup_start = time.time()
        _, _, _ = forward_jit(self.main_model, warmup_input, warmup_cache, pad_id=0)
        warmup_time = time.time() - warmup_start
        self._print(f"JIT 预热完成，耗时 {warmup_time:.2f}秒（包含编译时间）", "SUCCESS")
        self._print("后续推理将直接使用编译后的代码，速度会大幅提升", "INFO")

        try:
            # 加载文本 tokenizer
            try:
                from transformers import AutoTokenizer
                text_tokenizer_path = self.model_path
                text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_path)
                self._print(f"已加载文本 tokenizer，词表大小: {len(text_tokenizer)}")
            except Exception as e:
                self._print(f"加载文本 tokenizer 失败: {e}", "WARNING")
                self._print("使用模拟 tokenizer", "WARNING")
                text_tokenizer = None

            # 准备输入
            audio_channels = self.main_model.audio_channels
            group_size = self.main_model.group_size
            batch_size = 1

            # 文本通道：使用TTS prompt格式（关键！）
            if text_tokenizer:
                # text_to_speak = "床前明月光，疑是地上霜。举头望明月，低头思故乡。"  # 使用简短文本
                text_to_speak = "在那边，在大海的那一头，老人正睡在自己的棚子里。他依然脸朝下睡着，孩子坐在他身边守着他，老人正梦见狮子。一个人并不是生来要给打败的，你尽可以把他消灭掉，可就是打不败他。"

                # ✅ 使用官方推荐的标准TTS模板（见 templates.py:30-41）
                tts_template = "请将这段文字转换为语音"  # 官方推荐，列表第一个
                # 其他官方模板选项：
                # tts_template = "帮我把这个文本读出来"
                # tts_template = "请朗读这段内容"
                # tts_template = "帮我朗读这段文字"
                # tts_template = ""  # 无模板模式

                # 官方TTS格式：
                # <|im_start|>user\n{template}: {text}<|im_end|>\n<|im_start|>assistant\n<|sostm|>
                if tts_template:
                    chat_text = f"<|im_start|>user\n{tts_template}: {text_to_speak}<|im_end|>\n<|im_start|>assistant\n<|sostm|>"
                else:
                    # 无模板模式
                    chat_text = f"<|im_start|>user\n{text_to_speak}<|im_end|>\n<|im_start|>assistant\n<|sostm|>"

                # 1. 首先tokenize文本（不限制长度）
                text_tokens_raw = text_tokenizer.encode(chat_text)

                # 2. 使用insert_between添加间隔填充
                # ✅ 关键修复：官方使用 -100 作为填充值（padding）
                text_tokens_with_spacing = self.insert_between(
                    text_tokens_raw,
                    group_size,
                    -100  # 官方使用 -100 作为 padding value
                )

                # 3. ✅ 关键修复：根据实际tokens数量确定num_groups，不截断
                # 确保SOSTM不会被截断掉
                num_groups = len(text_tokens_with_spacing) // group_size
                if len(text_tokens_with_spacing) % group_size != 0:
                    # 填充到group_size的倍数
                    text_tokens_with_spacing.extend([-100] * (group_size - len(text_tokens_with_spacing) % group_size))
                    num_groups = len(text_tokens_with_spacing) // group_size

                text_tokens = text_tokens_with_spacing

                self._print(f"\n推理配置:")
                self._print(f"  - 批次大小: {batch_size}")
                self._print(f"  - 音频通道数: {audio_channels}")
                self._print(f"  - 分组大小: {group_size}")
                self._print(f"  - Prefill组数: {num_groups}")

                self._print(f"\n输入文本（待转语音）: {text_to_speak}")
                self._print(f"TTS 模板: {tts_template}")
                self._print(f"完整 prompt: {chat_text[:100]}...")
                self._print(f"原始 tokens 数量: {len(text_tokens_raw)}")
                self._print(f"添加间隔后 tokens 数量: {len(text_tokens_with_spacing)}")
                self._print(f"最终 tokens: {text_tokens[:20]}... (前20个)")
                self._print(f"SOSTM token ID: {self.main_model.args.sostm_idx}")
                self._print(f"EOSTM token ID: {self.main_model.args.eostm_idx}")

                # 检查SOSTM是否在tokens中
                if self.main_model.args.sostm_idx in text_tokens_raw:
                    sostm_pos = text_tokens_raw.index(self.main_model.args.sostm_idx)
                    self._print(f"✅ SOSTM在原始tokens位置: {sostm_pos}/{len(text_tokens_raw)}")
                else:
                    self._print(f"❌ 警告：SOSTM不在原始tokens中！")

            # 创建输入 - 现在知道num_groups了
            input_shape = (batch_size, audio_channels + 1, num_groups * group_size)
            input_ids = jnp.zeros(input_shape, dtype=jnp.int32)

            # 设置文本通道
            input_ids = input_ids.at[0, 0, :].set(jnp.array(text_tokens))

            # 音频通道：在prefill阶段使用每个通道自己的empty_id（表示无音频输入）
            for ch in range(1, audio_channels + 1):
                # 每个音频通道有自己的empty_id (例如：1024, 1024, 128, 128, ...)
                channel_empty_id = self.main_model.speech_empty_ids[ch - 1]  # ch-1因为speech_empty_ids是0-indexed
                audio_empty_tokens = jnp.full((num_groups * group_size,), channel_empty_id, dtype=jnp.int32)
                input_ids = input_ids.at[0, ch, :].set(audio_empty_tokens)

            self._print(f"\n输入形状: {input_ids.shape}")
            self._print(f"音频通道empty IDs: {self.main_model.speech_empty_ids}")
            self._print(f"文本 pad_id: {text_tokenizer.pad_token_id if text_tokenizer else 0}")

            # 初始化 cache
            generate_steps = 200  # 增加到30步，生成更长的序列
            cache = self.main_model.model.init_cache(
                self.main_model.qwen2_config,
                batch_size,
                num_groups,
                generate_steps=generate_steps,
                dtype=jnp.bfloat16,
            )

            # 执行推理
            self._print("\n开始推理...")
            start_time = time.time()

            # Prefill - 使用正确的pad_id
            pad_id = text_tokenizer.pad_token_id if text_tokenizer else 0
            text_logits, local_hidden_states, cache = forward_jit(
                self.main_model, input_ids, cache, pad_id
            )

            self._print(f"Prefill 完成")
            self._print(f"  - 文本 logits shape: {text_logits.shape}")
            self._print(f"  - 局部隐藏状态 shape: {local_hidden_states.shape}")

            # 生成循环
            generated_text_tokens = []
            generated_audio_tokens_list = []

            # 统计信息（用于诊断）
            num_empty_idx_generated = 0  # 生成了多少个empty_idx（对应音频）
            num_text_token_generated = 0  # 生成了多少个文本token

            # 创建 sampler（使用官方推荐参数）
            from bonsai.models.mimo_audio.modeling import MiMoSampler, MiMoSamplerConfig
            text_sampler = MiMoSampler(MiMoSamplerConfig(temperature=0.6,  top_p=1.0, do_sample=True))
            audio_sampler = MiMoSampler(MiMoSamplerConfig(temperature=0.9,  top_p=0.95, do_sample=True))
            # text_sampler = MiMoSampler(MiMoSamplerConfig(temperature=0.6, top_k=50, top_p=1.0, do_sample=False))
            # audio_sampler = MiMoSampler(MiMoSamplerConfig(temperature=0.9, top_k=50, top_p=0.95, do_sample=False))

            # Random key for sampling
            rng_key = jax.random.key(42)

            for step in range(generate_steps):
                # ============================================================
                # 诊断策略：追踪为什么音频后半段是无意义的语音
                # 1. 追踪连续的EMPTY token数量（EMPTY→生成音频）
                # 2. 每10步显示token类型和是否生成音频
                # 3. 每20步显示模型置信度，特别是EOSTM的概率
                # 4. 检查是否正常生成EOSTM停止，还是达到max_steps
                # ============================================================

                # 调试：打印logits统计（仅第一步）
                if step == 0:
                    self._print(f"\n调试信息（第1步）：")
                    self._print(f"  text_logits shape: {text_logits.shape}")
                    self._print(f"  text_logits 统计：均值={float(jnp.mean(text_logits)):.4f}, "
                               f"标准差={float(jnp.std(text_logits)):.4f}, "
                               f"最小值={float(jnp.min(text_logits)):.4f}, "
                               f"最大值={float(jnp.max(text_logits)):.4f}")

                    # 检查top-k概率
                    probs = jax.nn.softmax(text_logits[0, 0], axis=-1)

                # 采样下一个文本 token - 修正形状为2D
                key, subkey = jax.random.split(rng_key)
                logits_2d = text_logits[0, 0:1, :]  # [1, vocab_size]
                next_text_token = text_sampler.sample(logits_2d, subkey)
                next_text_token_int = int(next_text_token[0])
                generated_text_tokens.append(next_text_token_int)

                # 诊断：每20步显示模型对当前token的置信度
                if step % 20 == 0:
                    probs = jax.nn.softmax(text_logits[0, 0, :].astype(jnp.float32))
                    top5_indices = jnp.argsort(probs)[-5:][::-1]
                    top5_probs = probs[top5_indices]
                    self._print(f"\n  [步骤 {step + 1} 模型置信度]")
                    self._print(f"    选中token: {next_text_token_int} (概率={float(probs[next_text_token_int]):.4f})")
                    eostm_idx = self.main_model.args.eostm_idx
                    eostm_prob = float(probs[eostm_idx])
                    self._print(f"    EOSTM({eostm_idx})概率: {eostm_prob:.4f}")


                # 统计token类型
                empty_idx = self.main_model.args.empty_idx
                if next_text_token_int == empty_idx:
                    num_empty_idx_generated += 1
                elif next_text_token_int == self.main_model.args.eostm_idx or \
                     (text_tokenizer and next_text_token_int == text_tokenizer.eos_token_id):
                    pass  # 停止token，不计数
                else:
                    num_text_token_generated += 1

                # 每步都检查并打印（诊断用）
                token_type = "EOSTM" if next_text_token_int == self.main_model.args.eostm_idx else \
                            "EMPTY" if next_text_token_int == empty_idx else \
                            "EOS" if (text_tokenizer and next_text_token_int == text_tokenizer.eos_token_id) else \
                            "TEXT"

                # 跟踪连续的empty token数量
                if token_type == "EMPTY":
                    if not hasattr(self, '_consecutive_empty_count'):
                        self._consecutive_empty_count = 0
                    self._consecutive_empty_count += 1
                else:
                    if hasattr(self, '_consecutive_empty_count'):
                        if self._consecutive_empty_count > 0:
                            self._print(f"  [连续生成了 {self._consecutive_empty_count} 个EMPTY token]")
                    self._consecutive_empty_count = 0

                # 每10步或遇到特殊token时打印
                if step % 10 == 0 or token_type in ["EOSTM", "EOS"]:
                    audio_info = "→生成音频" if token_type == "EMPTY" else "→无音频"
                    self._print(f"  步骤 {step + 1}: token={next_text_token_int} ({token_type}) {audio_info}")

                # 检查是否生成了停止token（EOSTM或EOS）
                if next_text_token_int == self.main_model.args.eostm_idx:
                    self._print(f"\n  生成了EOSTM token，停止生成（步骤{step + 1}）")
                    break
                if text_tokenizer and next_text_token_int == text_tokenizer.eos_token_id:
                    self._print(f"\n  生成了EOS token，停止生成（步骤{step + 1}）")
                    break

                # 关键逻辑：只有当文本token是empty_idx时，才生成音频
                empty_idx = self.main_model.args.empty_idx
                audio_tokens = None  # 初始化

                if next_text_token_int != empty_idx:
                    # 不生成音频，为整个 group 的所有时间步填充 empty_ids
                    if step == 0:
                        self._print(f"\n  文本token不是empty_idx，使用empty_ids填充音频")
                        self._print(f"  生成的文本token: {next_text_token_int}")

                    # ✅ 修复：保存 group_size 个时间步（而不是只保存1个）
                    for t in range(group_size):
                        audio_tokens_step = jnp.array(self.main_model.speech_empty_ids)
                        generated_audio_tokens_list.append(audio_tokens_step)

                    if step == 0:
                        self._print(f"  已为 {group_size} 个时间步填充 empty_ids")
                else:
                    # 生成音频tokens
                    # ✅ 关键修复：local_forward 现在在内部创建自己的 cache（每次都是新的）
                    # 这与官方实现一致：past_key_values = DynamicCache() 每次都是新的
                    key, subkey = jax.random.split(key)
                    audio_tokens = self.main_model.local_forward(
                        local_hidden_states,  # [B, 1, local_dim]
                        subkey,
                        audio_sampler
                    )  # Returns [B, group_size, audio_channels]

                    # ✅ 修复：保存所有 group_size 个时间步（而不是只保存第一个）
                    for t in range(group_size):
                        audio_tokens_step = audio_tokens[0, t, :]  # [audio_channels]
                        generated_audio_tokens_list.append(audio_tokens_step)

                rng_key = key

                # 准备下一步输入 - 必须是 group_size 的倍数
                next_input = jnp.zeros((batch_size, audio_channels + 1, group_size), dtype=jnp.int32)

                # 填充文本通道：重复当前 token group_size 次
                for i in range(group_size):
                    next_input = next_input.at[0, 0, i].set(next_text_token[0])

                # 填充音频通道：
                if audio_tokens is None:
                    # 如果没有生成音频，音频通道用empty_ids填充
                    for ch in range(audio_channels):
                        channel_empty_id = self.main_model.speech_empty_ids[ch]
                        for i in range(group_size):
                            next_input = next_input.at[0, ch + 1, i].set(channel_empty_id)
                else:
                    for ch in range(audio_channels):
                        for i in range(group_size):
                            next_input = next_input.at[0, ch + 1, i].set(audio_tokens[0, i, ch])

                # 继续生成
                text_logits, local_hidden_states, cache = forward_jit(
                    self.main_model, next_input, cache, pad_id
                )

            inference_time = time.time() - start_time
            self._print(f"\n推理完成，总耗时: {inference_time:.3f}秒")

            # 诊断信息：检查是否正常停止
            if hasattr(self, '_consecutive_empty_count') and self._consecutive_empty_count > 0:
                self._print(f"  [最后连续生成了 {self._consecutive_empty_count} 个EMPTY token]")
            if step + 1 >= generate_steps:
                self._print(f"  ⚠️  达到最大步数 {generate_steps}，可能没有生成EOSTM token")

            # 统计报告
            self._print("\n" + "=" * 70)
            self._print("生成统计")
            self._print("=" * 70)
            self._print(f"生成的token类型分布:")
            self._print(f"  - Empty_idx（对应音频）: {num_empty_idx_generated} 个")
            self._print(f"  - 文本token: {num_text_token_generated} 个")
            self._print(f"  - 总共生成: {len(generated_text_tokens)} 个token")


            # 解析文本结果
            self._print("\n" + "=" * 70)
            self._print("生成结果")
            self._print("=" * 70)

            self._print("\n【文本输出】")
            if text_tokenizer:
                try:
                    generated_text = text_tokenizer.decode(generated_text_tokens, skip_special_tokens=True)
                    self._print(f"生成的文本: {generated_text}")
                except Exception as e:
                    self._print(f"解码文本失败: {e}")
                    self._print(f"生成的 tokens: {generated_text_tokens}")
            else:
                self._print(f"生成的 tokens: {generated_text_tokens}")

            # 保存音频
            self._print("\n【音频输出】")
            try:
                output_dir = "test_outputs"
                os.makedirs(output_dir, exist_ok=True)

                # 将 audio tokens 转换为音频
                # generated_audio_tokens_list: list of [audio_channels] arrays
                audio_tokens_array = jnp.stack(generated_audio_tokens_list, axis=0)  # [time, channels]
                audio_tokens_array = audio_tokens_array.T  # [channels, time]

                self._print(f"原始音频 tokens 形状: {audio_tokens_array.shape}")
                self._print(f"  - 通道数: {audio_tokens_array.shape[0]}")
                self._print(f"  - 时间步: {audio_tokens_array.shape[1]}")

                # ✅ 关键修复：过滤掉所有的empty_id时间步（不仅是前面，中间也可能有）
                # 只保留真正生成音频的时间步
                speech_empty_ids = self.main_model.speech_empty_ids

                # 标记每个时间步是否为真实音频
                # 真实音频 = 至少有一个通道不是empty_id
                is_real_audio_mask = jnp.zeros(audio_tokens_array.shape[1], dtype=bool)

                for ch in range(audio_channels):
                    empty_id = speech_empty_ids[ch]
                    # 这个通道不是empty的时间步
                    not_empty = audio_tokens_array[ch, :] != empty_id
                    is_real_audio_mask = is_real_audio_mask | not_empty

                # 统计有多少时间步是真实音频
                num_real_audio = int(jnp.sum(is_real_audio_mask))
                num_empty = audio_tokens_array.shape[1] - num_real_audio

                self._print(f"\n音频tokens分析:")
                self._print(f"  总时间步: {audio_tokens_array.shape[1]}")
                self._print(f"  真实音频时间步: {num_real_audio}")
                self._print(f"  Empty时间步: {num_empty}")

                # 可视化哪些位置是真实音频
                viz = ""
                for i in range(audio_tokens_array.shape[1]):
                    if is_real_audio_mask[i]:
                        viz += "■"  # 真实音频
                    else:
                        viz += "□"  # Empty
                self._print(f"  时间步可视化: {viz}")
                self._print(f"    (■=真实音频  □=empty)")


                if num_real_audio == 0:
                    self._print(f"\n⚠️  警告：所有时间步都是empty_id，没有真实音频内容！")
                else:
                    # 只保留真实音频的时间步
                    audio_tokens_array = audio_tokens_array[:, is_real_audio_mask]

                    self._print(f"✅ 过滤后只保留真实音频部分")
                    self._print(f"   过滤后 tokens 形状: {audio_tokens_array.shape}")

                # ✅ 关键修复：与官方实现一致，直接使用8个通道的codes
                # 官方代码：codes = tokens.reshape(-1, self.audio_channels).T  # [audio_channels, time]
                # tokenizer会使用前8个量化器层（layers 0-7）进行解码
                codes_for_decoder = audio_tokens_array  # [8, time] - 直接使用，不需要扩展到20层

                self._print(f"解码器输入 codes 形状: {codes_for_decoder.shape}")
                self._print(f"Token 值范围: [{int(codes_for_decoder.min())}, {int(codes_for_decoder.max())}]")

                # 打印每个通道的token统计
                self._print(f"\n各通道token统计:")
                speech_empty_ids = self.main_model.speech_empty_ids
                all_channels_good = True

                for ch in range(8):
                    ch_tokens = codes_for_decoder[ch, :]
                    unique_vals = jnp.unique(ch_tokens)
                    unique_count = len(unique_vals)

                    empty_id = speech_empty_ids[ch]

                    self._print(f"  通道{ch}: 最小={int(ch_tokens.min())}, "
                               f"最大={int(ch_tokens.max())}, "
                               f"均值={float(ch_tokens.mean()):.2f}, "
                               f"唯一值={unique_count}")

                    # 质量检查
                    if unique_count == 1:
                        if int(ch_tokens[0]) == empty_id:
                            self._print(f"    ❌ 全是empty_id ({empty_id})，没有实际音频内容")
                            all_channels_good = False
                        else:
                            self._print(f"    ⚠️ 所有tokens都是同一个值: {int(ch_tokens[0])}")
                            all_channels_good = False
                    elif unique_count < 5:
                        self._print(f"    ⚠️ 多样性很低，只有{unique_count}个不同值")
                    else:
                        self._print(f"    ✅ 有足够多样性")


                decoded_audio = self.tokenizer_model.decode(codes_for_decoder)
                self._print(f"解码后音频形状: {decoded_audio.shape}")

                # 保存音频
                try:
                    import soundfile as sf
                    audio_path = os.path.join(output_dir, "generated_audio.wav")
                    audio_np = np.array(decoded_audio[0, 0, :])
                    sample_rate = self.tokenizer_config.sampling_rate
                    sf.write(audio_path, audio_np, sample_rate)

                    audio_duration = len(audio_np) / sample_rate
                    self._print(f"✅ 已保存生成的音频: {audio_path}")
                    self._print(f"   - 时长: {audio_duration:.2f}秒")
                    self._print(f"   - 采样点: {len(audio_np)}")
                    self._print(f"   - 采样率: {sample_rate} Hz")
                    self._print(f"   - 音频统计: 均值={audio_np.mean():.4f}, 标准差={audio_np.std():.4f}")

                except ImportError:
                    self._print("soundfile 不可用，跳过音频保存", "WARNING")
                except Exception as e:
                    self._print(f"保存音频失败: {e}", "WARNING")

            except Exception as e:
                self._print(f"音频生成失败: {type(e).__name__}: {e}", "WARNING")
                import traceback
                traceback.print_exc()

            # 保存完整结果
            try:
                output_path = os.path.join(output_dir, "inference_results.npz")
                np.savez(
                    output_path,
                    generated_text_tokens=np.array(generated_text_tokens),
                    generated_audio_tokens=np.array(generated_audio_tokens_list),
                    input_ids=np.array(input_ids),
                )
                self._print(f"\n✅ 已保存推理结果: {output_path}")

            except Exception as e:
                self._print(f"保存结果失败: {e}", "WARNING")

            self.test_results["full_inference"] = {
                "success": True,
                "inference_time": inference_time,
                "generated_tokens": len(generated_text_tokens),
            }

            self._print("\n完整推理测试通过", "SUCCESS")

            # 清理内存
            self._clear_memory()
            return True

        except Exception as e:
            self._print(f"完整推理测试失败: {type(e).__name__}: {e}", "ERROR")
            self.test_results["full_inference"] = {
                "success": False,
                "error": str(e),
            }
            import traceback
            traceback.print_exc()
            return False

    def run_all_tests(self) -> int:
        """运行所有端到端测试"""
        self._print("MiMo Audio 端到端测试", "SECTION")
        self._print(f"模型路径: {self.model_path}")
        self._print(f"Tokenizer 路径: {self.tokenizer_path}")

        # Run tests in sequence
        # self.test_tokenizer_loading()
        # self.test_tokenizer_forward()
        # self.test_mel_extraction()
        # self.test_audio_reconstruction()
        # self.test_tokenizer_various_lengths()
        # self.test_main_model_loading()
        # self.test_main_model_forward()
        # self.test_input_output_shapes()
        # self.test_output_distributions()
        self.test_full_inference()  # 新增完整推理测试

        # Print summary
        return self.print_summary()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="MiMo Audio End-to-End Tests")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to main model (default: ModelScope cache)",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to tokenizer (default: ModelScope cache)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()

    tester = EndToEndTester(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        verbose=not args.quiet,
    )

    return tester.run_all_tests()


if __name__ == "__main__":
    exit(main())
