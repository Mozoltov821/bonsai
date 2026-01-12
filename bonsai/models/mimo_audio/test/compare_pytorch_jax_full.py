"""
完整对比 PyTorch 和 JAX 版本的 MiMo Audio Tokenizer
逐阶段验证，找出第一次出现差异的位置
"""
import os
import sys
import json
import numpy as np
import torch
import jax
import jax.numpy as jnp
from flax import nnx

print("=" * 80)
print("MiMo Audio Tokenizer: PyTorch vs JAX 完整对比")
print("=" * 80)

# ============================================================================
# 配置和路径
# ============================================================================
tokenizer_path = os.path.expanduser("~/.cache/modelscope/hub/models/XiaomiMiMo/MiMo-Audio-Tokenizer")
config_path = os.path.join(tokenizer_path, "config.json")
safetensors_path = os.path.join(tokenizer_path, "model.safetensors")

print(f"\n[配置]")
print(f"  Tokenizer 路径: {tokenizer_path}")

# 加载配置
with open(config_path) as f:
    config_dict = json.load(f)

print(f"  采样率: {config_dict['sampling_rate']} Hz")
print(f"  量化器层数: {config_dict['num_quantizers']}")
print(f"  n_fft: {config_dict['nfft']}")
print(f"  hop_length: {config_dict['hop_length']}")

# ============================================================================
# 加载 PyTorch 模型
# ============================================================================
print(f"\n[1] 加载 PyTorch 模型...")

# 添加 PyTorch 实现的路径
pytorch_impl_path = os.path.join(os.path.dirname(__file__), "../origin-mimo")
sys.path.insert(0, pytorch_impl_path)

try:
    # 先导入以便 patch
    import mimo_audio_tokenizer.modeling_audio_tokenizer as pytorch_modeling

    # Patch flash_attn_varlen_func 使用标准 attention
    def standard_attention_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                                dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1)):
        """标准 attention 实现，替代 flash_attn_varlen_func"""
        # q, k, v 格式: (total_seq_len, num_heads, head_dim)
        # cu_seqlens: 累积序列长度

        # 简化：假设 batch_size = 1
        # 重塑为 (batch, seq_len, num_heads, head_dim)
        batch_size = len(cu_seqlens_q) - 1

        # 分离各个 batch
        outputs = []
        for i in range(batch_size):
            start_q = cu_seqlens_q[i]
            end_q = cu_seqlens_q[i + 1]
            start_k = cu_seqlens_k[i]
            end_k = cu_seqlens_k[i + 1]

            q_batch = q[start_q:end_q]  # (seq_len, num_heads, head_dim)
            k_batch = k[start_k:end_k]
            v_batch = v[start_k:end_k]

            # 转置为 (num_heads, seq_len, head_dim)
            q_batch = q_batch.transpose(0, 1)
            k_batch = k_batch.transpose(0, 1)
            v_batch = v_batch.transpose(0, 1)

            # 计算 attention scores
            scale = softmax_scale if softmax_scale is not None else (q_batch.shape[-1] ** -0.5)
            attn_weights = torch.matmul(q_batch, k_batch.transpose(-2, -1)) * scale  # (num_heads, seq_len_q, seq_len_k)

            # 应用 causal mask
            if causal:
                seq_len_q, seq_len_k = attn_weights.shape[-2], attn_weights.shape[-1]
                causal_mask = torch.triu(torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=attn_weights.device), diagonal=1)
                attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

            # Softmax
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_batch.dtype)

            # Dropout
            if dropout_p > 0:
                attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_p)

            # 计算输出
            output = torch.matmul(attn_weights, v_batch)  # (num_heads, seq_len_q, head_dim)
            output = output.transpose(0, 1)  # (seq_len_q, num_heads, head_dim)

            outputs.append(output)

        # 拼接所有 batch
        return torch.cat(outputs, dim=0)

    # 替换 flash_attn_varlen_func
    pytorch_modeling.flash_attn_varlen_func = standard_attention_func
    print("  ✅ 已 patch flash_attn_varlen_func 使用标准 attention")

    # Patch AudioEncoder 的 __init__ 以禁用 flash_attention
    original_init = pytorch_modeling.AudioEncoder.__init__

    def patched_init(self, config):
        # 调用原始 __init__ 前先设置 attn_implementation
        config._attn_implementation = "eager"  # 使用标准 PyTorch attention
        original_init(self, config)

    pytorch_modeling.AudioEncoder.__init__ = patched_init
    print("  ✅ 已 patch AudioEncoder 使用标准 attention")

    from mimo_audio_tokenizer import MiMoAudioTokenizer as PyTorchTokenizer

    pytorch_model = PyTorchTokenizer.from_pretrained(tokenizer_path)
    pytorch_model.eval()
    print("  ✅ PyTorch 模型加载成功")
except Exception as e:
    print(f"  ❌ PyTorch 模型加载失败: {e}")
    print(f"  请确保 {pytorch_impl_path} 路径正确")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 加载 JAX 模型
# ============================================================================
print(f"\n[2] 加载 JAX 模型...")

from bonsai.models.mimo_audio.mimo_audio_tokenizer import MiMoAudioTokenizerConfig
from bonsai.models.mimo_audio.mimo_audio_tokenizer_params import load_tokenizer_weights_from_safetensors

jax_config = MiMoAudioTokenizerConfig(**config_dict)
jax_model = load_tokenizer_weights_from_safetensors(
    config=jax_config,
    safetensors_path=safetensors_path,
    dtype=jnp.float32,
    rngs=nnx.Rngs(0),
)
print("  ✅ JAX 模型加载成功")

# ============================================================================
# 生成测试音频
# ============================================================================
print(f"\n[3] 生成测试音频...")

sampling_rate = config_dict['sampling_rate']
duration = 1.0
frequency = 440.0

t = np.linspace(0, duration, int(sampling_rate * duration))
waveform = np.sin(2 * np.pi * frequency * t).astype(np.float32)

print(f"  频率: {frequency} Hz")
print(f"  时长: {duration} s")
print(f"  采样点数: {len(waveform)}")
print(f"  幅度范围: [{waveform.min():.4f}, {waveform.max():.4f}]")

# ============================================================================
# 阶段 1: Mel Spectrogram
# ============================================================================
print(f"\n[阶段 1] Mel Spectrogram")
print("-" * 80)

from bonsai.models.mimo_audio.melSpectrogram import MelSpectrogram

mel_extractor = MelSpectrogram(
    sample_rate=sampling_rate,
    n_fft=config_dict['nfft'],
    hop_length=config_dict['hop_length'],
    win_length=config_dict['window_size'],
    f_min=config_dict['fmin'],
    f_max=config_dict.get('fmax', sampling_rate // 2),
    n_mels=config_dict['n_mels'],
)

# JAX mel spectrogram
jax_mel = mel_extractor(jnp.array(waveform))
jax_mel_input = jax_mel.T[jnp.newaxis, :, :].astype(jnp.bfloat16)

# PyTorch mel spectrogram (假设 PyTorch 模型内部也用类似的处理)
# 为了公平对比，使用相同的 mel 输入
pytorch_mel_input = torch.from_numpy(np.array(jax_mel_input))

print(f"  JAX mel shape: {jax_mel_input.shape}")
print(f"  JAX mel range: [{float(jax_mel_input.min()):.4f}, {float(jax_mel_input.max()):.4f}]")
print(f"  PyTorch mel shape: {pytorch_mel_input.shape}")
print(f"  PyTorch mel range: [{pytorch_mel_input.min():.4f}, {pytorch_mel_input.max():.4f}]")

mel_diff = np.abs(np.array(jax_mel_input) - pytorch_mel_input.numpy())
print(f"  Mel 差异: max={mel_diff.max():.6f}, mean={mel_diff.mean():.6f}")

if mel_diff.max() < 1e-5:
    print(f"  ✅ Mel spectrogram 一致")
else:
    print(f"  ⚠️  Mel spectrogram 有差异")

# ============================================================================
# 阶段 2: Encoder (含 Quantizer)
# ============================================================================
print(f"\n[阶段 2] Encoder + Quantizer")
print("-" * 80)

# JAX encode
mel_lengths = jnp.array([jax_mel_input.shape[1]])
jax_encoder_output = jax_model.encode(jax_mel_input, mel_lengths, use_quantizer=True)
jax_codes = jax_encoder_output.codes

print(f"  JAX codes shape: {jax_codes.shape}")
print(f"  JAX codes range: [{int(jax_codes.min())}, {int(jax_codes.max())}]")
print(f"  JAX codes unique: {len(jnp.unique(jax_codes))}")

# PyTorch encode
with torch.no_grad():
    try:
        # PyTorch API: encode(mels, input_lens, use_quantizer=True)
        # 返回: (hidden_states, hidden_states_packed, encoder_output_length, codes)
        pytorch_input_lens = torch.tensor([pytorch_mel_input.shape[1]])
        pytorch_hidden, pytorch_hidden_packed, pytorch_output_len, pytorch_codes = pytorch_model.encode(
            pytorch_mel_input, pytorch_input_lens, use_quantizer=True
        )

        print(f"  PyTorch codes shape: {pytorch_codes.shape}")
        print(f"  PyTorch codes range: [{int(pytorch_codes.min())}, {int(pytorch_codes.max())}]")
        print(f"  PyTorch codes unique: {len(torch.unique(pytorch_codes))}")

        codes_diff = np.abs(np.array(jax_codes) - pytorch_codes.cpu().numpy())
        codes_match_rate = (np.array(jax_codes) == pytorch_codes.cpu().numpy()).mean()

        print(f"  Codes 匹配率: {codes_match_rate * 100:.2f}%")

        if codes_match_rate > 0.95:
            print(f"  ✅ Encoder/Quantizer 基本一致")
        else:
            print(f"  ❌ Encoder/Quantizer 有显著差异")
            print(f"  ⚠️  这可能导致后续所有阶段不同")
    except Exception as e:
        print(f"  ⚠️  PyTorch encode 失败: {e}")
        import traceback
        traceback.print_exc()
        print(f"  ⚠️  使用 JAX codes 继续测试（无法对比 encoder）")
        pytorch_codes = torch.from_numpy(np.array(jax_codes))

# ============================================================================
# 阶段 3: Decoder - decode_vq
# ============================================================================
print(f"\n[阶段 3] Decoder - decode_vq (ResidualVectorQuantizer)")
print("-" * 80)

# JAX decode_vq
jax_hidden = jax_model.encoder.decode_vq(jax_codes)

print(f"  JAX hidden shape: {jax_hidden.shape}")
print(f"  JAX hidden dtype: {jax_hidden.dtype}")
print(f"  JAX hidden range: [{float(jax_hidden.min()):.4f}, {float(jax_hidden.max()):.4f}]")
print(f"  JAX hidden stats: mean={float(jax_hidden.mean()):.4f}, std={float(jax_hidden.std()):.4f}")

# PyTorch decode_vq
with torch.no_grad():
    try:
        # PyTorch API: encoder.decode_vq(codes)
        pytorch_hidden_decoded = pytorch_model.encoder.decode_vq(pytorch_codes)

        print(f"  PyTorch hidden shape: {pytorch_hidden_decoded.shape}")
        print(f"  PyTorch hidden dtype: {pytorch_hidden_decoded.dtype}")
        print(f"  PyTorch hidden range: [{pytorch_hidden_decoded.min():.4f}, {pytorch_hidden_decoded.max():.4f}]")
        print(f"  PyTorch hidden stats: mean={pytorch_hidden_decoded.mean():.4f}, std={pytorch_hidden_decoded.std():.4f}")

        hidden_diff = np.abs(np.array(jax_hidden) - pytorch_hidden_decoded.cpu().numpy())
        print(f"  Hidden 差异: max={hidden_diff.max():.6f}, mean={hidden_diff.mean():.6f}")

        if hidden_diff.max() < 1e-4:
            print(f"  ✅ decode_vq 一致")
        else:
            print(f"  ⚠️  decode_vq 有差异")
    except Exception as e:
        print(f"  ⚠️  PyTorch decode_vq 失败: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# 阶段 4: AudioDecoder - dconv1 (上采样)
# ============================================================================
print(f"\n[阶段 4] AudioDecoder - dconv1 (上采样)")
print("-" * 80)

# JAX dconv1
jax_hidden_batch = jax_hidden[None, ...]
lengths = jnp.array([jax_hidden.shape[0]])

if jax_model.decoder.dconv1 is not None:
    jax_x, jax_lengths_up = jax_model.decoder.dconv1(jax_hidden_batch, lengths)
    print(f"  JAX dconv1 output shape: {jax_x.shape}")
    print(f"  JAX dconv1 output range: [{float(jax_x.min()):.4f}, {float(jax_x.max()):.4f}]")
    print(f"  JAX dconv1 output stats: mean={float(jax_x.mean()):.4f}, std={float(jax_x.std()):.4f}")
else:
    jax_x = jax_hidden_batch
    jax_lengths_up = lengths

# PyTorch dconv1
with torch.no_grad():
    try:
        pytorch_hidden_batch = pytorch_hidden.unsqueeze(0).transpose(1, 2)  # 调整维度

        if hasattr(pytorch_model.decoder, 'dconv1') and pytorch_model.decoder.dconv1 is not None:
            pytorch_x = pytorch_model.decoder.dconv1(pytorch_hidden_batch)

            print(f"  PyTorch dconv1 output shape: {pytorch_x.shape}")
            print(f"  PyTorch dconv1 output range: [{pytorch_x.min():.4f}, {pytorch_x.max():.4f}]")
            print(f"  PyTorch dconv1 output stats: mean={pytorch_x.mean():.4f}, std={pytorch_x.std():.4f}")

            # 对比
            dconv1_diff = np.abs(np.array(jax_x) - pytorch_x.numpy())
            print(f"  dconv1 差异: max={dconv1_diff.max():.6f}, mean={dconv1_diff.mean():.6f}")

            if dconv1_diff.max() < 1e-3:
                print(f"  ✅ dconv1 一致")
            else:
                print(f"  ⚠️  dconv1 有差异")
    except Exception as e:
        print(f"  ⚠️  PyTorch dconv1 对比失败: {e}")

# ============================================================================
# 阶段 5: AudioDecoder - Transformer layers
# ============================================================================
print(f"\n[阶段 5] AudioDecoder - Transformer layers")
print("-" * 80)

from bonsai.models.mimo_audio.mimo_audio_tokenizer import make_sequence_mask, get_position_ids

# JAX transformer
mask = make_sequence_mask(jax_lengths_up, jax_x.shape[1])
pos = get_position_ids(jax_lengths_up, jax_x.shape[1])
rope = jax_model.decoder.position_embedding(jax_x, pos)

for i, layer in enumerate(jax_model.decoder.layers):
    jax_x = layer(jax_x, mask, rope)
    if i == 0:
        print(f"  JAX Layer {i}: mean={float(jax_x.mean()):.4f}, std={float(jax_x.std()):.4f}")
    elif i == len(jax_model.decoder.layers) - 1:
        print(f"  JAX Layer {i}: mean={float(jax_x.mean()):.4f}, std={float(jax_x.std()):.4f}")

jax_x = jax_model.decoder.layer_norm(jax_x)
print(f"  JAX LayerNorm: mean={float(jax_x.mean()):.4f}, std={float(jax_x.std()):.4f}")

# PyTorch transformer
print(f"  ⚠️  PyTorch transformer 逐层对比较复杂，跳过中间层")

# ============================================================================
# 阶段 6: AudioDecoder - dconv2 (coarse mel)
# ============================================================================
print(f"\n[阶段 6] AudioDecoder - dconv2 (生成 coarse mel)")
print("-" * 80)

# JAX dconv2
jax_coarse, jax_mel_lengths = jax_model.decoder.dconv2(jax_x, jax_lengths_up)

print(f"  JAX coarse shape: {jax_coarse.shape}")
print(f"  JAX coarse range: [{float(jax_coarse.min()):.4f}, {float(jax_coarse.max()):.4f}]")
print(f"  JAX coarse stats: mean={float(jax_coarse.mean()):.4f}, std={float(jax_coarse.std()):.4f}")

# ============================================================================
# 阶段 7: Vocoder (mel -> waveform)
# ============================================================================
print(f"\n[阶段 7] Vocoder (mel -> waveform)")
print("-" * 80)

# JAX vocoder
jax_vocoder_output = jax_model.decoder.vocoder(jax_coarse, jax_mel_lengths)
jax_decoded_wav = jax_vocoder_output.wav

print(f"  JAX vocoder output shape: {jax_decoded_wav.shape}")
print(f"  JAX vocoder output range: [{float(jax_decoded_wav.min()):.4f}, {float(jax_decoded_wav.max()):.4f}]")
print(f"  JAX vocoder output stats: mean={float(jax_decoded_wav.mean()):.4f}, std={float(jax_decoded_wav.std()):.4f}")

# PyTorch 完整 decode（从 codes 到 waveform）
with torch.no_grad():
    try:
        # PyTorch API: decode(codes) 返回 dict 或 tensor
        pytorch_decoded_output = pytorch_model.decode(pytorch_codes)

        # 提取 waveform
        if isinstance(pytorch_decoded_output, dict):
            pytorch_decoded_wav = pytorch_decoded_output['wav']
        elif isinstance(pytorch_decoded_output, torch.Tensor):
            pytorch_decoded_wav = pytorch_decoded_output
        else:
            raise ValueError(f"Unexpected output type: {type(pytorch_decoded_output)}")

        print(f"  PyTorch decode output shape: {pytorch_decoded_wav.shape}")
        print(f"  PyTorch decode output range: [{pytorch_decoded_wav.min():.4f}, {pytorch_decoded_wav.max():.4f}]")
        print(f"  PyTorch decode output stats: mean={pytorch_decoded_wav.mean():.4f}, std={pytorch_decoded_wav.std():.4f}")

        # 对齐长度和维度
        # JAX: (B, 1, T)
        # PyTorch: 可能是 (B, T) 或 (B, 1, T)
        jax_wav_np = np.array(jax_decoded_wav[0, 0, :])  # (T,)
        pytorch_wav_np = pytorch_decoded_wav.squeeze().cpu().numpy()  # (T,)

        min_len = min(len(jax_wav_np), len(pytorch_wav_np))
        jax_wav_crop = jax_wav_np[:min_len]
        pytorch_wav_crop = pytorch_wav_np[:min_len]

        wav_diff = np.abs(jax_wav_crop - pytorch_wav_crop)
        print(f"\n  Waveform 差异 (前 {min_len} 采样点):")
        print(f"    Max diff: {wav_diff.max():.6f}")
        print(f"    Mean diff: {wav_diff.mean():.6f}")
        print(f"    Std diff: {wav_diff.std():.6f}")

        # 计算相关系数
        correlation = np.corrcoef(jax_wav_crop, pytorch_wav_crop)[0, 1]
        print(f"    Correlation: {correlation:.6f}")

        # 计算 SNR
        signal_power = np.mean(pytorch_wav_crop ** 2)
        noise_power = np.mean((jax_wav_crop - pytorch_wav_crop) ** 2)

        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
            print(f"    SNR: {snr:.2f} dB")

            if snr > 20:
                print(f"  ✅ Waveform 一致性好 (SNR > 20dB)")
            elif snr > 10:
                print(f"  ⚠️  Waveform 有一定差异 (10dB < SNR < 20dB)")
            else:
                print(f"  ❌ Waveform 差异显著 (SNR < 10dB)")
                print(f"\n  ⚠️  这是主要问题！需要逐阶段排查")
        else:
            print(f"  ⚠️  noise_power = 0, 无法计算 SNR")

    except Exception as e:
        print(f"  ❌ PyTorch decode 失败: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# 总结
# ============================================================================
print(f"\n" + "=" * 80)
print("总结")
print("=" * 80)

print("""
1. 运行此脚本需要确保：
   - PyTorch 版本的 MiMo Audio Tokenizer 在 origin-mimo/ 目录下
   - 两个模型加载同一个权重文件
   - 使用相同的输入

2. 如果发现某个阶段开始出现差异，说明问题在该阶段或之前

3. 重点关注：
   - Codes 匹配率（应该 >95%）
   - decode_vq 输出差异（应该 <1e-4）
   - 最终 waveform 的 SNR（应该 >20dB）

4. 下一步：
   - 找到第一次出现显著差异的阶段
   - 仔细对比该阶段的实现细节
   - 修复差异后重新运行此脚本验证
""")

print("=" * 80)
