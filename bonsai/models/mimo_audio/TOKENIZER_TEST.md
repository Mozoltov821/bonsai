# MiMo-Audio Tokenizer Test Guide

测试 MiMo-Audio Tokenizer 的编码和解码功能。

## 功能测试

测试脚本会进行以下测试：

1. **加载 Tokenizer** - 验证模型和配置加载
2. **音频编码** - 将音频转换为 token 序列
3. **音频解码** - 将 token 序列重建为音频
4. **往返质量** - 比较原始音频和重建音频的质量

## 使用方法

### 使用 ModelScope 缓存

```bash
python bonsai/models/mimo_audio/test_tokenizer.py \
    --tokenizer_id xiaomi/MiMo-Audio-Tokenizer \
    --audio_path your_audio.wav \
    --output_path reconstructed.wav
```

### 使用显式路径

```bash
python bonsai/models/mimo_audio/test_tokenizer.py \
    --tokenizer_path /path/to/tokenizer \
    --audio_path your_audio.wav
```

### 禁用 Sharding

```bash
python bonsai/models/mimo_audio/test_tokenizer.py \
    --tokenizer_id xiaomi/MiMo-Audio-Tokenizer \
    --audio_path your_audio.wav \
    --no_sharding
```

## 参数说明

- `--tokenizer_id` - ModelScope tokenizer ID（例如 `xiaomi/MiMo-Audio-Tokenizer`）
- `--tokenizer_path` - Tokenizer 路径（替代 --tokenizer_id）
- `--audio_path` - 输入音频文件（必需）
- `--output_path` - 输出音频文件（默认：`reconstructed.wav`）
- `--no_sharding` - 禁用 sharding

## 输出示例

```
============================================================
MiMo-Audio Tokenizer Test Suite
============================================================
Input audio: test.wav
Output audio: reconstructed.wav
Sharding: Enabled

============================================================
Test 1: Tokenizer Loading
============================================================
Path: ~/.cache/modelscope/hub/xiaomi/MiMo-Audio-Tokenizer

Config:
  - Encoder layers: 32
  - Decoder layers: 32
  - Vocoder layers: 16
  - Quantizers: 12
  - Sample rate: 24000 Hz
  - Mel bins: 128
  - FFT size: 960

✅ Tokenizer loaded in 15.23s
  - Sharding: Enabled

============================================================
Test 2: Audio Encoding
============================================================

Loading audio: test.wav
  - Original sample rate: 16000 Hz
  - Duration: 3.50s
  - Samples: 56000
  - Resampled to: 24000 Hz

Computing mel spectrogram...
  - Mel shape: (350, 128)
  - Mel range: [-7.234, 2.145]

Encoding to tokens...
  - Codes shape: (12, 87)
  - Num quantizers: 12
  - Sequence length: 87
  - Code range: [0, 1023]
  - Encoding time: 0.234s
  - Compression ratio: 812.6x

✅ Encoding successful

============================================================
Test 3: Audio Decoding
============================================================
Input codes shape: (12, 87)

Decoding tokens to audio...
  - Output shape: (83712,)
  - Output range: [-0.876, 0.921]
  - Decoding time: 0.189s

✅ Decoding successful
  - Output saved: reconstructed.wav (328.45 KB)

============================================================
Test 4: Roundtrip Quality
============================================================
Quality metrics:
  - MSE: 0.002134
  - RMSE: 0.046193
  - SNR: 22.45 dB
  - Correlation: 0.9823

✅ Overall quality: Excellent

============================================================
✅ All tests passed!
============================================================

Summary:
  - Input: test.wav
  - Output: reconstructed.wav
  - Quantizers: 12
  - Quality (SNR): 22.45 dB
```

## 质量指标说明

### SNR (Signal-to-Noise Ratio) - 信噪比
- \> 20 dB: **Excellent** - 优秀
- 15-20 dB: **Good** - 良好
- 10-15 dB: **Fair** - 一般
- < 10 dB: **Poor** - 较差

### Correlation - 相关性
- \> 0.95: 非常高的相似度
- 0.90-0.95: 高相似度
- 0.80-0.90: 中等相似度
- < 0.80: 较低相似度

### Compression Ratio - 压缩比
- 表示音频被压缩的倍数
- 典型值：500x - 1000x
- 越高表示压缩效率越好

## 注意事项

1. **音频格式支持**：
   - 支持所有 `soundfile` 库支持的格式（.wav, .flac, .ogg 等）
   - 自动转换为单声道
   - 自动重采样到 24000 Hz

2. **Sharding**：
   - 多设备时默认启用 tensor parallelism
   - Mesh shape: (1, num_devices)
   - 单设备时自动禁用

3. **内存需求**：
   - 取决于音频长度和 tokenizer 大小
   - 建议音频不超过 30 秒用于测试

## 故障排除

### 找不到模型
```
Error: Model not found in modelscope cache: xiaomi/MiMo-Audio-Tokenizer
```

**解决方案**：先下载模型
```python
from modelscope import snapshot_download
snapshot_download('xiaomi/MiMo-Audio-Tokenizer')
```

### 音频文件错误
```
Error: Audio file not found: test.wav
```

**解决方案**：检查音频文件路径是否正确

### 内存不足
如果遇到内存问题，可以：
1. 使用 `--no_sharding` 禁用 sharding
2. 使用较短的音频文件进行测试
3. 减少可用设备数量
