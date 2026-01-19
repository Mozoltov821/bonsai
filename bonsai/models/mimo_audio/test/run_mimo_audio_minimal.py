#!/usr/bin/env python3
"""
MiMo Audio 最小推理脚本

从 test_end_to_end.py 提取的核心功能：
- 加载 audio tokenizer
- 加载主模型
- 执行文本转语音推理
"""

import os
import json
import time
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


def load_audio_tokenizer(tokenizer_path: str):
    """加载音频 tokenizer"""
    print(f"📥 加载音频 tokenizer: {tokenizer_path}")

    from bonsai.models.mimo_audio.mimo_audio_tokenizer_configuration import MiMoAudioTokenizerConfig
    from bonsai.models.mimo_audio.mimo_audio_tokenizer_params import load_tokenizer_weights_from_safetensors

    # 加载配置
    config_path = os.path.join(tokenizer_path, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)

    config_dict['use_sharding'] = False  # 单卡模式
    config = MiMoAudioTokenizerConfig(**config_dict)

    print(f"   - 编码器层数: {config.encoder_layers}")
    print(f"   - 解码器层数: {config.decoder_layers}")
    print(f"   - 量化器数量: {config.num_quantizers}")
    print(f"   - 采样率: {config.sampling_rate} Hz")

    # 加载权重
    safetensors_path = os.path.join(tokenizer_path, "model.safetensors")
    start_time = time.time()

    tokenizer_model = load_tokenizer_weights_from_safetensors(
        config=config,
        safetensors_path=safetensors_path,
        dtype=jnp.float32,  # Tokenizer必须用float32
        mesh=None,
        rngs=nnx.Rngs(0),
    )

    load_time = time.time() - start_time
    print(f"✅ Tokenizer 加载完成 ({load_time:.2f}s)\n")

    return tokenizer_model, config


def load_main_model(model_path: str):
    """加载 MiMo Audio 主模型"""
    print(f"📥 加载主模型: {model_path}")

    from bonsai.models.mimo_audio.mimo_audio_configuration import MiMoAudioConfig, MiMoAudioArguments
    from bonsai.models.mimo_audio.params import create_model_with_weights
    from transformers import AutoTokenizer

    # 加载配置
    config_path = os.path.join(model_path, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)

    print(f"   - 模型类型: {config_dict.get('model_type')}")
    print(f"   - 隐藏层大小: {config_dict.get('hidden_size')}")
    print(f"   - 层数: {config_dict.get('num_hidden_layers')}")

    # 创建配置
    config_kwargs = {k: v for k, v in config_dict.items() if k in MiMoAudioConfig.__dataclass_fields__}
    config = MiMoAudioConfig(**config_kwargs)

    # 从tokenizer获取special token IDs
    text_tokenizer = AutoTokenizer.from_pretrained(model_path)

    args = MiMoAudioArguments(
        model_name_or_path=model_path,
        sosp_idx=text_tokenizer.convert_tokens_to_ids("<|sosp|>"),
        eosp_idx=text_tokenizer.convert_tokens_to_ids("<|eosp|>"),
        sostm_idx=text_tokenizer.convert_tokens_to_ids("<|sostm|>"),
        eostm_idx=text_tokenizer.convert_tokens_to_ids("<|eostm|>"),
        eot_idx=text_tokenizer.convert_tokens_to_ids("<|eot|>"),
        empty_idx=text_tokenizer.convert_tokens_to_ids("<|empty|>"),
    )

    print(f"   - SOSTM: {args.sostm_idx}")
    print(f"   - EOSTM: {args.eostm_idx}")
    print(f"   - Empty: {args.empty_idx}")

    # 加载模型
    start_time = time.time()
    model = create_model_with_weights(
        model_path=model_path,
        config=config,
        args=args,
        rngs=nnx.Rngs(0),
        mesh=None,
    )
    load_time = time.time() - start_time

    print(f"✅ 主模型加载完成 ({load_time:.2f}s)\n")

    return model, config, args, text_tokenizer


def insert_between(tokens: list, group_size: int, fill_value: int) -> list:
    """在tokens之间插入填充值"""
    if group_size <= 1:
        return tokens

    result = []
    for token in tokens:
        result.append(token)
        result.extend([fill_value] * (group_size - 1))

    return result


def run_inference(
    main_model,
    tokenizer_model,
    text_tokenizer,
    config,
    args,
    tokenizer_config,
    text_to_speak: str,
    max_steps: int = 100,
    output_dir: str = "test_outputs"
):
    """执行文本转语音推理"""
    print("=" * 70)
    print("🎙️  开始推理")
    print("=" * 70)

    from bonsai.models.mimo_audio.modeling import forward_jit, MiMoSampler
    from bonsai.models.mimo_audio.mimo_audio_configuration import MiMoSamplerConfig

    audio_channels = main_model.audio_channels
    group_size = main_model.group_size
    batch_size = 1

    # 准备输入 - 使用TTS格式
    tts_template = "请将这段文字转换为语音"
    chat_text = f"<|im_start|>user\n{tts_template}: {text_to_speak}<|im_end|>\n<|im_start|>assistant\n<|sostm|>"

    print(f"\n📝 输入文本: {text_to_speak}")
    print(f"   TTS模板: {tts_template}")

    # Tokenize
    text_tokens_raw = text_tokenizer.encode(chat_text)
    text_tokens_with_spacing = insert_between(text_tokens_raw, group_size, -100)

    # 计算num_groups
    num_groups = len(text_tokens_with_spacing) // group_size
    if len(text_tokens_with_spacing) % group_size != 0:
        text_tokens_with_spacing.extend([-100] * (group_size - len(text_tokens_with_spacing) % group_size))
        num_groups = len(text_tokens_with_spacing) // group_size

    print(f"   原始tokens: {len(text_tokens_raw)}")
    print(f"   间隔后tokens: {len(text_tokens_with_spacing)}")
    print(f"   组数: {num_groups}")

    # 创建输入
    input_shape = (batch_size, audio_channels + 1, num_groups * group_size)
    input_ids = jnp.zeros(input_shape, dtype=jnp.int32)

    # 设置文本通道
    input_ids = input_ids.at[0, 0, :].set(jnp.array(text_tokens_with_spacing))

    # 设置音频通道为empty_ids
    for ch in range(1, audio_channels + 1):
        channel_empty_id = main_model.speech_empty_ids[ch - 1]
        audio_empty_tokens = jnp.full((num_groups * group_size,), channel_empty_id, dtype=jnp.int32)
        input_ids = input_ids.at[0, ch, :].set(audio_empty_tokens)

    # 初始化cache
    cache = main_model.model.init_cache(
        main_model.qwen2_config,
        batch_size,
        num_groups,
        generate_steps=max_steps,
        dtype=jnp.bfloat16,
    )

    # JIT预热
    print("\n⚡ JIT预热中...")
    warmup_cache = main_model.model.init_cache(
        main_model.qwen2_config, 1, 1, 0, jnp.bfloat16
    )
    warmup_input = jnp.zeros((1, audio_channels + 1, group_size), dtype=jnp.int32)
    _, _, _ = forward_jit(main_model, warmup_input, warmup_cache, pad_id=0)
    print("✅ JIT预热完成\n")

    # 创建samplers
    text_sampler = MiMoSampler(MiMoSamplerConfig(temperature=0.6, top_p=1.0, do_sample=True))
    audio_sampler = MiMoSampler(MiMoSamplerConfig(temperature=0.9, top_p=0.95, do_sample=True))

    # Prefill
    print("🔄 执行prefill...")
    start_time = time.time()

    pad_id = text_tokenizer.pad_token_id
    text_logits, local_hidden_states, cache = forward_jit(
        main_model, input_ids, cache, pad_id
    )

    print(f"✅ Prefill完成 ({time.time() - start_time:.3f}s)")
    print(f"   文本logits: {text_logits.shape}")
    print(f"   局部隐藏状态: {local_hidden_states.shape}\n")

    # 生成循环
    print(f"🔄 开始生成 (最多{max_steps}步)...\n")

    generated_text_tokens = []
    generated_audio_tokens_list = []

    rng_key = jax.random.key(42)
    empty_idx = args.empty_idx

    for step in range(max_steps):
        # 采样文本token
        key, subkey = jax.random.split(rng_key)
        logits_2d = text_logits[0, 0:1, :]
        next_text_token = text_sampler.sample(logits_2d, subkey)
        next_text_token_int = int(next_text_token[0])
        generated_text_tokens.append(next_text_token_int)

        # 每10步打印进度
        if step % 10 == 0:
            token_type = "EOSTM" if next_text_token_int == args.eostm_idx else \
                        "EMPTY" if next_text_token_int == empty_idx else \
                        "TEXT"
            print(f"   步骤 {step + 1}: token={next_text_token_int} ({token_type})")

        # 检查停止条件
        if next_text_token_int == args.eostm_idx:
            print(f"\n✅ 生成EOSTM，停止生成 (步骤{step + 1})")
            break
        if next_text_token_int == text_tokenizer.eos_token_id:
            print(f"\n✅ 生成EOS，停止生成 (步骤{step + 1})")
            break

        # 生成音频或使用empty_ids
        audio_tokens = None

        if next_text_token_int != empty_idx:
            # 不生成音频
            for t in range(group_size):
                audio_tokens_step = jnp.array(main_model.speech_empty_ids)
                generated_audio_tokens_list.append(audio_tokens_step)
        else:
            # 生成音频
            key, subkey = jax.random.split(key)
            audio_tokens = main_model.local_forward(
                local_hidden_states,
                subkey,
                audio_sampler
            )

            for t in range(group_size):
                audio_tokens_step = audio_tokens[0, t, :]
                generated_audio_tokens_list.append(audio_tokens_step)

        rng_key = key

        # 准备下一步输入
        next_input = jnp.zeros((batch_size, audio_channels + 1, group_size), dtype=jnp.int32)

        # 文本通道
        for i in range(group_size):
            next_input = next_input.at[0, 0, i].set(next_text_token[0])

        # 音频通道
        if audio_tokens is None:
            for ch in range(audio_channels):
                channel_empty_id = main_model.speech_empty_ids[ch]
                for i in range(group_size):
                    next_input = next_input.at[0, ch + 1, i].set(channel_empty_id)
        else:
            for ch in range(audio_channels):
                for i in range(group_size):
                    next_input = next_input.at[0, ch + 1, i].set(audio_tokens[0, i, ch])

        # 继续生成
        text_logits, local_hidden_states, cache = forward_jit(
            main_model, next_input, cache, pad_id
        )

    inference_time = time.time() - start_time
    print(f"\n✅ 推理完成 (总耗时: {inference_time:.3f}s)")
    print(f"   生成了 {len(generated_text_tokens)} 个tokens\n")

    # 解码文本
    print("=" * 70)
    print("📄 生成结果")
    print("=" * 70)

    try:
        generated_text = text_tokenizer.decode(generated_text_tokens, skip_special_tokens=True)
        print(f"\n文本: {generated_text}")
    except Exception as e:
        print(f"\n文本解码失败: {e}")
        print(f"Tokens: {generated_text_tokens}")

    # 处理音频
    print("\n🔊 处理音频...")

    try:
        # 转换为数组
        audio_tokens_array = jnp.stack(generated_audio_tokens_list, axis=0).T
        print(f"   原始tokens: {audio_tokens_array.shape}")

        # 过滤empty_ids
        speech_empty_ids = main_model.speech_empty_ids
        is_real_audio_mask = jnp.zeros(audio_tokens_array.shape[1], dtype=bool)

        for ch in range(audio_channels):
            empty_id = speech_empty_ids[ch]
            not_empty = audio_tokens_array[ch, :] != empty_id
            is_real_audio_mask = is_real_audio_mask | not_empty

        num_real_audio = int(jnp.sum(is_real_audio_mask))
        print(f"   真实音频时间步: {num_real_audio}/{audio_tokens_array.shape[1]}")

        if num_real_audio > 0:
            # 过滤并解码
            audio_tokens_array = audio_tokens_array[:, is_real_audio_mask]
            print(f"   过滤后: {audio_tokens_array.shape}")

            decoded_audio = tokenizer_model.decode(audio_tokens_array)
            print(f"   解码后: {decoded_audio.shape}")

            # 保存音频
            os.makedirs(output_dir, exist_ok=True)

            try:
                import soundfile as sf
                audio_path = os.path.join(output_dir, "generated_audio.wav")
                audio_np = np.array(decoded_audio[0, 0, :])
                sample_rate = tokenizer_config.sampling_rate
                sf.write(audio_path, audio_np, sample_rate)

                audio_duration = len(audio_np) / sample_rate
                print(f"\n✅ 音频已保存: {audio_path}")
                print(f"   时长: {audio_duration:.2f}s")
                print(f"   采样率: {sample_rate} Hz")
            except Exception as e:
                print(f"\n❌ 保存音频失败: {e}")
        else:
            print("\n⚠️  警告: 没有真实音频内容")

    except Exception as e:
        print(f"\n❌ 音频处理失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("✅ 推理完成")
    print("=" * 70)


def main():
    """主函数"""
    # 模型路径（使用默认ModelScope缓存）
    model_path = os.path.expanduser(
        "~/.cache/modelscope/hub/models/XiaomiMiMo/MiMo-Audio-7B-Instruct"
    )
    tokenizer_path = os.path.expanduser(
        "~/.cache/modelscope/hub/models/XiaomiMiMo/MiMo-Audio-Tokenizer"
    )

    print("=" * 70)
    print("🎵 MiMo Audio 最小推理脚本")
    print("=" * 70)
    print(f"主模型: {model_path}")
    print(f"Tokenizer: {tokenizer_path}")
    print()

    # 加载模型
    tokenizer_model, tokenizer_config = load_audio_tokenizer(tokenizer_path)
    main_model, config, args, text_tokenizer = load_main_model(model_path)

    # 执行推理
    text_to_speak = "床前明月光，疑是地上霜。举头望明月，低头思故乡。"

    run_inference(
        main_model=main_model,
        tokenizer_model=tokenizer_model,
        text_tokenizer=text_tokenizer,
        config=config,
        args=args,
        tokenizer_config=tokenizer_config,
        text_to_speak=text_to_speak,
        max_steps=100,
    )


if __name__ == "__main__":
    main()
