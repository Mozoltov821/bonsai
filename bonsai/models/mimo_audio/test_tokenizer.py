#!/usr/bin/env python3
"""Test script for MiMo-Audio Tokenizer"""

import os
import sys
import time
import argparse
import numpy as np

import jax
import jax.numpy as jnp
from flax import nnx
import soundfile as sf

# Add bonsai to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bonsai.models.mimo_audio.mimo_audio_tokenizer import MiMoAudioTokenizerConfig, MelSpectrogram
from bonsai.models.mimo_audio.mimo_audio_tokenizer_params import load_tokenizer_weights_from_safetensors


def get_modelscope_cache_dir(model_id):
    """Get modelscope cache directory for a model"""
    cache_root = os.path.expanduser("~/.cache/modelscope/hub")

    # Try direct path
    model_dir = os.path.join(cache_root, model_id)
    if os.path.exists(model_dir):
        return model_dir

    # Try alternative format
    alt_model_dir = model_id.replace("/", "___")
    alt_path = os.path.join(cache_root, alt_model_dir)
    if os.path.exists(alt_path):
        return alt_path

    raise FileNotFoundError(f"Model not found in modelscope cache: {model_id}")


def load_audio(audio_path, target_sr=24000):
    """Load and resample audio file"""
    print(f"\nLoading audio: {audio_path}")
    wav, sr = sf.read(audio_path)

    # Convert to mono if stereo
    if wav.ndim == 2:
        wav = np.mean(wav, axis=1)

    print(f"  - Original sample rate: {sr} Hz")
    print(f"  - Duration: {len(wav) / sr:.2f}s")
    print(f"  - Samples: {len(wav)}")

    # Simple resampling if needed
    if sr != target_sr:
        duration = len(wav) / sr
        num_samples = int(duration * target_sr)
        x_old = np.linspace(0, 1, len(wav))
        x_new = np.linspace(0, 1, num_samples)
        wav = np.interp(x_new, x_old, wav)
        print(f"  - Resampled to: {target_sr} Hz")

    return jnp.array(wav, dtype=jnp.float32)


def test_tokenizer_loading(tokenizer_path, use_sharding=True, mesh=None):
    """Test 1: Load tokenizer"""
    print(f"\n{'='*60}")
    print("Test 1: Tokenizer Loading")
    print(f"{'='*60}")
    print(f"Path: {tokenizer_path}")

    # Load config
    import json
    config_path = os.path.join(tokenizer_path, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)

    if use_sharding:
        config_dict['use_sharding'] = True

    config = MiMoAudioTokenizerConfig(**config_dict)

    print(f"\nConfig:")
    print(f"  - Encoder layers: {config.encoder_layers}")
    print(f"  - Decoder layers: {config.decoder_layers}")
    print(f"  - Vocoder layers: {config.vocoder_num_layers}")
    print(f"  - Quantizers: {config.num_quantizers}")
    print(f"  - Sample rate: {config.sampling_rate} Hz")
    print(f"  - Mel bins: {config.n_mels}")
    print(f"  - FFT size: {config.nfft}")

    # Load weights
    weights_path = os.path.join(tokenizer_path, "model.safetensors")

    start_time = time.time()
    tokenizer = load_tokenizer_weights_from_safetensors(
        config,
        weights_path,
        dtype=jnp.float32,
        mesh=mesh,
        rngs=nnx.Rngs(0)
    )
    load_time = time.time() - start_time

    print(f"\n✅ Tokenizer loaded in {load_time:.2f}s")
    if use_sharding:
        print(f"  - Sharding: Enabled")

    return tokenizer, config


def test_audio_encoding(tokenizer, config, audio_path):
    """Test 2: Encode audio to tokens"""
    print(f"\n{'='*60}")
    print("Test 2: Audio Encoding")
    print(f"{'='*60}")

    # Load audio
    wav = load_audio(audio_path, config.sampling_rate)

    # Compute mel spectrogram
    print(f"\nComputing mel spectrogram...")
    mel_transform = MelSpectrogram(
        sample_rate=config.sampling_rate,
        n_fft=config.nfft,
        hop_length=config.hop_length,
        win_length=config.window_size,
        f_min=config.fmin,
        f_max=config.fmax if config.fmax is not None else config.sampling_rate / 2.0,
        n_mels=config.n_mels,
        power=1.0,
        center=True,
    )

    mel = mel_transform(wav)  # (n_mels, time)
    mel = jnp.log(jnp.maximum(mel, 1e-7)).T  # (time, n_mels)

    print(f"  - Mel shape: {mel.shape}")
    print(f"  - Mel range: [{jnp.min(mel):.3f}, {jnp.max(mel):.3f}]")

    # Encode
    print(f"\nEncoding to tokens...")
    start_time = time.time()

    input_len = mel.shape[0]
    codes_output = tokenizer.encode(
        mel[jnp.newaxis, :, :],  # Add batch dimension
        jnp.array([input_len]),
        use_quantizer=True
    )

    encode_time = time.time() - start_time

    codes = codes_output.codes[0]  # Remove batch dimension
    print(f"  - Codes shape: {codes.shape}")
    print(f"  - Num quantizers: {codes.shape[0]}")
    print(f"  - Sequence length: {codes.shape[1]}")
    print(f"  - Code range: [{jnp.min(codes):.0f}, {jnp.max(codes):.0f}]")
    print(f"  - Encoding time: {encode_time:.3f}s")

    # Calculate compression ratio
    original_samples = len(wav)
    compressed_tokens = codes.shape[1] * codes.shape[0]
    compression_ratio = original_samples / compressed_tokens
    print(f"  - Compression ratio: {compression_ratio:.1f}x")

    print(f"\n✅ Encoding successful")

    return codes


def test_audio_decoding(tokenizer, codes, output_path):
    """Test 3: Decode tokens back to audio"""
    print(f"\n{'='*60}")
    print("Test 3: Audio Decoding")
    print(f"{'='*60}")

    print(f"Input codes shape: {codes.shape}")

    # Decode
    print(f"\nDecoding tokens to audio...")
    start_time = time.time()

    wav_decoded = tokenizer.decode(codes[jnp.newaxis, :, :])  # Add batch dimension
    wav_decoded = wav_decoded[0]  # Remove batch dimension

    decode_time = time.time() - start_time

    print(f"  - Output shape: {wav_decoded.shape}")
    print(f"  - Output range: [{jnp.min(wav_decoded):.3f}, {jnp.max(wav_decoded):.3f}]")
    print(f"  - Decoding time: {decode_time:.3f}s")

    # Save output
    wav_np = np.array(wav_decoded).reshape(-1)
    sf.write(output_path, wav_np, 24000)
    file_size = os.path.getsize(output_path) / 1024  # KB

    print(f"\n✅ Decoding successful")
    print(f"  - Output saved: {output_path} ({file_size:.2f} KB)")

    return wav_decoded


def test_roundtrip(wav_original, wav_decoded):
    """Test 4: Compare original and reconstructed audio"""
    print(f"\n{'='*60}")
    print("Test 4: Roundtrip Quality")
    print(f"{'='*60}")

    # Trim to same length
    min_len = min(len(wav_original), len(wav_decoded))
    wav_original = wav_original[:min_len]
    wav_decoded = wav_decoded[:min_len]

    # Calculate metrics
    mse = jnp.mean((wav_original - wav_decoded) ** 2)
    rmse = jnp.sqrt(mse)

    # Signal-to-noise ratio
    signal_power = jnp.mean(wav_original ** 2)
    noise_power = jnp.mean((wav_original - wav_decoded) ** 2)
    snr = 10 * jnp.log10(signal_power / (noise_power + 1e-8))

    # Correlation
    correlation = jnp.corrcoef(wav_original, wav_decoded)[0, 1]

    print(f"Quality metrics:")
    print(f"  - MSE: {mse:.6f}")
    print(f"  - RMSE: {rmse:.6f}")
    print(f"  - SNR: {snr:.2f} dB")
    print(f"  - Correlation: {correlation:.4f}")

    # Quality assessment
    if snr > 20:
        quality = "Excellent"
    elif snr > 15:
        quality = "Good"
    elif snr > 10:
        quality = "Fair"
    else:
        quality = "Poor"

    print(f"\n✅ Overall quality: {quality}")

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "snr": float(snr),
        "correlation": float(correlation)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test MiMo-Audio Tokenizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using modelscope cache
  python test_tokenizer.py \\
      --tokenizer_id xiaomi/MiMo-Audio-Tokenizer \\
      --audio_path input.wav

  # Using explicit path
  python test_tokenizer.py \\
      --tokenizer_path /path/to/tokenizer \\
      --audio_path input.wav

  # Without sharding
  python test_tokenizer.py \\
      --tokenizer_id xiaomi/MiMo-Audio-Tokenizer \\
      --audio_path input.wav \\
      --no_sharding
"""
    )

    # Tokenizer location
    tokenizer_group = parser.add_argument_group('Tokenizer Location')
    tokenizer_group.add_argument("--tokenizer_id", help="ModelScope tokenizer ID")
    tokenizer_group.add_argument("--tokenizer_path", help="Direct path to tokenizer")

    # Test options
    parser.add_argument("--audio_path", required=True, help="Input audio file")
    parser.add_argument("--output_path", default="reconstructed.wav", help="Output audio file")
    parser.add_argument("--no_sharding", action="store_true", help="Disable sharding")

    args = parser.parse_args()

    # Determine tokenizer path
    if args.tokenizer_id:
        try:
            tokenizer_path = get_modelscope_cache_dir(args.tokenizer_id)
            print(f"Found tokenizer: {tokenizer_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return 1
    elif args.tokenizer_path:
        tokenizer_path = args.tokenizer_path
    else:
        print("Error: Must specify --tokenizer_id or --tokenizer_path")
        parser.print_help()
        return 1

    # Check audio file
    if not os.path.exists(args.audio_path):
        print(f"Error: Audio file not found: {args.audio_path}")
        return 1

    print("\n" + "="*60)
    print("MiMo-Audio Tokenizer Test Suite")
    print("="*60)
    print(f"Input audio: {args.audio_path}")
    print(f"Output audio: {args.output_path}")
    print(f"Sharding: {'Disabled' if args.no_sharding else 'Enabled'}")

    try:
        # Setup sharding
        mesh = None
        use_sharding = not args.no_sharding

        if use_sharding:
            devices = jax.devices()
            num_devices = len(devices)
            print(f"Available devices: {num_devices}")

            if num_devices >= 2:
                mesh_shape = (1, num_devices)
                devices_reshaped = np.array(devices).reshape(mesh_shape)
                mesh = jax.sharding.Mesh(devices_reshaped, ('fsdp', 'tp'))
                print(f"Created mesh: fsdp=1, tp={num_devices}")
            else:
                print("Single device, sharding disabled")
                use_sharding = False

        # Test 1: Load tokenizer
        tokenizer, config = test_tokenizer_loading(
            tokenizer_path,
            use_sharding=use_sharding,
            mesh=mesh
        )

        # Test 2: Encode audio
        codes = test_audio_encoding(tokenizer, config, args.audio_path)

        # Test 3: Decode audio
        wav_decoded = test_audio_decoding(tokenizer, codes, args.output_path)

        # Test 4: Roundtrip quality
        wav_original = load_audio(args.audio_path, config.sampling_rate)
        metrics = test_roundtrip(wav_original, wav_decoded)

        # Summary
        print(f"\n{'='*60}")
        print("✅ All tests passed!")
        print(f"{'='*60}")
        print(f"\nSummary:")
        print(f"  - Input: {args.audio_path}")
        print(f"  - Output: {args.output_path}")
        print(f"  - Quantizers: {config.num_quantizers}")
        print(f"  - Quality (SNR): {metrics['snr']:.2f} dB")
        print()

        return 0

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ Test failed: {type(e).__name__}: {e}")
        print(f"{'='*60}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
