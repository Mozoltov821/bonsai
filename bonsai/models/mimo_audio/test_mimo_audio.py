#!/usr/bin/env python3
"""Test script for MiMo-Audio JAX interface"""

import os
import sys
import time
import argparse

# Add bonsai to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bonsai.models.mimo_audio.mimo_audio import MimoAudio


def get_modelscope_cache_dir(model_id):
    """Get modelscope cache directory for a model"""
    cache_root = os.path.expanduser("~/.cache/modelscope/hub")

    # Convert model_id to cache path (e.g., "xiaomi/MiMo-Audio" -> "xiaomi/MiMo-Audio")
    model_dir = os.path.join(cache_root, model_id)

    if os.path.exists(model_dir):
        return model_dir

    # Try alternative: organization___model format
    alt_model_dir = model_id.replace("/", "___")
    alt_path = os.path.join(cache_root, alt_model_dir)
    if os.path.exists(alt_path):
        return alt_path

    raise FileNotFoundError(f"Model not found in modelscope cache: {model_id}\n"
                           f"Tried: {model_dir} and {alt_path}")


def test_initialization(model_path, tokenizer_path, use_sharding=True):
    """Test model initialization"""
    print(f"\n{'='*60}")
    print("Test 1: Model Initialization")
    print(f"{'='*60}")
    print(f"Model path: {model_path}")
    print(f"Tokenizer path: {tokenizer_path}")

    start_time = time.time()
    mimo = MimoAudio(
        model_path=model_path,
        mimo_audio_tokenizer_path=tokenizer_path,
        use_sharding=use_sharding,
    )
    load_time = time.time() - start_time

    print(f"\n✅ Model initialized successfully in {load_time:.2f}s")
    print(f"   - Group size: {mimo.group_size}")
    print(f"   - Audio channels: {mimo.audio_channels}")
    print(f"   - Vocab size: {mimo.vocab_size}")
    print(f"   - Special tokens:")
    print(f"     * SOSP: {mimo.sosp_idx}")
    print(f"     * EOSP: {mimo.eosp_idx}")
    print(f"     * SOSTM: {mimo.sostm_idx}")
    print(f"     * EOSTM: {mimo.eostm_idx}")

    return mimo


def test_text_preprocessing(mimo):
    """Test text preprocessing"""
    print(f"\n{'='*60}")
    print("Test 2: Text Preprocessing")
    print(f"{'='*60}")

    test_texts = [
        "Hello, world!",
        "你好，世界！",
        "THIS IS ALL CAPS",
        "this is all lowercase",
    ]

    for text in test_texts:
        processed = mimo.preprocess_input(text)
        print(f"   Input: {text}")
        print(f"   Output: {processed}")

    print("\n✅ Text preprocessing test passed")


def test_asr(mimo, audio_path):
    """Test ASR functionality"""
    if not os.path.exists(audio_path):
        print(f"\n⚠️  Test 3: ASR skipped (audio file not found: {audio_path})")
        return

    print(f"\n{'='*60}")
    print("Test 3: Automatic Speech Recognition")
    print(f"{'='*60}")

    print(f"   Processing audio: {audio_path}")

    start_time = time.time()
    result = mimo.asr_sft(audio_path)
    elapsed = time.time() - start_time

    print(f"\n   Transcription: {result}")
    print(f"   Time: {elapsed:.2f}s")
    print("\n✅ ASR test passed")


def test_tts(mimo, text, output_path):
    """Test TTS functionality"""
    print(f"\n{'='*60}")
    print("Test 4: Text-to-Speech")
    print(f"{'='*60}")

    print(f"   Input text: {text}")
    print(f"   Output path: {output_path}")

    start_time = time.time()
    result = mimo.tts_sft(text, output_path)
    elapsed = time.time() - start_time

    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / 1024  # KB
        print(f"\n   Generated audio: {output_path} ({file_size:.2f} KB)")
    print(f"   Text output: {result}")
    print(f"   Time: {elapsed:.2f}s")
    print("\n✅ TTS test passed")


def main():
    parser = argparse.ArgumentParser(
        description="Test MiMo-Audio JAX interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using modelscope cache (recommended)
  python test_mimo_audio.py \\
      --model_id xiaomi/MiMo-Audio \\
      --tokenizer_id xiaomi/MiMo-Audio-Tokenizer

  # Using explicit paths
  python test_mimo_audio.py \\
      --model_path /path/to/model \\
      --tokenizer_path /path/to/tokenizer

  # With audio test
  python test_mimo_audio.py \\
      --model_id xiaomi/MiMo-Audio \\
      --tokenizer_id xiaomi/MiMo-Audio-Tokenizer \\
      --audio_path test.wav
"""
    )

    # Model location arguments (use either IDs or paths)
    model_group = parser.add_argument_group('Model Location')
    model_group.add_argument("--model_id", help="ModelScope model ID (e.g., xiaomi/MiMo-Audio)")
    model_group.add_argument("--tokenizer_id", help="ModelScope tokenizer ID (e.g., xiaomi/MiMo-Audio-Tokenizer)")
    model_group.add_argument("--model_path", help="Direct path to main model (alternative to --model_id)")
    model_group.add_argument("--tokenizer_path", help="Direct path to audio tokenizer (alternative to --tokenizer_id)")

    # Test arguments
    test_group = parser.add_argument_group('Test Options')
    test_group.add_argument("--audio_path", help="Path to test audio file for ASR")
    test_group.add_argument("--test_text", default="你好，世界！", help="Test text for TTS")
    test_group.add_argument("--output_path", default="test_output.wav", help="Output path for TTS")
    test_group.add_argument("--no_sharding", action="store_true", help="Disable sharding")

    args = parser.parse_args()

    # Determine model and tokenizer paths
    if args.model_id:
        try:
            model_path = get_modelscope_cache_dir(args.model_id)
            print(f"Found model in modelscope cache: {model_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("\nPlease download the model first using:")
            print(f"  from modelscope import snapshot_download")
            print(f"  snapshot_download('{args.model_id}')")
            return 1
    elif args.model_path:
        model_path = args.model_path
    else:
        print("Error: Must specify either --model_id or --model_path")
        parser.print_help()
        return 1

    if args.tokenizer_id:
        try:
            tokenizer_path = get_modelscope_cache_dir(args.tokenizer_id)
            print(f"Found tokenizer in modelscope cache: {tokenizer_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("\nPlease download the tokenizer first using:")
            print(f"  from modelscope import snapshot_download")
            print(f"  snapshot_download('{args.tokenizer_id}')")
            return 1
    elif args.tokenizer_path:
        tokenizer_path = args.tokenizer_path
    else:
        print("Error: Must specify either --tokenizer_id or --tokenizer_path")
        parser.print_help()
        return 1

    print("\n" + "="*60)
    print("MiMo-Audio JAX Interface Test Suite")
    print("="*60)
    print(f"Sharding: {'Disabled' if args.no_sharding else 'Enabled'}")

    try:
        # Test 1: Initialization
        mimo = test_initialization(
            model_path,
            tokenizer_path,
            use_sharding=not args.no_sharding
        )

        # Test 2: Text preprocessing
        test_text_preprocessing(mimo)

        # Test 3: ASR (if audio provided)
        if args.audio_path:
            test_asr(mimo, args.audio_path)

        # Test 4: TTS
        test_tts(mimo, args.test_text, args.output_path)

        print(f"\n{'='*60}")
        print("✅ All tests passed!")
        print(f"{'='*60}\n")

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
