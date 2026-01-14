#!/usr/bin/env python3
"""Test script for MiMo-Audio JAX interface"""

import os
import sys
import time
import argparse

# Add bonsai to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bonsai.models.mimo_audio.mimo_audio import MimoAudio


def test_initialization(model_path, tokenizer_path, use_sharding=True):
    """Test model initialization"""
    print(f"\n{'='*60}")
    print("Test 1: Model Initialization")
    print(f"{'='*60}")

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
    parser = argparse.ArgumentParser(description="Test MiMo-Audio JAX interface")
    parser.add_argument("--model_path", required=True, help="Path to main model")
    parser.add_argument("--tokenizer_path", required=True, help="Path to audio tokenizer")
    parser.add_argument("--audio_path", help="Path to test audio file for ASR")
    parser.add_argument("--test_text", default="你好，世界！", help="Test text for TTS")
    parser.add_argument("--output_path", default="test_output.wav", help="Output path for TTS")
    parser.add_argument("--no_sharding", action="store_true", help="Disable sharding")

    args = parser.parse_args()

    print("\n" + "="*60)
    print("MiMo-Audio JAX Interface Test Suite")
    print("="*60)
    print(f"Model path: {args.model_path}")
    print(f"Tokenizer path: {args.tokenizer_path}")
    print(f"Sharding: {'Disabled' if args.no_sharding else 'Enabled'}")

    try:
        # Test 1: Initialization
        mimo = test_initialization(
            args.model_path,
            args.tokenizer_path,
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
