"""View and analyze test results from end-to-end tests."""

import os
import numpy as np


def view_saved_outputs():
    """View saved test outputs."""
    output_dir = "test_outputs"

    print("=" * 70)
    print("Viewing Test Results")
    print("=" * 70)

    # Check for audio files
    print("\n1. Audio Files:")
    original_path = os.path.join(output_dir, "original_audio.wav")
    reconstructed_path = os.path.join(output_dir, "reconstructed_audio.wav")

    if os.path.exists(original_path):
        size = os.path.getsize(original_path)
        print(f"  ✓ Original audio: {original_path} ({size/1024:.1f} KB)")
    else:
        print(f"  ✗ Original audio not found")

    if os.path.exists(reconstructed_path):
        size = os.path.getsize(reconstructed_path)
        print(f"  ✓ Reconstructed audio: {reconstructed_path} ({size/1024:.1f} KB)")
    else:
        print(f"  ✗ Reconstructed audio not found")

    # Check for model outputs
    print("\n2. Model Outputs:")
    npz_path = os.path.join(output_dir, "model_outputs.npz")

    if os.path.exists(npz_path):
        print(f"  ✓ Model outputs: {npz_path}")

        # Load and display
        data = np.load(npz_path)

        print("\n  Contents:")
        print(f"    - text_logits: shape {data['text_logits'].shape}")
        print(f"    - local_hidden_states: shape {data['local_hidden_states'].shape}")
        print(f"    - input_ids: shape {data['input_ids'].shape}")

        print("\n  Text Logits Statistics:")
        logits = data['text_logits']
        print(f"    Mean: {logits.mean():.4f}")
        print(f"    Std: {logits.std():.4f}")
        print(f"    Min: {logits.min():.4f}")
        print(f"    Max: {logits.max():.4f}")

        # Compute softmax probabilities
        from scipy.special import softmax
        probs = softmax(logits[0, 0], axis=-1)
        top_k = 10

        # Get top k
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_probs = probs[top_indices]

        print(f"\n  Top {top_k} Predicted Tokens:")
        for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
            print(f"    {i+1}. Token {idx}: {prob:.6f}")

        print("\n  Local Hidden States Statistics:")
        hidden = data['local_hidden_states']
        print(f"    Mean: {hidden.mean():.4f}")
        print(f"    Std: {hidden.std():.4f}")
        print(f"    Min: {hidden.min():.4f}")
        print(f"    Max: {hidden.max():.4f}")

        print("\n  Input IDs Sample (first 3 channels, first 10 tokens):")
        input_ids = data['input_ids']
        for ch in range(min(3, input_ids.shape[1])):
            tokens = input_ids[0, ch, :10]
            print(f"    Channel {ch}: {tokens}")

    else:
        print(f"  ✗ Model outputs not found")

    # Instructions
    print("\n" + "=" * 70)
    print("How to play audio files:")
    print("=" * 70)
    print(f"  ffplay {original_path}")
    print(f"  ffplay {reconstructed_path}")
    print("\nOr use any audio player (VLC, audacity, etc.)")


if __name__ == "__main__":
    view_saved_outputs()
