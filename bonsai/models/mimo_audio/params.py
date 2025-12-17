"""MiMo Audio Weight Loading - Optimized Version"""

import gc
import re
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from bonsai.models.qwen2.params import Transform, TRANSFORM_LINEAR, TRANSFORM_NONE, _stoi, _assign_weights


def _get_qwen2_key_mapping(prefix: str) -> dict[str, tuple[str, Transform]]:
    """Generate key mapping for Qwen2 submodules with prefix."""
    return {
        rf"{prefix}\.embed_tokens\.weight": ("embedder.embedding", TRANSFORM_NONE),
        rf"{prefix}\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": (r"layers.\1.attn.q_proj.kernel", TRANSFORM_LINEAR),
        rf"{prefix}\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": (r"layers.\1.attn.k_proj.kernel", TRANSFORM_LINEAR),
        rf"{prefix}\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": (r"layers.\1.attn.v_proj.kernel", TRANSFORM_LINEAR),
        rf"{prefix}\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": (r"layers.\1.attn.o_proj.kernel", TRANSFORM_LINEAR),
        rf"{prefix}\.layers\.([0-9]+)\.self_attn\.q_proj\.bias": (r"layers.\1.attn.q_proj.bias", TRANSFORM_NONE),
        rf"{prefix}\.layers\.([0-9]+)\.self_attn\.k_proj\.bias": (r"layers.\1.attn.k_proj.bias", TRANSFORM_NONE),
        rf"{prefix}\.layers\.([0-9]+)\.self_attn\.v_proj\.bias": (r"layers.\1.attn.v_proj.bias", TRANSFORM_NONE),
        rf"{prefix}\.layers\.([0-9]+)\.mlp\.gate_proj\.weight": (r"layers.\1.mlp.gate_proj.kernel", TRANSFORM_LINEAR),
        rf"{prefix}\.layers\.([0-9]+)\.mlp\.up_proj\.weight": (r"layers.\1.mlp.up_proj.kernel", TRANSFORM_LINEAR),
        rf"{prefix}\.layers\.([0-9]+)\.mlp\.down_proj\.weight": (r"layers.\1.mlp.down_proj.kernel", TRANSFORM_LINEAR),
        rf"{prefix}\.norm\.weight": ("final_norm.scale", TRANSFORM_NONE),
        rf"{prefix}\.layers\.([0-9]+)\.input_layernorm\.weight": (r"layers.\1.input_layernorm.scale", TRANSFORM_NONE),
        rf"{prefix}\.layers\.([0-9]+)\.post_attention_layernorm\.weight": (
            r"layers.\1.post_attention_layernorm.scale",
            TRANSFORM_NONE,
        ),
    }


def _get_mimo_key_mapping(audio_channels: int) -> dict[str, tuple[str, Transform]]:
    """Generate key mapping for MiMo-specific layers."""
    mapping = {
        r"lm_head\.weight": ("model.lm_head.w", TRANSFORM_LINEAR),
        r"hidden_states_downcast\.weight": ("hidden_states_downcast.kernel", TRANSFORM_LINEAR),
        r"speech_group_downcast\.weight": ("speech_group_downcast.kernel", TRANSFORM_LINEAR),
        r"speech_embeddings_to_local\.weight": ("speech_embeddings_to_local.kernel", TRANSFORM_LINEAR),
    }

    for i in range(audio_channels):
        mapping[rf"speech_embeddings\.{i}\.weight"] = (f"speech_embeddings.{i}.embedding", TRANSFORM_NONE)
        mapping[rf"local_transformer_lm_heads\.{i}\.weight"] = (f"local_transformer_lm_heads.{i}.kernel", TRANSFORM_LINEAR)

    return mapping


def _get_jax_key(
    mapping: dict[str, tuple[str, Transform]], source_key: str
) -> tuple[str | None, Transform | None]:
    """Get JAX key from source key using regex mapping."""
    for pat, (jax_key, transform) in mapping.items():
        if re.fullmatch(pat, source_key):
            return jax_key, transform
        # Also try re.match for partial matches with groups
        match = re.match(pat, source_key)
        if match:
            return re.sub(pat, jax_key, source_key), transform
    return None, None


def load_mimo_audio_weights(model, state_dict: dict):
    """Load all weights for MiMo Audio model using declarative approach."""
    print("Loading MiMo Audio weights from safetensors...")

    # Split model into graph and state
    graph_def, abs_state = nnx.split(model)
    pure_state_dict = abs_state.to_pure_dict()

    # Build comprehensive key mapping
    full_mapping = {}

    # Qwen2 main model
    full_mapping.update(_get_qwen2_key_mapping("model"))
    # Local transformer
    full_mapping.update(_get_qwen2_key_mapping("local_transformer"))
    # Input local transformer
    full_mapping.update(_get_qwen2_key_mapping("input_local_transformer"))
    # MiMo-specific layers
    full_mapping.update(_get_mimo_key_mapping(model.audio_channels))

    conversion_errors = []
    loaded_count = 0
    skipped_keys = []

    for torch_key, tensor in state_dict.items():
        jax_key, transform = _get_jax_key(full_mapping, torch_key)
        if jax_key is None:
            skipped_keys.append(torch_key)
            continue

        keys = [_stoi(k) for k in jax_key.split(".")]
        try:
            # Convert to bfloat16
            tensor_jax = jnp.asarray(tensor, dtype=jnp.bfloat16)

            # Assign to state dict (no sharding for now)
            _assign_weights(keys, tensor_jax, pure_state_dict, torch_key, transform, None)
            loaded_count += 1
        except Exception as e:
            full_jax_key = ".".join([str(k) for k in keys])
            conversion_errors.append(f"Failed to assign '{torch_key}' to '{full_jax_key}': {type(e).__name__}: {e}")

    if conversion_errors:
        print(f"Warning: {len(conversion_errors)} conversion errors occurred")
        for err in conversion_errors[:5]:  # Show first 5 errors
            print(f"  - {err}")
        if len(conversion_errors) > 5:
            print(f"  ... and {len(conversion_errors) - 5} more")

    if skipped_keys:
        print(f"Info: Skipped {len(skipped_keys)} keys not in mapping")

    # Merge back
    model_updated = nnx.merge(graph_def, pure_state_dict)

    # Copy updated state back to original model
    nnx.update(model, nnx.state(model_updated))

    gc.collect()

    print(f"MiMo Audio weights loaded successfully! ({loaded_count} parameters)")
    return model
